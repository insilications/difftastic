use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, RwLock},
};

use anyhow::{Context, Result};
use git2::{DiffOptions, ObjectType, Oid, Repository, Sort};

/// Return all commits at/behind `rev` that modified `path`, newest first.
/// Each entry is (summary, oid, *optional* `file_content`).
///
/// `file_content` is
///   • `Some(Arc<str>)` – file existed in that commit (content is shared),
///   • `None`           – the commit deleted the file.
fn commits_touching_path(repo: &Repository, rev: &str, path: &Path) -> Result<Vec<(String, Oid, Option<Arc<str>>)>> {
    let start_oid = repo.revparse_single(rev)?.id();

    let mut walk = repo.revwalk()?;
    walk.set_sorting(Sort::TIME | Sort::TOPOLOGICAL)?;
    walk.push(start_oid)?;

    let mut diff_opts = DiffOptions::new();
    // libgit2 needs a UTF-8 pathspec
    let pathspec = path
        .to_str() // Fallible but zero-cost if valid. Or use `to_string_lossy()`
        .context(format!(
            "Path {} is not valid UTF-8 (libgit2 requires UTF-8 pathspecs)",
            path.display()
        ))?;
    diff_opts.pathspec(pathspec);

    let mut out = Vec::new();

    for oid in walk {
        let oid = oid?;
        let commit = repo.find_commit(oid)?;
        let this_tree = commit.tree()?;

        // ── Did the commit touch the given path? ───────────────────────────────
        let touched = if commit.parent_count() == 0 {
            repo.diff_tree_to_tree(None, Some(&this_tree), Some(&mut diff_opts))?
                .deltas()
                .len()
                > 0
        } else {
            commit.parents().any(|p| {
                repo.diff_tree_to_tree(Some(&p.tree().unwrap()), Some(&this_tree), Some(&mut diff_opts))
                    .map(|d| d.deltas().len() > 0)
                    .unwrap_or(false)
            })
        };
        // ───────────────────────────────────────────────────────────────────────

        if touched {
            // 1) Commit summary (unchanged from before)
            let summary = commit
                .summary()
                .map_or_else(|| "<no subject>".into(), std::borrow::ToOwned::to_owned);
            // let summary = commit.summary().map_or_else(|| "<no subject>".into(), |s| s.to_owned());
            // let summary = commit
            //     .summary()
            //     .map(|s| s.to_owned())
            //     .unwrap_or_else(|| "<no subject>".into());

            // 2) Fetch the file contents, if the blob still exists
            let file_content: Option<Arc<str>> = match this_tree.get_path(path) {
                Ok(entry) if entry.kind() == Some(ObjectType::Blob) => {
                    let blob = repo.find_blob(entry.id())?;
                    // ??????????????????????????????????????????????????????
                    // Allocate exactly once and share via Arc
                    Some(Arc::<str>::from(String::from_utf8_lossy(blob.content()).into_owned()))
                }
                // Deleted / missing file
                _ => None,
            };

            // 3) Push the triple (summary, oid, content)
            out.push((summary, oid, file_content));
        }
    }
    Ok(out)
}

// ────────────────────────────────────────────────────────────────────────────────
//  1. Domain types
// ────────────────────────────────────────────────────────────────────────────────
//

/// A *strongly-typed* git commit object id (20 raw bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CommitId([u8; 20]);

impl CommitId {
    /// Create a `CommitId` from the usual 40-character hexadecimal SHA-1 string.
    ///
    /// NOTE: This function does *no* full error recovery – it just bubbles
    ///       back a `String` with the problem description because the purpose
    ///       here is to keep the example dependency-free.
    pub fn from_hex(hex: &str) -> Result<Self, String> {
        if hex.len() != 40 {
            return Err(format!("expected 40 hex chars, got {}", hex.len()));
        }

        let mut bytes = [0u8; 20];
        for (i, chunk) in hex.as_bytes().chunks_exact(2).enumerate() {
            let h = char::from(chunk[0]).to_digit(16).ok_or("invalid hex")?;
            let l = char::from(chunk[1]).to_digit(16).ok_or("invalid hex")?;
            bytes[i] = ((h << 4) + l) as u8;
        }
        Ok(Self(bytes))
    }

    /// A traditional “short” (7-char) textual representation – handy for logs.
    #[allow(dead_code)]
    pub fn short(&self) -> String {
        self.0
            .iter()
            .take(4) // 4 × 2 hex digits = 8 chars – slice after join.
            .flat_map(|b| format!("{b:02x}").chars().collect::<Vec<_>>())
            .take(7)
            .collect()
    }
}

// ────────────────────────────────────────────────────────────────────────────────
//  2. Two indices on top of those domain types
// ────────────────────────────────────────────────────────────────────────────────
//

/// We store revspecs (“HEAD~1”, “v1.2.3”, …) exactly as typed by the user.
/// If the set is small/repeated you *could* intern them.
type RevSpec = String;

/// An *interned* absolute or repository-relative path.
///
/// Using `Arc` means every unique path lives only once in memory, even when
/// referenced by thousands of commits.
type FilePath = Arc<PathBuf>;

/// What you actually want to know about a given file *version*.
// Make sure FileVersion derives Clone if using the .map(|(...) (...).clone()) approach in lookup_version
#[derive(Clone, Debug)] // Ensure FileVersion is Cloneable
pub struct FileVersion {
    /// Full file contents at this revision.
    pub content: Arc<str>,
    /// First line of the commit message (“commit summary”).
    pub summary: Arc<str>,
}

/// A unique key for a `FileVersion`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VersionKey {
    commit: CommitId,
    path: FilePath,
}

/// (commit id, path) ➜ `FileVersion`
type VersionStore = HashMap<VersionKey, FileVersion>;

/// For one file:  revspec ➜ commit id
type RevIndexPerPath = HashMap<RevSpec, CommitId>;

/// Top-level:  path ➜ (revspec ➜ commit id)
type RevStore = HashMap<FilePath, RevIndexPerPath>;

// Define the stores using Mutex/RwLock
// Use RwLock if reads are much more frequent than writes
type SharedVersionStore = Arc<RwLock<HashMap<VersionKey, FileVersion>>>;
type SharedRevStore = Arc<RwLock<HashMap<FilePath, RevIndexPerPath>>>;
type SharedRepo = Arc<Mutex<Repository>>;

// ────────────────────────────────────────────────────────────────────────────────
//  3. Helper functions
// ────────────────────────────────────────────────────────────────────────────────
//

/// Insert (or overwrite) one complete `FileVersion` and keep the two indices in sync.
///
/// Because we hand out `Arc`s, the *data* is never copied – only the 8-byte
/// pointer inside each `Arc` is cloned.
fn put_version(
    versions: &mut VersionStore,
    revs: &mut RevStore,
    path: PathBuf,
    commit: CommitId,
    revspec: RevSpec,
    content: &Arc<str>,
    summary: &Arc<str>,
    // content: Arc<str>
    // summary: Arc<str>
) {
    // 1) Create one shared PathBuf
    let path_arc = Arc::new(path);

    // 2) Primary store (commit, path) ➜ payload
    versions.insert(
        VersionKey {
            commit,
            path: Arc::clone(&path_arc),
        },
        FileVersion {
            // Just clone the Arcs – no re-allocation
            content: Arc::clone(content),
            summary: Arc::clone(summary),
        },
    );

    // 3) Secondary store  path ➜ (revspec ➜ commit)
    revs.entry(Arc::clone(&path_arc)).or_default().insert(revspec, commit);
}

/// Resolve (“HEAD~2”, some/path) → (commit-id, &`FileVersion`)
///
/// Returns `None` when either the path or the revspec is unknown.
fn lookup<'a>(
    versions: &'a VersionStore,
    revs: &'a RevStore,
    path: &Path,
    revspec: &str,
) -> Option<(CommitId, &'a FileVersion)> {
    // 1) Which commit belongs to that (path, revspec)?
    // The rev-store tells us which *commit* corresponds to that revspec for that (canonical) path.
    // Only a PathBuf clone, no Arc allocation
    let path_buf = path.to_path_buf(); // Path → owned PathBuf
    let (path_canonical, per_path_index) = revs.get_key_value(&path_buf)?;
    let commit_id = *per_path_index.get(revspec)?;

    // 2) Primary store maps that (commit, path) to the payload.
    versions
        .get(&VersionKey {
            commit: commit_id,
            path: Arc::clone(path_canonical),
        })
        .map(|fv| (commit_id, fv)) // <- return both
}

/// Iterate through recorded revspecs in `RevStore` for a particular path
#[allow(dead_code)]
fn iterate_lookup(versions: &VersionStore, revs: &RevStore, path: &Path) {
    // allocate only the PathBuf needed for the HashMap probe
    let path_buf = path.to_path_buf();

    let Some((path_canonical, per_path_index)) = revs.get_key_value(&path_buf) else {
        tracing::error!("No information stored for path {:?}", path);
        return; // Nothing to do
    };

    tracing::info!("Path           : {}\n", path.display());
    for (revspec, commit_id) in per_path_index {
        if let Some(version) = versions.get(&VersionKey {
            commit: *commit_id,
            path: Arc::clone(path_canonical),
        }) {
            tracing::info!("Revspec        : {revspec}");
            tracing::info!("Summary        : {}", version.summary);
            tracing::info!("Content Length : {}\n", version.content.len());
        }
    }
}

#[derive(Clone)]
pub struct CacheStateShared {
    repo: Option<SharedRepo>, // `SharedRepo` = `Arc<Mutex<Repository>>`. `Repository` is `Send`, but not `Sync`.
    versions: SharedVersionStore,
    revs: SharedRevStore,
}

impl CacheStateShared {
    pub fn new() -> Self {
        Self {
            repo: None, // Set to None initially.
            // Initialize empty HashMaps inside RwLock and Arc
            versions: Arc::new(RwLock::new(HashMap::new())),
            revs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    pub fn set_repo(&mut self, repo_path: &Path) -> Result<()> {
        let repo = Repository::open(repo_path)?;
        self.repo = Some(Arc::new(Mutex::new(repo)));
        Ok(())
    }

    // Method to populate - takes &self because mutation happens *inside* the locks
    pub fn populate_history(&self, rev: &str, path: &Path) -> Result<()> {
        let history = {
            // let repo = self.repo.as_ref().unwrap();
            let repo_guard = self.repo.as_ref().unwrap().lock().expect("Repo mutex poisoned");
            commits_touching_path(&*repo_guard, rev, path)?
        }; // ← repo_guard lock released right here
        {
            // Maybe use .write().map_err(...) for better error handling.
            let mut versions_guard = self.versions.write().expect("Version store lock poisoned");
            let mut revs_guard = self.revs.write().expect("Rev store lock poisoned");

            let mut index = 1;
            for (summary_str, oid, maybe_content) in history {
                let commit = CommitId::from_hex(&oid.to_string()).map_err(anyhow::Error::msg)?;
                let revspec = format!("HEAD~{index}");
                let content_arc = maybe_content.unwrap_or_else(|| Arc::<str>::from(""));
                let summary_arc = Arc::<str>::from(summary_str);

                // Pass mutable references obtained from the lock guards
                put_version(
                    &mut *versions_guard, // Dereference the guard
                    &mut *revs_guard,     // Dereference the guard
                    path.to_path_buf(),
                    commit,
                    revspec,
                    &content_arc,
                    &summary_arc,
                );
                index += 1;
            }
            // Locks are automatically released when versions_guard and revs_guard go out of scope
        }
        Ok(())
    }

    // Method to lookup - takes &self, uses read locks
    pub fn lookup_version(&self, path: &Path, revspec: &str) -> Option<(CommitId, FileVersion)> {
        // Acquire read locks - multiple readers can coexist
        let versions_guard = self.versions.read().expect("Version store lock poisoned");
        let revs_guard = self.revs.read().expect("Rev store lock poisoned");

        // Call the standalone lookup function with immutable references from guards
        // We need to clone the FileVersion because the reference (&'a FileVersion)
        // returned by lookup is tied to the lifetime of the lock guard.
        // Returning the owned FileVersion avoids lifetime issues.
        lookup(&*versions_guard, &*revs_guard, path, revspec)
            .map(|(commit_id, file_version_ref)| (commit_id, file_version_ref.clone())) // Clone FileVersion
    }

    /// Method to iterate - takes &self, uses read locks
    #[allow(dead_code)]
    pub fn iterate_path_versions(&self, path: &Path) {
        let versions_guard = self.versions.read().expect("Version store lock poisoned");
        let revs_guard = self.revs.read().expect("Rev store lock poisoned");
        // Note: iterate_lookup prints directly, so it doesn't return references
        // tied to the lock guard's lifetime, which is fine here.
        iterate_lookup(&*versions_guard, &*revs_guard, path);
    }
}

impl std::fmt::Debug for CacheStateShared {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Start building the debug representation for a struct.
        // The string "CacheStateShared" is the name of the struct.
        let mut ds = f.debug_struct("CacheStateShared");

        // Handle the 'repo' field: Option<Arc<Mutex<Repository>>>
        match &self.repo {
            Some(repo_mutex_arc) => {
                // The repo is present, try to lock the Mutex.
                match repo_mutex_arc.lock() {
                    Ok(repo_guard) => {
                        // Successfully locked. git2::Repository doesn't implement Debug.
                        // We'll print its path. repo_guard.path() returns a &Path,
                        // which is Debug.
                        ds.field("repo", &format_args!("Some(Repository path: {:?})", repo_guard.path()));
                    }
                    Err(_) => {
                        // The Mutex is poisoned. This means a thread panicked while holding the lock.
                        // It's good practice for Debug not to panic itself.
                        ds.field("repo", &"Some(Repository <Mutex poisoned>)");
                    }
                }
            }
            None => {
                // The repo is not set.
                ds.field("repo", &"None");
            }
        }

        // Handle the 'versions' field: Arc<RwLock<HashMap<VersionKey, FileVersion>>>
        match self.versions.read() {
            Ok(versions_guard) => {
                // Successfully acquired a read lock.
                // HashMap<VersionKey, FileVersion> implements Debug if VersionKey and FileVersion do.
                // (VersionKey, FileVersion, CommitId, FilePath all derive Debug or are Debug).
                ds.field("versions", &*versions_guard);
            }
            Err(_) => {
                // The RwLock is poisoned.
                ds.field("versions", &"<RwLock versions poisoned>");
            }
        }

        // Handle the 'revs' field: Arc<RwLock<HashMap<FilePath, RevIndexPerPath>>>
        match self.revs.read() {
            Ok(revs_guard) => {
                // Successfully acquired a read lock.
                // HashMap<FilePath, RevIndexPerPath> implements Debug.
                // (FilePath, RevIndexPerPath, RevSpec, CommitId all derive Debug or are Debug).
                ds.field("revs", &*revs_guard);
            }
            Err(_) => {
                // The RwLock is poisoned.
                ds.field("revs", &"<RwLock revs poisoned>");
            }
        }

        // Finalize the debug struct representation.
        ds.finish()
    }
}
