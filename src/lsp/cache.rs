use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, RwLock},
};

use anyhow::{Context, Result};
use git2::{DiffOptions, ObjectType, Oid, Repository, Sort};

/// Return all commits at/behind `rev` that modified `path`, newest first.
/// Each entry is (summary, oid, *optional* file_content).
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
            "Path {:?} is not valid UTF-8 (libgit2 requires UTF-8 pathspecs)",
            path
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
                .map(|s| s.to_owned())
                .unwrap_or_else(|| "<no subject>".into());

            // 2) Fetch the file contents, if the blob still exists
            let file_content: Option<Arc<str>> = match this_tree.get_path(path) {
                Ok(entry) if entry.kind() == Some(ObjectType::Blob) => {
                    let blob = repo.find_blob(entry.id())?;
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
struct CommitId([u8; 20]);

impl CommitId {
    /// Create a `CommitId` from the usual 40-character hexadecimal SHA-1 string.
    ///
    /// NOTE: This function does *no* full error recovery – it just bubbles
    ///       back a `String` with the problem description because the purpose
    ///       here is to keep the example dependency-free.
    fn from_hex(hex: &str) -> Result<Self, String> {
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
    fn short(&self) -> String {
        self.0
            .iter()
            .take(4) // 4 × 2 hex digits = 8 chars – slice after join.
            .flat_map(|b| format!("{:02x}", b).chars().collect::<Vec<_>>())
            .take(7)
            .collect()
    }
}

/// We store revspecs (“HEAD~1”, “v1.2.3”, …) exactly as typed by the user.
/// If the set is small/repeated you *could* intern them.
type RevSpec = String;

/// An *interned* absolute or repository-relative path.
///
/// Using `Arc` means every unique path lives only once in memory, even when
/// referenced by thousands of commits.
type FilePath = Arc<PathBuf>;

/// What you actually want to know about a given file *version*.
#[derive(Clone, Debug)]
struct FileVersion {
    /// Full file contents at this revision.
    content: Arc<str>,
    /// First line of the commit message (“commit summary”).
    summary: Arc<str>,
}

/// A unique key for a `FileVersion`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct VersionKey {
    commit: CommitId,
    path: FilePath,
}

// ────────────────────────────────────────────────────────────────────────────────
//  2. Two indices on top of those domain types
// ────────────────────────────────────────────────────────────────────────────────
//

/// (commit, path) ➜ `FileVersion`
type VersionStore = HashMap<VersionKey, FileVersion>;

/// For one file:  revspec ➜ commit id
type RevIndexPerPath = HashMap<RevSpec, CommitId>;

/// Top-level:  path ➜ (revspec ➜ commit)
type RevStore = HashMap<FilePath, RevIndexPerPath>;

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
    content: Arc<str>, // was String
    summary: Arc<str>, // was String
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
            content: Arc::clone(&content),
            summary: Arc::clone(&summary),
        },
    );

    // 3) Secondary store  path ➜ (revspec ➜ commit)
    revs.entry(Arc::clone(&path_arc)).or_default().insert(revspec, commit);
}

/// Resolve (“HEAD~2”, some/path) → (commit-id, &FileVersion)
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
    // For simplicity we create a temporary Arc<PathBuf> for lookup.
    let path_tmp = Arc::new(path.to_path_buf());
    let (path_canonical, per_path_index) = revs.get_key_value(&path_tmp)?;
    let commit_id = *per_path_index.get(revspec)?;

    // 2) Primary store maps that (commit, path) to the payload.
    versions
        .get(&VersionKey {
            commit: commit_id,
            path: Arc::clone(path_canonical),
        })
        .map(|fv| (commit_id, fv)) // <- return both
}

/// Iterate through recorded revspecs in RevStore for a particular path
fn iterate_lookup(versions: &VersionStore, revs: &RevStore, path: &Path) {
    let path_tmp = Arc::new(path.to_path_buf());

    let Some((path_canonical, per_path_index)) = revs.get_key_value(&path_tmp) else {
        eprintln!("No information stored for path {:?}", path);
        return; // Nothing to do
    };

    println!("Path           : {}\n", path.display());
    for (revspec, commit_id) in per_path_index {
        if let Some(version) = versions.get(&VersionKey {
            commit: *commit_id,
            path: Arc::clone(path_canonical),
        }) {
            println!("Revspec        : {revspec}");
            println!("Summary        : {}", version.summary);
            println!("Content Length : {}\n", version.content.len());
        }
    }
}
