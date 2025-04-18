use std::collections::btree_map::Entry as BTreeMapEntry;
use std::collections::hash_map::Entry as HashMapEntry;
use std::collections::{BTreeMap, HashMap};
use std::hash::{Hash, Hasher};

use line_numbers::LineNumber;
use serde::{Serialize, Serializer, ser::SerializeStruct};

use crate::{
    display::{
        context::{all_matched_lines_filled, opposite_positions},
        hunks::{matched_lines_indexes_for_hunk, matched_pos_to_hunks, merge_adjacent},
        side_by_side::lines_with_novel,
    },
    lines::MaxLine,
    parse::syntax::{self, MatchedPos},
    summary::{DiffResult, FileContent, FileFormat},
};

#[derive(Debug, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum Status {
    Unchanged,
    Changed,
    Created,
    Deleted,
}

#[derive(Debug)]
struct File<'f> {
    language: &'f FileFormat,
    path: &'f str,
    chunks: Vec<Vec<Line<'f>>>,
    status: Status,
}

impl<'f> File<'f> {
    fn with_sections(language: &'f FileFormat, path: &'f str, chunks: Vec<Vec<Line<'f>>>) -> File<'f> {
        File {
            language,
            path,
            chunks,
            status: Status::Changed,
        }
    }

    fn with_status(language: &'f FileFormat, path: &'f str, status: Status) -> File<'f> {
        File {
            language,
            path,
            chunks: Vec::new(),
            status,
        }
    }
}

impl<'f> From<&'f DiffResult> for File<'f> {
    // This function converts a DiffResult (containing information about file differences, including source content and
    // matched positions) into a File struct suitable for JSON serialization. It calculates hunks of changes, filters
    // relevant lines, and delegates the extraction of specific changes within lines to add_changes_to_side.
    fn from(summary: &'f DiffResult) -> Self {
        match (&summary.lhs_src, &summary.rhs_src) {
            (FileContent::Text(lhs_src), FileContent::Text(rhs_src)) => {
                // TODO: move into function as it is effectively duplicates lines 365-375 of main::print_diff_result
                let opposite_to_lhs = opposite_positions(&summary.lhs_positions);
                let opposite_to_rhs = opposite_positions(&summary.rhs_positions);

                let hunks = matched_pos_to_hunks(&summary.lhs_positions, &summary.rhs_positions);
                let hunks = merge_adjacent(
                    &hunks,
                    &opposite_to_lhs,
                    &opposite_to_rhs,
                    lhs_src.max_line(),
                    rhs_src.max_line(),
                    0,
                );

                if hunks.is_empty() {
                    return File::with_status(&summary.file_format, &summary.display_path, Status::Unchanged);
                }

                if lhs_src.is_empty() {
                    return File::with_status(&summary.file_format, &summary.display_path, Status::Created);
                }
                if rhs_src.is_empty() {
                    return File::with_status(&summary.file_format, &summary.display_path, Status::Deleted);
                }

                let lhs_lines = lhs_src.split('\n').collect::<Vec<&str>>();
                let rhs_lines = rhs_src.split('\n').collect::<Vec<&str>>();

                let (_, rhs_lines_with_novel) = lines_with_novel(&summary.lhs_positions, &summary.rhs_positions);

                let matched_lines =
                    all_matched_lines_filled(&summary.lhs_positions, &summary.rhs_positions, &lhs_lines, &rhs_lines);
                let mut matched_lines = &matched_lines[..];

                // `lines_for_all_chunks` will be used for deduplication lookups. Keep using `HashMap` as it offers
                // average O(1) lookups/insertions compared to BTreeMap's O(log N).
                let mut lines_for_all_chunks: HashMap<u32, AllChunks> = HashMap::new();

                let mut chunks = Vec::with_capacity(hunks.len());
                for hunk in &hunks {
                    // Sorted iteration is necessary for `lines`. Keep using `BTreeMap` here.
                    let mut lines: BTreeMap<Option<u32>, Line<'f>> = BTreeMap::new();

                    let (start_i, end_i) = matched_lines_indexes_for_hunk(matched_lines, hunk, 0);
                    let aligned_lines = &matched_lines[start_i..end_i];
                    matched_lines = &matched_lines[start_i..];
                    // Efficiently advance the slice view for the next iteration
                    // matched_lines = &matched_lines[end_i..]; // Corrected: Use end_i to avoid reprocessing

                    for (_, rhs_line_num) in aligned_lines {
                        if !rhs_lines_with_novel.contains(&rhs_line_num.unwrap_or(LineNumber(0))) {
                            continue;
                        }

                        if let Some(line_num) = rhs_line_num {
                            add_changes_to_side(
                                &mut lines,
                                *line_num,
                                &rhs_lines,
                                &summary.rhs_positions,
                                &mut lines_for_all_chunks,
                            );
                        }
                    }

                    // If changes were added to `lines` for this hunk, collect them.
                    // BTreeMap ensures they are collected in line number order.
                    if !lines.is_empty() {
                        chunks.push(lines.into_values().collect());
                    }
                }

                File::with_sections(&summary.file_format, &summary.display_path, chunks)
            }
            (FileContent::Binary, FileContent::Binary) => {
                let status = if summary.has_byte_changes {
                    Status::Changed
                } else {
                    Status::Unchanged
                };
                File::with_status(&FileFormat::Binary, &summary.display_path, status)
            }
            (_, FileContent::Binary) | (FileContent::Binary, _) => {
                File::with_status(&FileFormat::Binary, &summary.display_path, Status::Changed)
            }
        }
    }
}

impl Serialize for File<'_> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // equivalent to #[serde(skip_serializing_if = "Vec::is_empty")]
        let mut file = if self.chunks.is_empty() {
            serializer.serialize_struct("File", 3)?
        } else {
            let mut file = serializer.serialize_struct("File", 4)?;
            file.serialize_field("chunks", &self.chunks)?;
            file
        };

        file.serialize_field("language", &format!("{}", self.language))?;
        file.serialize_field("path", &self.path)?;
        file.serialize_field("status", &self.status)?;

        file.end()
    }
}

#[derive(Debug, Serialize)]
struct Line<'l> {
    #[serde(skip_serializing_if = "Option::is_none")]
    rhs: Option<Side<'l>>,
}

impl<'l> Line<'l> {
    fn new(rhs_number: Option<u32>) -> Line<'l> {
        Line {
            rhs: rhs_number.map(Side::new),
        }
    }
}

#[derive(Debug, Serialize)]
struct Side<'s> {
    line_number: u32,
    changes: Vec<Change2<'s>>,
}

impl<'s> Side<'s> {
    fn new(line_number: u32) -> Side<'s> {
        Side {
            line_number,
            changes: Vec::new(),
        }
    }
}

#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
struct ChangeKey {
    start: u32,
    end: u32,
}

struct AllChunks {
    // Stores lightweight keys for O(1) average time complexity duplicate checks.
    change_keys: std::collections::HashSet<ChangeKey>,
}

impl AllChunks {
    fn new() -> AllChunks {
        AllChunks {
            change_keys: std::collections::HashSet::new(), // Initialize HashSet
        }
    }
}

#[derive(Debug, Serialize)]
struct Change2<'c> {
    start: u32,
    end: u32,
    content: &'c str,
    highlight_type: &'c syntax::MatchKind,
}

pub(crate) fn print_directory(diffs: Vec<DiffResult>, print_unchanged: bool) {
    let files = diffs
        .iter()
        .map(File::from)
        .filter(|f| print_unchanged || f.status != Status::Unchanged)
        .collect::<Vec<File>>();
    println!("{}", serde_json::to_string(&files).expect("failed to serialize files"));
}

pub(crate) fn print(diff: &DiffResult) {
    let file = File::from(diff);
    println!("{}", serde_json::to_string(&file).expect("failed to serialize file"))
}

fn add_changes_to_side<'s>(
    lines: &mut BTreeMap<Option<u32>, Line<'s>>,
    line_num: LineNumber,
    src_lines: &[&'s str],
    all_matches: &'s [MatchedPos],
    lines_for_all_chunks: &mut HashMap<u32, AllChunks>,
) {
    use syntax::MatchKind;
    // Ensure line_num is valid before indexing
    let line_idx = line_num.0 as usize;
    if line_idx >= src_lines.len() {
        eprintln!("Warning: Invalid line number {} encountered.", line_num.0);
        return;
    }
    let src_line = src_lines[line_idx];

    // Get matches relevant to this line that are considered novel
    let matches = matches_for_line(all_matches, line_num);

    let mut iter = matches.into_iter().peekable();
    while let Some(m) = iter.next() {
        // Ignore specified kinds.
        match m.kind {
            MatchKind::UnchangedPartOfNovelItem { .. } | MatchKind::UnchangedToken { .. } => {
                continue; // Skip deliberately ignored kinds
            }
            _ => {} // Process other kinds allowed by matches_for_line
        }

        let change_to_add: Change2<'s>;
        let change_key: ChangeKey; // Use the lightweight key for lookups

        //  Merge Novel kinds before deduplication check
        if matches!(m.kind, MatchKind::Novel { .. }) {
            let current_start = m.pos.start_col;
            let mut current_end = m.pos.end_col;
            let highlight_type_ref = &m.kind; // Use kind from the *first* item in the merged sequence

            // Peek ahead and merge consecutive Novel items
            while let Some(next_m) = iter.peek() {
                if matches!(next_m.kind, MatchKind::Novel { .. }) {
                    // Extend the range to the end of the next item
                    current_end = next_m.pos.end_col;
                    // Consume the peeked item as it's now part of the merged range
                    iter.next();
                } else {
                    break; // The next item is not a Novel item, stop merging
                }
            }

            change_key = ChangeKey {
                start: current_start,
                end: current_end,
            };
            change_to_add = Change2 {
                start: current_start,
                end: current_end,
                content: &src_line[(current_start as usize)..(current_end as usize)],
                highlight_type: highlight_type_ref,
            };
        } else {
            // This match is not MatchKind::Novel (e.g., could be NovelWord). Add it individually.
            let start_idx = m.pos.start_col;
            let end_idx = m.pos.end_col;

            change_key = ChangeKey {
                start: start_idx,
                end: end_idx,
            };
            change_to_add = Change2 {
                start: start_idx,
                end: end_idx,
                content: &src_line[(start_idx as usize)..(end_idx as usize)],
                highlight_type: &m.kind,
            };
        }

        let line_entry = lines_for_all_chunks.entry(line_num.0);
        let all_chunks_for_line = match line_entry {
            // Use HashMap's Entry API
            HashMapEntry::Occupied(occupied_entry) => occupied_entry.into_mut(),
            HashMapEntry::Vacant(vacant_entry) => vacant_entry.insert(AllChunks::new()),
        };

        // HashSet::insert returns true if value was not present
        if all_chunks_for_line.change_keys.insert(change_key) {
            let line = lines
                .entry(Some(line_num.0))
                .or_insert_with(|| Line::new(Some(line_num.0)));

            line.rhs.as_mut().unwrap().changes.push(change_to_add);
        }
    }
}

fn matches_for_line(matches: &[MatchedPos], line_num: LineNumber) -> Vec<&MatchedPos> {
    // This implementation is reasonably clear. While it iterates twice (filter, filter)
    // and allocates a Vec, optimizing it further might add complexity for potentially
    // minor gains unless `all_matches` is enormous and this function is extremely hot.
    // The current approach is likely fine.
    matches
        .iter()
        .filter(|m| m.pos.line == line_num)
        .filter(|m| m.kind.is_novel())
        .collect()
}
