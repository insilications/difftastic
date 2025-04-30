use std::{
    collections::{BTreeMap, HashMap, hash_map::Entry as HashMapEntry},
    hash::Hash,
};

use line_numbers::LineNumber;
use lsp_types::{Position, Range};
use serde::Serialize;

use crate::{
    display::{
        context::{all_matched_lines_filled, opposite_positions},
        hunks::{matched_lines_indexes_for_hunk, matched_pos_to_hunks, merge_adjacent},
        side_by_side::lines_with_novel,
    },
    lines::MaxLine,
    parse::syntax::{self, MatchedPos},
    summary::DiffResultLsp,
};

// pub fn diffresult_to_ranges2(summary: &DiffResultLsp) -> Vec<Range> {
//     use syntax::MatchKind;
//     let mut ranges = Vec::new();

//     // If there's no RHS content or no positions, nothing to do.
//     if summary.rhs_src.is_empty() {
//         return ranges;
//     }

//     // 1) Group only the novel spans by their line number. We skip UnchangedToken and UnchangedPartOfNovelItem here.
//     let mut by_line: BTreeMap<u32, Vec<(u32, u32)>> = BTreeMap::new();
//     for m in &summary.rhs_positions {
//         match m.kind {
//             MatchKind::Novel { .. } | MatchKind::NovelWord { .. } => {
//                 let ln = m.pos.line.0;
//                 by_line.entry(ln).or_default().push((m.pos.start_col, m.pos.end_col));
//             }
//             _ => {}
//         }
//     }

//     // 2) For each line, sort its spans and merge any that overlap or touch. Emit a single Range per merged span.
//     for (ln, mut spans) in by_line {
//         spans.sort_unstable_by_key(|&(s, _)| s);
//         let mut iter = spans.into_iter();
//         let (mut cur_start, mut cur_end) = iter.next().unwrap();
//         for (s, e) in iter {
//             if s <= cur_end {
//                 // overlap or abut → extend
//                 cur_end = cur_end.max(e);
//             } else {
//                 // gap → flush previous
//                 ranges.push(Range {
//                     start: Position::new(ln, cur_start),
//                     end: Position::new(ln, cur_end),
//                 });
//                 cur_start = s;
//                 cur_end = e;
//             }
//         }
//         // flush last
//         ranges.push(Range {
//             start: Position::new(ln, cur_start),
//             end: Position::new(ln, cur_end),
//         });
//     }

//     ranges
// }

pub fn diffresult_to_ranges2(summary: &DiffResultLsp) -> Vec<Range> {
    use syntax::MatchKind;
    let mut ranges = Vec::new();

    // Bail early if there's no rhs text or no positions.
    if summary.rhs_src.is_empty() {
        return ranges;
    }

    // 1) For each novel/rhs‐only MatchedPos, bucket them by line, preserving the MatchKind so we know which are `Novel`
    //    vs `NovelWord`.
    let mut by_line: BTreeMap<u32, Vec<(MatchKind, u32, u32)>> = BTreeMap::new();
    for m in &summary.rhs_positions {
        match &m.kind {
            MatchKind::Novel { .. } | MatchKind::NovelWord { .. } => {
                let ln = m.pos.line.0;
                by_line
                    .entry(ln)
                    .or_default()
                    .push((m.kind.clone(), m.pos.start_col, m.pos.end_col));
            }
            _ => {}
        }
    }

    // 2) For each line in order, sort by start_col, then walk once and merge _only_ consecutive MatchKind::Novel spans.
    for (ln, mut spans) in by_line {
        // sort by the start column
        spans.sort_unstable_by_key(|&(_, s, _)| s);

        let mut i = 0;
        while i < spans.len() {
            let (kind, start, mut end) = spans[i].clone();

            if let MatchKind::Novel { .. } = kind {
                // merge all immediately following Novel spans
                i += 1;
                while i < spans.len() {
                    let (ref next_kind, ns, ne) = spans[i];
                    if let MatchKind::Novel { .. } = next_kind {
                        // extend our end
                        end = end.max(ne);
                        i += 1;
                    } else {
                        break;
                    }
                }
                // emit one big range covering all merged Novel
                ranges.push(Range {
                    start: Position::new(ln, start),
                    end: Position::new(ln, end),
                });
            } else {
                // NovelWord (or any other novel‐type): emit single
                ranges.push(Range {
                    start: Position::new(ln, start),
                    end: Position::new(ln, end),
                });
                i += 1;
            }
        }
    }

    ranges
}

pub fn diffresult_to_ranges<'f>(summary: &'f DiffResultLsp) -> Vec<Range> {
    let lhs_src = &summary.lhs_src;
    let rhs_src = &summary.rhs_src;
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

    if hunks.is_empty() || lhs_src.is_empty() || rhs_src.is_empty() {
        return vec![];
    }

    let lhs_lines = lhs_src.split('\n').collect::<Vec<&str>>();
    let rhs_lines = rhs_src.split('\n').collect::<Vec<&str>>();

    let (_, rhs_lines_with_novel) = lines_with_novel(&summary.lhs_positions, &summary.rhs_positions);

    tracing::debug!("summary.rhs_positions.len(): {}", summary.rhs_positions.len());
    tracing::debug!("rhs_lines_with_novel.len(): {}", rhs_lines_with_novel.len());

    let matched_lines =
        all_matched_lines_filled(&summary.lhs_positions, &summary.rhs_positions, &lhs_lines, &rhs_lines);
    let mut matched_lines = &matched_lines[..];

    // `lines_for_all_chunks` will be used for deduplication lookups. Keep using `HashMap` as it offers
    // average O(1) lookups/insertions compared to BTreeMap's O(log N).
    let mut lines_for_all_chunks: HashMap<u32, AllChunks> = HashMap::new();

    let mut ranges: Vec<Range> = Vec::with_capacity(rhs_lines_with_novel.len());
    tracing::debug!("hunks.len(): {}", hunks.len());
    for hunk in &hunks {
        // Sorted iteration is necessary for `lines`. Keep using `BTreeMap` here.
        let mut lines: BTreeMap<Option<u32>, Line<'f>> = BTreeMap::new();

        let (start_i, end_i) = matched_lines_indexes_for_hunk(matched_lines, hunk, 0);
        let aligned_lines = &matched_lines[start_i..end_i];
        matched_lines = &matched_lines[start_i..];

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
            let line_vec: Vec<Line<'_>> = lines.into_values().collect();
            for line in &line_vec {
                if let Some(side) = &line.rhs {
                    let ln = side.line_number;
                    ranges.extend(side.changes.iter().map(|ch| Range {
                        start: Position::new(ln, ch.start),
                        end: Position::new(ln, ch.end),
                    }));
                }
            }
        }
    }

    tracing::debug!("ranges.len(): {}", ranges.len());
    ranges
}

pub fn print(diff: &DiffResultLsp) {
    let file = diffresult_to_ranges(diff);
    tracing::debug!(
        "diffresult_to_ranges: {}",
        serde_json::to_string(&file).expect("failed to serialize file")
    );
}

pub fn print2(diff: &DiffResultLsp) {
    let file = diffresult_to_ranges2(diff);
    tracing::debug!(
        "diffresult_to_ranges2: {}",
        serde_json::to_string(&file).expect("failed to serialize file")
    );
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
        tracing::error!("Warning: Invalid line number {} encountered.", line_num.0);
        return;
    }
    let src_line = src_lines[line_idx];

    let mut relevant_matches = all_matches
        .iter()
        .filter(|m| m.pos.line == line_num && m.kind.is_novel()) // Combine filters
        .filter(|m| {
            !matches!(
                m.kind,
                syntax::MatchKind::UnchangedPartOfNovelItem { .. } | syntax::MatchKind::UnchangedToken { .. }
            )
        }) // Filter the deliberately ignored kinds early
        .peekable(); // Make the filtered iterator peekable for merging

    while let Some(m) = relevant_matches.next() {
        let change_to_add: Change2<'s>;
        let change_key: ChangeKey; // Use the lightweight key for lookups

        //  Merge Novel kinds before deduplication check (using the filtered iterator 'relevant_matches')
        if matches!(m.kind, MatchKind::Novel { .. }) {
            let current_start = m.pos.start_col;
            let mut current_end = m.pos.end_col;
            let highlight_type_ref = &m.kind; // Use kind from the *first* item in the merged sequence

            // Peek ahead and merge consecutive Novel items
            while let Some(next_m) = relevant_matches.peek() {
                // Peek ahead on the filtered iterator
                if matches!(next_m.kind, MatchKind::Novel { .. }) {
                    // Extend the range to the end of the next item
                    current_end = next_m.pos.end_col;
                    // Consume the peeked item as it's now part of the merged range
                    relevant_matches.next();
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
            // This match is not MatchKind::Novel (e.g., could be NovelWord).
            // Add it individually.
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
