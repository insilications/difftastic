use std::collections::HashMap;

use lsp_types::{Position, Range};
use serde::Serialize;

use crate::{parse::syntax::MatchKind, summary::DiffResultLsp};

pub fn diffresult_to_ranges(summary: &DiffResultLsp) -> Vec<Range> {
    // Early out if there's no RHS positions.
    if summary.rhs_positions.is_empty() {
        return Vec::new();
    }

    // 1) Bucket only the novel spans by line.  No `MatchKind` clones!
    let mut by_line: HashMap<u32, Vec<(bool, u32, u32)>> = HashMap::with_capacity(summary.rhs_positions.len());
    for m in &summary.rhs_positions {
        if m.kind.is_novel() {
            let ln = m.pos.line.0;
            let is_novel = matches!(m.kind, MatchKind::Novel { .. });
            by_line
                .entry(ln)
                .or_default()
                .push((is_novel, m.pos.start_col, m.pos.end_col));
        }
    }

    // 2) Sort line‐numbers (so output is in order).
    let mut lines: Vec<(u32, Vec<(bool, u32, u32)>)> = by_line.into_iter().collect();
    lines.sort_unstable_by_key(|&(ln, _)| ln);

    // 3) For each line, sort its spans by start_col, then walk once merging *only* consecutive `MatchKind::Novel`
    //    spans.
    let mut ranges = Vec::with_capacity(summary.rhs_positions.len()); // upper bound

    for (ln, mut spans) in lines {
        spans.sort_unstable_by_key(|&(_, start, _)| start);

        let mut i = 0;
        while i < spans.len() {
            let (is_novel, start, mut end) = spans[i];
            i += 1;

            if is_novel {
                // Merge all *consecutive* MatchKind::Novel spans
                while i < spans.len() && spans[i].0 {
                    end = end.max(spans[i].2);
                    i += 1;
                }
            }

            ranges.push(Range {
                start: Position::new(ln, start),
                end: Position::new(ln, end),
            });
        }
    }

    ranges
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
    const fn new(line_number: u32) -> Side<'s> {
        Side {
            line_number,
            changes: Vec::new(),
        }
    }
}

#[derive(Debug, Serialize)]
struct Change2<'c> {
    start: u32,
    end: u32,
    content: &'c str,
    highlight_type: &'c MatchKind,
}
