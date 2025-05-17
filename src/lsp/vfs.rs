//! Simple VFS for an LSP server.
//!
//! Public surface:
//!   `Vfs::open`
//!   `Vfs::apply_changes`
//!   `Vfs::get_text` / `Vfs::get_document`
//!
//! Intended usage inside `LanguageServer` handler methods.

use std::{
    collections::HashMap,
    sync::{Arc, RwLock},
};

use anyhow::{Context, Result, bail};
use lsp_types::{Position, TextDocumentContentChangeEvent, Uri};
use ropey::Rope;

// use unicode_segmentation::UnicodeSegmentation;
use crate::lsp::uri_ext::UriExt;

/// A single text document tracked by the server.
#[derive(Debug, Clone)]
pub struct Document {
    pub version: i32,
    pub text: Rope,
}

impl Document {
    // pub fn new(version: i32, initial_text: String) -> Self {
    pub fn new(version: i32, initial_text: &str) -> Self {
        Self {
            version,
            text: Rope::from_str(initial_text),
        }
    }
}

/// Inner map (not public). Wrapped in `Arc<RwLock<_>>` for sharing.
#[derive(Default, Debug)]
pub struct VfsInner {
    docs: HashMap<Uri, Document>,
}

/// Cheap clone = another Arc handle.
#[derive(Default, Debug, Clone)]
pub struct Vfs(pub Arc<RwLock<VfsInner>>);

impl Vfs {
    // ───────────────────────────────────────────────────────────────
    // `textDocument/didOpen` notification
    // ───────────────────────────────────────────────────────────────
    /// Insert or replace a document that was just opened.
    pub fn open(&self, uri: Uri, version: i32, text: &str) {
        let mut inner = self.0.write().unwrap();
        inner.docs.insert(uri, Document::new(version, text));
    }

    // ───────────────────────────────────────────────────────────────
    // `textDocument/didChange` notification
    // ───────────────────────────────────────────────────────────────
    /// Apply one `didChange` batch to an already-tracked document.
    ///
    /// * `changes` come exactly in the order the client sent them (spec §3.17.2).
    /// * `version` must be strictly newer than the stored version, otherwise the call is rejected with an error
    ///   (`Err`).
    pub fn apply_changes(&self, uri: &Uri, new_version: i32, changes: &[TextDocumentContentChangeEvent]) -> Result<()> {
        let mut inner = self.0.write().unwrap();
        let doc = inner.docs.get_mut(uri).with_context(|| {
            format!("Document {} not found (did you forget didOpen?)", uri.to_file_path().unwrap_or_default().display())
        })?;

        // Prevent out-of-order edits.
        if new_version <= doc.version {
            bail!("Stale change: incoming version {new_version} <= stored version {}", doc.version);
        }

        for change in changes {
            apply_change(doc, change)?;
        }

        doc.version = new_version;
        Ok(())
    }

    // ───────────────────────────────────────────────────────────────
    // READERS
    // ───────────────────────────────────────────────────────────────
    /// Obtain *owned* text (expensive – clones the rope).
    pub fn get_text(&self, uri: &Uri) -> Option<String> {
        let inner = self.0.read().unwrap();
        inner.docs.get(uri).map(|d| d.text.to_string())
    }

    /// Borrow a document immutably (cheap, requires the caller to hold a lock
    /// only for as long as the reference lives).
    pub fn get_document(&self, uri: &Uri) -> Option<Document> {
        let inner = self.0.read().unwrap();
        inner.docs.get(uri).cloned()
    }
}

// ───────────────────────────────────────────────────────────────
// Internal helpers
// ───────────────────────────────────────────────────────────────
fn apply_change(doc: &mut Document, change: &TextDocumentContentChangeEvent) -> Result<()> {
    match change.range {
        // Full-document replace.
        None => {
            doc.text = Rope::from_str(&change.text);
        }

        // Incremental edits.
        Some(range) => {
            let start_idx = lsp_position_to_char_index(&doc.text, range.start)?;
            let end_idx = lsp_position_to_char_index(&doc.text, range.end)?;
            doc.text.remove(start_idx..end_idx);
            doc.text.insert(start_idx, &change.text);
        }
    }
    Ok(())
}

/// Convert LSP's UTF-16‐based Position to a char index inside the Rope.
///
/// ropey is char-oriented, so we need to translate UTF-16 offsets.
/// (Most files are ASCII/UTF-8 only => `O(number_of_code_points_in_line)` walk.)
fn lsp_position_to_char_index(text: &Rope, pos: Position) -> Result<usize> {
    let line = pos.line as usize;

    // 1. Guard: line must exist.
    if line >= text.len_lines() {
        bail!("Position line {} out of bounds – document has {} lines", line, text.len_lines());
    }

    // 2. Find char offset of line start.
    let line_start_char = text.line_to_char(line);

    // 3. Convert the UTF-16 column to a char offset by scanning this line.
    let wanted_utf16_units = pos.character as usize;
    let mut seen_utf16_units = 0;
    for (idx, ch) in text.line(line).chars().enumerate() {
        if seen_utf16_units == wanted_utf16_units {
            return Ok(line_start_char + idx);
        }
        seen_utf16_units += ch.len_utf16();
        if seen_utf16_units > wanted_utf16_units {
            // Cursor is inside a surrogate pair – snap to next char boundary.
            return Ok(line_start_char + idx + 1);
        }
    }

    // Cursor can legally be “one past the end”.
    if seen_utf16_units == wanted_utf16_units {
        Ok(line_start_char + text.line(line).chars().count())
    } else {
        bail!("character offset out of bounds for that line");
    }
}
