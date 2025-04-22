#![allow(non_upper_case_globals)]
#![forbid(unsafe_code)]
// #[macro_use]
// extern crate bitflags;

use lsp_types::{ServerCapabilities, ServerInfo};

use std::{collections::HashMap, fmt::Debug};

use serde::{Deserialize, Serialize, de, de::Error};
use serde_json::Value;

// pub use uri::Uri;
// mod uri;

#[derive(Debug, PartialEq, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MyServerCapabilities {
    // Embed the standard capabilities using composition
    // `#[serde(flatten)]` makes the JSON look like extension/inheritance
    #[serde(flatten)]
    pub standard: ServerCapabilities, // Field name can be anything, 'standard' is descriptive

    // Add your new capability field(s)
    #[serde(skip_serializing_if = "Option::is_none")] // Good practice for capabilities
    pub diff: Option<bool>,
}

#[derive(Debug, PartialEq, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct MyInitializeResult {
    /// The capabilities the language server provides.
    pub capabilities: MyServerCapabilities,

    /// Information about the server.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub server_info: Option<ServerInfo>,

    /// Unofficial UT8-offsets extension.
    ///
    /// See https://clangd.llvm.org/extensions.html#utf-8-offsets.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub offset_encoding: Option<String>,
}
