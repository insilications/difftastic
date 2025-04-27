use lsp_types::Range;
// use lsp_types::notification::Notification;
use lsp_types::request::Request;
use serde::{Deserialize, Serialize};

// /// <https://github.com/microsoft/language-server-protocol/issues/1002>
// pub enum ParentModule {}

// impl Request for ParentModule {
// type Params = lsp_types::TextDocumentPositionParams;
// type Result = Option<lsp_types::GotoDefinitionResponse>;
//     const METHOD: &'static str = "experimental/parentModule";
// }

// pub enum ReloadFlake {}

// impl Notification for ReloadFlake {
//     type Params = ();
//     const METHOD: &'static str = "nil/reloadFlake";
// }

pub enum DidOpenTextDocumentCustom {}

impl Request for DidOpenTextDocumentCustom {
    type Params = DidOpenTextDocumentCustomParams;
    type Result = Option<DiffRangesResponse>;

    const METHOD: &'static str = "textDocument/didOpenCustom";
}

#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DidOpenTextDocumentCustomParams {
    pub rev: String,
    pub text_document: DidOpenTextDocumentCustomItem,
}

#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DidOpenTextDocumentCustomItem {
    pub uri: String,
    pub language_id: String,
}

#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DiffRangesResponse {
    pub ranges: Vec<Range>,
}

// pub enum DidOpenTextDocumentCustom {}

// impl Notification for DidOpenTextDocumentCustom {
//     type Params = DidOpenTextDocumentCustomParams;
//     const METHOD: &'static str = "textDocument/didOpenCustom";
// }

// #[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
// #[serde(rename_all = "camelCase")]
// pub struct DidOpenTextDocumentCustomParams {
//     pub text_document: DidOpenTextDocumentCustomItem,
// }

// #[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
// #[serde(rename_all = "camelCase")]
// pub struct DidOpenTextDocumentCustomItem {
//     pub uri: String,
//     pub language_id: String,
// }
