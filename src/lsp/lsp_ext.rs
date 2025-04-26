use lsp_types::notification::Notification;
use serde::{Deserialize, Serialize};
// use lsp_types::request::Request;

// /// <https://github.com/microsoft/language-server-protocol/issues/1002>
// pub enum ParentModule {}

// impl Request for ParentModule {
//     type Params = lsp_types::TextDocumentPositionParams;
//     type Result = Option<lsp_types::GotoDefinitionResponse>;
//     const METHOD: &'static str = "experimental/parentModule";
// }

// pub enum ReloadFlake {}

// impl Notification for ReloadFlake {
//     type Params = ();
//     const METHOD: &'static str = "nil/reloadFlake";
// }

#[derive(Debug)]
pub enum DidOpenTextDocumentCustom {}

impl Notification for DidOpenTextDocumentCustom {
    type Params = DidOpenTextDocumentCustomParams;
    const METHOD: &'static str = "textDocument/didOpenCustom";
}

#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DidOpenTextDocumentCustomParams {
    pub text_document: DidOpenTextDocumentCustomItem,
}

#[derive(Debug, Eq, PartialEq, Clone, Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DidOpenTextDocumentCustomItem {
    pub uri: String,
    pub language_id: String,
}
