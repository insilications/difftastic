use lsp_types::{
    InitializeParams, PositionEncodingKind, SaveOptions, ServerCapabilities, TextDocumentSyncCapability,
    TextDocumentSyncKind, TextDocumentSyncOptions, TextDocumentSyncSaveOptions, WorkspaceFoldersServerCapabilities,
    WorkspaceServerCapabilities,
};

macro_rules! test {
    ($lhs:ident $(.$field:ident)*) => {
        Some($lhs)
            $(.and_then(|opt| opt.$field.as_ref()))*
        == Some(&true)
    };
}

pub fn negotiate_capabilities(init_params: &InitializeParams) -> (ServerCapabilities, NegotiatedCapabilities) {
    let client_caps = &init_params.capabilities;

    let final_caps = NegotiatedCapabilities {
        client_show_message_request: test!(
            client_caps
                .window
                .show_message
                .message_action_item
                // This is required for knowing which action is performed.
                .additional_properties_support
        ),
        server_initiated_progress: test!(client_caps.window.work_done_progress),
        watch_files: test!(client_caps.workspace.did_change_watched_files.dynamic_registration),
        // Workaround: https://github.com/neovim/neovim/issues/23380
        watch_files_relative_pattern: test!(client_caps.workspace.did_change_watched_files.relative_pattern_support),
        workspace_configuration: test!(client_caps.workspace.configuration),
    };

    let server_caps = ServerCapabilities {
        workspace: Some(WorkspaceServerCapabilities {
            workspace_folders: Some(WorkspaceFoldersServerCapabilities {
                supported: Some(true),
                change_notifications: None,
            }),
            file_operations: None,
        }),
        position_encoding: Some(PositionEncodingKind::UTF16),
        text_document_sync: Some(TextDocumentSyncCapability::Options(TextDocumentSyncOptions {
            open_close: Some(true),
            change: Some(TextDocumentSyncKind::INCREMENTAL),
            will_save: None,
            will_save_wait_until: None,
            // save: Some(TextDocumentSyncSaveOptions::Supported(true)),
            save: Some(TextDocumentSyncSaveOptions::SaveOptions(SaveOptions {
                include_text: Some(false),
            })),
        })),
        experimental: Some(serde_json::json!({
            "diff": true,
        })),
        ..ServerCapabilities::default()
    };

    (server_caps, final_caps)
}

#[derive(Clone, Debug, Default)]
#[allow(clippy::struct_excessive_bools)]
pub struct NegotiatedCapabilities {
    pub client_show_message_request: bool,
    pub server_initiated_progress: bool,
    pub watch_files: bool,
    pub watch_files_relative_pattern: bool,
    pub workspace_configuration: bool,
}
