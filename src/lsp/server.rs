use std::{
    future::{Future, ready},
    ops::ControlFlow,
    path::{Path, PathBuf},
    sync::Arc,
};

use async_lsp::{ClientSocket, ErrorCode, LanguageClient, ResponseError, Result, router::Router};
use lsp_types::{
    ConfigurationItem, ConfigurationParams, DidChangeConfigurationParams, DidChangeTextDocumentParams,
    DidCloseTextDocumentParams, DidOpenTextDocumentParams, DidSaveTextDocumentParams, InitializeParams,
    InitializeResult, InitializedParams, Registration, RegistrationParams, ServerInfo, notification as notif,
    notification::Notification, request, request::Request,
};

use crate::{
    diff_for_lsp,
    display::json3::diffresult_to_ranges,
    lsp::{
        cache,
        capabilities::{NegotiatedCapabilities, negotiate_capabilities},
        config::{Config, WORKSPACE_CONFIG_KEY},
        lsp_ext,
        uri_ext::UriExt,
    },
    tracing_to_json, tracing_to_json_pretty,
};

type NotifyResult = ControlFlow<async_lsp::Result<()>>;
struct UpdateConfigEvent(serde_json::Value);

const LSP_SERVER_NAME: &str = "difftastic-lsp";
const LSP_SERVER_VERSION: &str = "0.1.0";

pub struct Server {
    // Immutable (mostly).
    client: ClientSocket,
    // States.
    // config: Arc<Config>,
    // cache_state: Option<cache::AppStateShared>,
    root_path: PathBuf,
    capabilities: NegotiatedCapabilities,
}

impl Server {
    pub fn new_router(client: ClientSocket) -> Router<Self> {
        let this = Self::new(client);
        let mut router = Router::new(this);
        router
            //// Lifecycle ////
            // .request::<request::Initialize, _>(|server, params| std::future::ready(server.on_initialize(params)))
            // .request::<request::Initialize, _>(|server, params| server.on_initialize(params))
            .request::<request::Initialize, _>(Self::on_initialize)
            // .request::<request::Initialize, _>(|st, params| {
            //     tracing::debug!("1 req::Initialize");
            //     // tracing_to_json!(&params, "Initialize");
            //     let (server_caps, final_caps) = negotiate_capabilities(&params);
            //     st.capabilities = final_caps;
            //     st.root_path = params
            //         .workspace_folders
            //         .as_ref()
            //         .into_iter()
            //         .flatten()
            //         .next()
            //         .and_then(|ws| ws.uri.to_file_path())
            //         .map_or_else(|| PathBuf::from("."), PathBuf::from);
            //     tracing::info!("root_path: {}", st.root_path.display());
            //     // tracing_to_json_pretty!(&server_caps, "Server Capabilities");
            //     // tracing_to_json_pretty!(&self.capabilities, "Client Capabilities");
            //     // *Arc::get_mut(&mut self.config).expect("No concurrent access yet") = Config::new(root_path);
            //     // *Arc::get_mut(&mut st.config).expect("No concurrent access yet") =
            // Config::new(st.root_path.clone());     // if let Some(options) =
            // params.initialization_options {     //     if options.as_object().filter(|o|
            // !o.is_empty()).is_some() {     //         tracing::debug!("initialization_options:
            // {options}");     //         #[allow(unused_must_use)]
            //     //         st.on_update_config(UpdateConfigEvent(options));
            //     //     }
            //     // }
            //     // st.cache_state = Some(cache::AppStateShared::new(&st.root_path).expect("Failed to create cache
            //     // state"));
            //     async move {
            //         tracing::debug!("2 req::Initialize");
            //         tracing_to_json!(&params, "Initialize");
            //         Ok(InitializeResult {
            //             capabilities: server_caps,
            //             server_info: Some(ServerInfo {
            //                 name: LSP_SERVER_NAME.into(),
            //                 version: Some(LSP_SERVER_VERSION.into()),
            //             }),
            //             offset_encoding: None,
            //         })
            //     }
            // })
            .notification::<notif::Initialized>(Self::on_initialized)
            .request::<request::Shutdown, _>(|_, ()| {
                tracing::info!("req::Shutdown");
                ready(Ok(()))
            })
            .notification::<notif::Exit>(|_, ()| {
                tracing::info!("notif::Exit");
                ControlFlow::Break(Ok(()))
            })
            //// Requests ////
            // .request::<lsp_ext::DidOpenTextDocumentCustom, _>(Self::on_did_open_custom)
            //// Notifications ////
            .notification::<notif::DidOpenTextDocument>(Self::on_did_open)
            .notification::<notif::DidCloseTextDocument>(Self::on_did_close)
            .notification::<notif::DidChangeTextDocument>(Self::on_did_change)
            .notification::<notif::DidChangeConfiguration>(Self::on_did_change_configuration)
            .notification::<notif::DidSaveTextDocument>(Self::on_did_save);
        //// Events ////
        // .event(Self::on_update_config);
        router
    }

    pub fn new(client: ClientSocket) -> Self {
        Self {
            // vfs: Arc::new(RwLock::new(Vfs::new())),
            // config: Arc::new(Config::new("/non-existing-path".into())),
            client,
            // cache_state: None,
            root_path: PathBuf::new(),
            capabilities: NegotiatedCapabilities::default(),
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn on_initialize(
        &mut self,
        params: InitializeParams,
    ) -> impl Future<Output = Result<InitializeResult, ResponseError>> {
        // ) -> Result<InitializeResult, Box<dyn std::error::Error + Send + Sync>> {
        tracing::debug!("1 req::Initialize");
        // tracing_to_json!(&params, "Initialize");

        let (server_caps, final_caps) = negotiate_capabilities(&params);
        // self.capabilities = final_caps;

        // self.root_path = params
        //     .workspace_folders
        //     .as_ref()
        //     .into_iter()
        //     .flatten()
        //     .next()
        //     .and_then(|ws| ws.uri.to_file_path())
        //     .map_or_else(|| PathBuf::from("."), PathBuf::from);

        // tracing::info!("root_path: {}", self.root_path.display());

        // tracing_to_json_pretty!(&server_caps, "Server Capabilities");
        // tracing_to_json_pretty!(&self.capabilities, "Client Capabilities");

        // *Arc::get_mut(&mut self.config).expect("No concurrent access yet") = Config::new(root_path);
        // *Arc::get_mut(&mut self.config).expect("No concurrent access yet") = Config::new(self.root_path.clone());

        // if let Some(options) = params.initialization_options {
        //     if options.as_object().filter(|o| !o.is_empty()).is_some() {
        //         tracing::debug!("initialization_options: {options}");
        //         #[allow(unused_must_use)]
        //         self.on_update_config(UpdateConfigEvent(options));
        //     }
        // }

        // self.cache_state = Some(cache::AppStateShared::new(&self.root_path).expect("Failed to create cache state"));

        async move {
            tracing::debug!("2 req::Initialize");
            tracing_to_json!(&params, "Initialize");
            self.capabilities = final_caps;
            Ok(InitializeResult {
                capabilities: server_caps,
                server_info: Some(ServerInfo {
                    name: LSP_SERVER_NAME.into(),
                    version: Some(LSP_SERVER_VERSION.into()),
                }),
                // offset_encoding: Some("utf-8".to_string()),
                offset_encoding: None,
            })
        }
        // ready(Ok(InitializeResult {
        //     capabilities: server_caps,
        //     server_info: Some(ServerInfo {
        //         name: LSP_SERVER_NAME.into(),
        //         version: Some(LSP_SERVER_VERSION.into()),
        //     }),
        //     // offset_encoding: Some("utf-8".to_string()),
        //     offset_encoding: None,
        // }))
    }

    #[allow(clippy::unused_self)]
    fn on_initialized(&mut self, _params: InitializedParams) -> NotifyResult {
        tracing::debug!("notif::Initialized");

        // if self.capabilities.workspace_configuration {
        //     tokio::spawn({
        //         let mut client = self.client.clone();
        //         async move {
        //             Self::register_did_change_configuration(&mut client).await;
        //         }
        //     });
        // }

        ControlFlow::Continue(())
    }

    // #[allow(clippy::needless_pass_by_value)]
    // fn on_did_open_custom(
    //     &mut self,
    //     params: lsp_ext::DidOpenTextDocumentCustomParams,
    // ) -> impl Future<Output = Result<Option<lsp_ext::DiffRangesResponse>, ResponseError>> {
    //     tracing_to_json_pretty!(&params, "lsp_ext::DidOpenTextDocumentCustom");
    //     tracing::debug!("TESTE");

    //     let relative_stripped_path = Path::new(&params.text_document.uri)
    //         .strip_prefix(&self.root_path)
    //         .map_err(|err| {
    //             tracing::error!("Failed to strip prefix: {err}");
    //             ready(Err::<Option<lsp_ext::DiffRangesResponse>, ResponseError>(
    //                 ResponseError::new(ErrorCode::REQUEST_FAILED, format!("Failed to strip prefix: {err}")),
    //             ))
    //         })
    //         .unwrap();

    //     // let _ = self
    //     //     .cache_state
    //     //     .as_ref()
    //     //     .unwrap()
    //     //     .populate_history(&params.rev, &relative_stripped_path);

    //     // Handle the Result returned by populate_history
    //     if let Err(err) = self
    //         .cache_state
    //         .as_ref()
    //         .unwrap()
    //         .populate_history(&params.rev, relative_stripped_path)
    //     {
    //         tracing::error!("Failed to populate history: {err}");
    //         return ready(Err(ResponseError::new(
    //             ErrorCode::REQUEST_FAILED,
    //             format!("Failed to populate history: {err}"),
    //         )));
    //     }
    //     // let response = lsp_ext::DiffRangesResponse { ranges: vec![] };

    //     // self.cache_state
    //     //     .as_ref()
    //     //     .unwrap()
    //     //     .iterate_path_versions(&relative_stripped_path);

    //     let rhs_path_buf = PathBuf::from(&relative_stripped_path);

    //     // Note: lookup_version now returns an owned FileVersion due to cloning
    //     if let Some((commit_id, version)) = self
    //         .cache_state
    //         .as_ref()
    //         .unwrap()
    //         .lookup_version(relative_stripped_path, &params.rev)
    //     {
    //         // Arc counts inside the cloned FileVersion will reflect sharing
    //         tracing::debug!(
    //             "Arc Counts : content: {} - summary: {}",
    //             Arc::strong_count(&version.content), // Count on the cloned Arc
    //             Arc::strong_count(&version.summary)  // Count on the cloned Arc
    //         );
    //         tracing::debug!("Path           : {}", relative_stripped_path.display());
    //         tracing::debug!("Revspec        : {}", params.rev);
    //         tracing::debug!("Commit         : {}", commit_id.short());
    //         tracing::debug!("Summary        : {}", version.summary);
    //         tracing::debug!("Content Length : {}", version.content.len());

    //         match diff_for_lsp(&rhs_path_buf, &version.content, &params.text_document.language_id) {
    //             Ok(diff_result) => {
    //                 if diff_result.has_reportable_change() {
    //                     ready(Ok(Some(lsp_ext::DiffRangesResponse {
    //                         ranges: diffresult_to_ranges(&diff_result),
    //                     })))
    //                     // match json3::print(&diff_result) {
    //                     //     Ok(json) => ready(Ok(Some(lsp_ext::DiffRangesResponse { ranges: json }))),
    //                     //     Err(err) => {
    //                     //         tracing::error!("Failed to serialize lsp_ext::DiffRangesResponse: {err}");
    //                     //         ready(Err(ResponseError::new(
    //                     //             ErrorCode::INTERNAL_ERROR,
    //                     //             format!("Failed to serialize lsp_ext::DiffRangesResponse: {err}"),
    //                     //         )))
    //                     //     }
    //                     // }
    //                 } else {
    //                     tracing::debug!("No changes detected for path {}", rhs_path_buf.display());
    //                     ready(Ok(Some(lsp_ext::DiffRangesResponse { ranges: vec![] })))
    //                 }
    //             }
    //             Err(err) => ready(Err(err)),
    //         }
    //     } else {
    //         tracing::debug!("Version {} not found for path {}", &params.rev, rhs_path_buf.display());
    //         ready(Err(ResponseError::new(
    //             ErrorCode::REQUEST_FAILED,
    //             format!("Version {} not found for path {}", &params.rev, rhs_path_buf.display()),
    //         )))
    //     }
    // }

    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::unused_self)]
    fn on_did_open(&mut self, params: DidOpenTextDocumentParams) -> NotifyResult {
        tracing::debug!(
            "notif::DidOpenTextDocument - params.text_document.uri: {} - params.text_document.version: {}",
            params.text_document.uri.to_file_path().unwrap_or_default().display(),
            params.text_document.version
        );

        ControlFlow::Continue(())
    }

    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::unused_self)]
    fn on_did_close(&mut self, params: DidCloseTextDocumentParams) -> NotifyResult {
        tracing::debug!(
            "notif::DidCloseTextDocument - params.text_document.uri: {}",
            params.text_document.uri.to_file_path().unwrap_or_default().display()
        );

        ControlFlow::Continue(())
    }

    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::unused_self)]
    fn on_did_change(&mut self, params: DidChangeTextDocumentParams) -> NotifyResult {
        let file_path = params.text_document.uri.to_file_path().unwrap_or_default();
        let file_path_display = file_path.display();
        tracing::debug!(
            "notif::DidChangeTextDocument - params.text_document.uri: {} - params.text_document.version: {} -
        params.content_changes.len(): {}",
            file_path_display,
            params.text_document.version,
            params.content_changes.len()
        );
        for c in &params.content_changes {
            tracing::debug!("{} - content_change.range: {:?}", file_path_display, c.range);
            tracing::debug!("{} - content_change.text: {}", file_path_display, c.text);
        }

        ControlFlow::Continue(())
    }

    #[allow(clippy::unused_self)]
    fn on_did_save(&mut self, _params: DidSaveTextDocumentParams) -> NotifyResult {
        tracing::debug!("notif::DidSaveTextDocument");

        ControlFlow::Continue(())
    }

    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::unused_self)]
    fn on_did_change_configuration(&mut self, params: DidChangeConfigurationParams) -> NotifyResult {
        tracing_to_json_pretty!(&params, "notif::DidChangeConfiguration");
        self.spawn_reload_config();

        ControlFlow::Continue(())
    }

    fn spawn_reload_config(&self) {
        if !self.capabilities.workspace_configuration {
            return;
        }
        let mut client = self.client.clone();
        tokio::spawn(async move {
            let ret = client
                .configuration(ConfigurationParams {
                    items: vec![ConfigurationItem {
                        scope_uri: None,
                        section: Some(WORKSPACE_CONFIG_KEY.into()),
                    }],
                })
                .await;
            let mut v = match ret {
                Ok(v) => v,
                Err(err) => {
                    tracing::error!("Failed to update config: {err}");
                    // client.show_message_ext(MessageType::ERROR, format_args!("Failed to update config: {err}"));
                    return;
                }
            };

            let v = v.pop().unwrap_or_default();
            tracing::debug!("Updating config: {v}");
            let _: Result<_, _> = client.emit(UpdateConfigEvent(v));
        });
    }

    // #[allow(clippy::needless_pass_by_value)]
    // fn on_update_config(&mut self, value: UpdateConfigEvent) -> NotifyResult {
    //     let mut config = Config::clone(&self.config);
    //     let mut errors = Vec::new();
    //     config.update(&value.0, &mut errors);

    //     tracing::info!("Updated config, errors: {errors:?}, config: {config:?}");

    //     if errors.is_empty() {
    //         self.config = Arc::new(config);
    //     } else {
    //         let msg = std::iter::once("Failed to apply some settings:")
    //             .chain(errors.iter().flat_map(|s| ["\n- ", s]))
    //             .collect::<String>();
    //         tracing::error!("{msg}");
    //         // self.client.show_message_ext(MessageType::ERROR, msg);
    //     }

    //     ControlFlow::Continue(())
    // }

    async fn register_did_change_configuration(client: &mut ClientSocket) {
        let register_options = DidChangeConfigurationParams {
            settings: serde_json::to_value(WORKSPACE_CONFIG_KEY).unwrap(),
        };
        let params = RegistrationParams {
            registrations: vec![Registration {
                id: notif::DidChangeConfiguration::METHOD.into(),
                method: notif::DidChangeConfiguration::METHOD.into(),
                register_options: Some(serde_json::to_value(register_options).unwrap()),
            }],
        };
        if let Err(err) = client.register_capability(params).await {
            tracing::error!("Failed to register DidChangeConfiguration: {err:#}");
            // client.show_message_ext(MessageType::ERROR, format!("Failed to watch flake files: {err:#}"));
        }
        tracing::info!("Registered DidChangeConfiguration");
    }
}
