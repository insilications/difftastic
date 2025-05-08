use std::{
    backtrace::Backtrace,
    borrow::BorrowMut,
    cell::Cell,
    fmt,
    future::{Future, ready},
    ops::ControlFlow,
    panic,
    panic::UnwindSafe,
    path::{Path, PathBuf},
    sync::{Arc, Once},
};

use anyhow::{Result, bail};
use async_lsp::{ClientSocket, ErrorCode, LanguageClient, ResponseError, router::Router};
use lsp_types::{
    ConfigurationItem, ConfigurationParams, DidChangeConfigurationParams, DidChangeTextDocumentParams,
    DidCloseTextDocumentParams, DidOpenTextDocumentParams, DidSaveTextDocumentParams, InitializeParams,
    InitializeResult, InitializedParams, Registration, RegistrationParams, ServerInfo, notification as notif,
    notification::Notification,
    request::{
        Request, {self as req},
    },
};
use tokio::{task, task::JoinHandle};

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

#[derive(Debug)]
pub struct StateSnapshot {
    pub config: Arc<Config>,
    pub root_path: PathBuf,
}

pub struct Server {
    // Immutable (mostly).
    client: ClientSocket,
    // States.
    config: Arc<Config>,
    cache_state: cache::CacheStateShared,
    root_path: PathBuf,
    capabilities: NegotiatedCapabilities,
}

impl Server {
    pub fn new_router(client: ClientSocket) -> Router<Self> {
        let this = Self::new(client);
        let mut router = Router::new(this);
        router
            //// Lifecycle ////
            .request::<req::Initialize, _>(Self::on_initialize)
            .notification::<notif::Initialized>(Self::on_initialized)
            .request::<req::Shutdown, _>(|_, ()| {
                tracing::info!("req::Shutdown");
                ready(Ok(()))
            })
            .notification::<notif::Exit>(|_, ()| {
                tracing::info!("notif::Exit");
                ControlFlow::Break(Ok(()))
            })
            //// Requests ////
            // .request_snap::<lsp_ext::DidOpenTextDocumentCustom>(on_did_open_custom)
            .request::<lsp_ext::DidOpenTextDocumentCustom, _>(Self::on_did_open_custom)
            //// Notifications ////
            .notification::<notif::DidOpenTextDocument>(Self::on_did_open)
            .notification::<notif::DidCloseTextDocument>(Self::on_did_close)
            .notification::<notif::DidChangeTextDocument>(Self::on_did_change)
            .notification::<notif::DidChangeConfiguration>(Self::on_did_change_configuration)
            .notification::<notif::DidSaveTextDocument>(Self::on_did_save)
            //// Events ////
            .event(Self::on_update_config);
        router
    }

    pub fn new(client: ClientSocket) -> Self {
        Self {
            config: Arc::new(Config::new("/non-existing-path".into())),
            client,
            cache_state: cache::CacheStateShared::new(),
            root_path: PathBuf::new(),
            capabilities: NegotiatedCapabilities::default(),
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    pub fn on_initialize(
        &mut self,
        params: InitializeParams,
    ) -> impl Future<Output = Result<InitializeResult, ResponseError>> {
        tracing_to_json!(&params, "Initialize");

        let (server_caps, final_caps) = negotiate_capabilities(&params);
        self.capabilities = final_caps;

        self.root_path = params
            .workspace_folders
            .as_ref()
            .into_iter()
            .flatten()
            .next()
            .and_then(|ws| ws.uri.to_file_path())
            .map_or_else(|| PathBuf::from("."), PathBuf::from);

        tracing::info!("root_path: {}", self.root_path.display());

        tracing_to_json_pretty!(&server_caps, "Server Capabilities");
        tracing_to_json_pretty!(&self.capabilities, "Client Capabilities");

        if let Err(err) = self.cache_state.set_repo(&self.root_path) {
            tracing::error!("Failed to set cache_state repo for {}: {err}", self.root_path.display());
            return ready(Err(ResponseError::new(
                ErrorCode::REQUEST_FAILED,
                format!("Failed to populate history: {err}"),
            )));
        }

        // *Arc::get_mut(&mut self.config).expect("No concurrent access yet") = Config::new(root_path);
        *Arc::get_mut(&mut self.config).expect("No concurrent access yet") = Config::new(self.root_path.clone());

        if let Some(options) = params.initialization_options {
            if options.as_object().filter(|o| !o.is_empty()).is_some() {
                tracing::debug!("initialization_options: {options}");
                #[allow(unused_must_use)]
                self.on_update_config(UpdateConfigEvent(options));
            }
        }

        // self.cache_state = Some(cache::CacheStateShared::new(&self.root_path).expect(
        //     "Failed to create cache
        // state",
        // ));

        // async move {
        //     tracing::debug!("req::Initialize");
        //     Ok(InitializeResult {
        //         capabilities: server_caps,
        //         server_info: Some(ServerInfo {
        //             name: LSP_SERVER_NAME.into(),
        //             version: Some(LSP_SERVER_VERSION.into()),
        //         }),
        //         // offset_encoding: Some("utf-8".to_string()),
        //         offset_encoding: None,
        //     })
        // }
        ready(Ok(InitializeResult {
            capabilities: server_caps,
            server_info: Some(ServerInfo {
                name: LSP_SERVER_NAME.into(),
                version: Some(LSP_SERVER_VERSION.into()),
            }),
            // offset_encoding: Some("utf-8".to_string()),
            offset_encoding: None,
        }))
    }

    #[allow(clippy::unused_self)]
    fn on_initialized(&mut self, _params: InitializedParams) -> NotifyResult {
        tracing::debug!("notif::Initialized");

        if self.capabilities.workspace_configuration {
            tokio::spawn({
                let mut client = self.client.clone();
                async move {
                    Self::register_did_change_configuration(&mut client).await;
                }
            });
        }

        ControlFlow::Continue(())
    }

    #[allow(clippy::needless_pass_by_value)]
    fn on_did_open_custom(
        &mut self,
        params: lsp_ext::DidOpenTextDocumentCustomParams,
    ) -> impl Future<Output = Result<Option<lsp_ext::DiffRangesResponse>, ResponseError>> {
        tracing_to_json_pretty!(&params, "lsp_ext::DidOpenTextDocumentCustom");

        let relative_stripped_path = Path::new(&params.text_document.uri)
            .strip_prefix(&self.root_path)
            .map_err(|err| {
                tracing::error!("Failed to strip prefix: {err}");
                ready(Err::<Option<lsp_ext::DiffRangesResponse>, ResponseError>(
                    ResponseError::new(ErrorCode::REQUEST_FAILED, format!("Failed to strip prefix: {err}")),
                ))
            })
            .unwrap();

        tracing::debug!(
            "params.rev: {} - relative_stripped_path: {}",
            &params.rev,
            relative_stripped_path.display()
        );

        // Handle the Result returned by populate_history
        if let Err(err) = self.cache_state.populate_history(&params.rev, relative_stripped_path) {
            tracing::error!("Failed to populate history: {err}");
            return ready(Err(ResponseError::new(
                ErrorCode::REQUEST_FAILED,
                format!("Failed to populate history: {err}"),
            )));
        }
        // let response = lsp_ext::DiffRangesResponse { ranges: vec![] };

        // self.cache_state
        //     .as_ref()
        //     .unwrap()
        //     .iterate_path_versions(&relative_stripped_path);

        let rhs_path_buf = PathBuf::from(&relative_stripped_path);

        // Note: lookup_version now returns an owned FileVersion due to cloning
        if let Some((commit_id, version)) = self.cache_state.lookup_version(relative_stripped_path, &params.rev) {
            // Arc counts inside the cloned FileVersion will reflect sharing
            tracing::debug!(
                "Arc Counts : content: {} - summary: {}",
                Arc::strong_count(&version.content), // Count on the cloned Arc
                Arc::strong_count(&version.summary)  // Count on the cloned Arc
            );
            tracing::debug!("Path           : {}", relative_stripped_path.display());
            tracing::debug!("Revspec        : {}", params.rev);
            tracing::debug!("Commit         : {}", commit_id.short());
            tracing::debug!("Summary        : {}", version.summary);
            tracing::debug!("Content Length : {}", version.content.len());

            match diff_for_lsp(&rhs_path_buf, &version.content, &params.text_document.language_id) {
                Ok(diff_result) => {
                    if diff_result.has_reportable_change() {
                        ready(Ok(Some(lsp_ext::DiffRangesResponse {
                            ranges: diffresult_to_ranges(&diff_result),
                        })))
                        // match json3::print(&diff_result) {
                        //     Ok(json) => ready(Ok(Some(lsp_ext::DiffRangesResponse { ranges: json }))),
                        //     Err(err) => {
                        //         tracing::error!("Failed to serialize lsp_ext::DiffRangesResponse: {err}");
                        //         ready(Err(ResponseError::new(
                        //             ErrorCode::INTERNAL_ERROR,
                        //             format!("Failed to serialize lsp_ext::DiffRangesResponse: {err}"),
                        //         )))
                        //     }
                        // }
                    } else {
                        tracing::debug!("No changes detected for path {}", rhs_path_buf.display());
                        ready(Ok(Some(lsp_ext::DiffRangesResponse { ranges: vec![] })))
                    }
                }
                Err(err) => ready(Err(err)),
            }
        } else {
            tracing::debug!("Version {} not found for path {}", &params.rev, rhs_path_buf.display());
            ready(Err(ResponseError::new(
                ErrorCode::REQUEST_FAILED,
                format!("Version {} not found for path {}", &params.rev, rhs_path_buf.display()),
            )))
        }
    }

    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::unused_self)]
    fn on_did_open(&mut self, params: DidOpenTextDocumentParams) -> NotifyResult {
        tracing::debug!(
            "notif::DidOpenTextDocument - params.text_document.uri: {} - params.text_document.language_id: {} - params.text_document.version: {}",
            params.text_document.uri.to_file_path().unwrap_or_default().display(),
            params.text_document.language_id,
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

    #[allow(clippy::needless_pass_by_value)]
    fn on_update_config(&mut self, value: UpdateConfigEvent) -> NotifyResult {
        let mut config = Config::clone(&self.config);
        let mut errors = Vec::new();
        config.update(&value.0, &mut errors);

        tracing::info!("Updated config, errors: {errors:?}, config: {config:?}");

        if errors.is_empty() {
            self.config = Arc::new(config);
        } else {
            let msg = std::iter::once("Failed to apply some settings:")
                .chain(errors.iter().flat_map(|s| ["\n- ", s]))
                .collect::<String>();
            tracing::error!("{msg}");
            // self.client.show_message_ext(MessageType::ERROR, msg);
        }

        ControlFlow::Continue(())
    }

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

    /// Create a blocking task with a database snapshot as the input.
    // NB. `spawn_blocking` must be called immediately after snapshotting, so that the read guard
    // held in `Analysis` is sent out of the async runtime worker. Otherwise, the read guard
    // is held by the async runtime, and the next `apply_change` acquiring the write guard would
    // deadlock.
    fn spawn_with_snapshot<T: Send + 'static>(
        &self,
        f: impl FnOnce(StateSnapshot) -> T + Send + 'static,
    ) -> JoinHandle<T> {
        let snap = StateSnapshot {
            // analysis: self.host.snapshot(),
            // vfs: Arc::clone(&self.vfs),
            config: Arc::clone(&self.config),
            // cache_state: cache::CacheStateShared,
            // root_path: PathBuf::from(&self.root_path),
            root_path: self.root_path.clone(),
        };
        task::spawn_blocking(move || f(snap))
    }
}

trait RouterExt: BorrowMut<Router<Server>> {
    fn request_snap<R: Request>(
        &mut self,
        f: impl Fn(StateSnapshot, R::Params) -> Result<R::Result> + Send + Copy + UnwindSafe + 'static,
    ) -> &mut Self
    where
        R::Params: Send + UnwindSafe + 'static,
        R::Result: Send + 'static,
    {
        self.borrow_mut().request::<R, _>(move |this, params| {
            let task = this.spawn_with_snapshot(move |snap| with_catch_unwind(R::METHOD, move || f(snap, params)));
            async move { task.await.expect("Already catch_unwind").map_err(error_to_response) }
        });
        self
    }
}

impl RouterExt for Router<Server> {}

fn with_catch_unwind<T>(ctx: &str, f: impl FnOnce() -> Result<T> + UnwindSafe) -> Result<T> {
    static INSTALL_PANIC_HOOK: Once = Once::new();
    thread_local! {
        static PANIC_LOCATION: Cell<String> = const { Cell::new(String::new()) };
    }

    INSTALL_PANIC_HOOK.call_once(|| {
        let old_hook = panic::take_hook();
        panic::set_hook(Box::new(move |info| {
            let loc = info.location().map(|loc| loc.to_string()).unwrap_or_default();
            let backtrace = Backtrace::force_capture();
            PANIC_LOCATION.with(|inner| {
                inner.set(format!("Location: {loc:#}\nBacktrace: {backtrace:#}"));
            });
            old_hook(info);
        }));
    });

    match panic::catch_unwind(f) {
        Ok(ret) => ret,
        Err(payload) => {
            let reason = payload
                .downcast_ref::<String>()
                .map(|s| &**s)
                .or_else(|| payload.downcast_ref::<&str>().map(|s| &**s))
                .unwrap_or("unknown");
            let mut loc = PANIC_LOCATION.with(|inner| inner.take());
            if loc.is_empty() {
                loc = "Location: unknown".into();
            }
            tracing::error!("Panicked in {ctx}: {reason}\n{loc}");
            bail!("Panicked in {ctx}: {reason}\n{loc}");
        }
    }
}

fn error_to_response(err: anyhow::Error) -> ResponseError {
    if err.is::<Cancelled>() {
        return ResponseError::new(ErrorCode::REQUEST_CANCELLED, "Client cancelled");
    }
    match err.downcast::<ResponseError>() {
        Ok(resp) => resp,
        Err(err) => ResponseError::new(ErrorCode::INTERNAL_ERROR, err),
    }
}

#[derive(Debug)]
pub enum Cancelled {
    PendingWrite,
    PropagatedPanic,
}

impl Cancelled {
    pub(crate) fn throw(self) -> ! {
        // We use resume and not panic here to avoid running the panic
        // hook (that is, to avoid collecting and printing backtrace).
        std::panic::resume_unwind(Box::new(self));
    }

    /// Runs `f`, and catches any salsa cancellation.
    pub fn catch<F, T>(f: F) -> Result<T, Cancelled>
    where
        F: FnOnce() -> T + UnwindSafe,
    {
        match panic::catch_unwind(f) {
            Ok(t) => Ok(t),
            Err(payload) => match payload.downcast() {
                Ok(cancelled) => Err(*cancelled),
                Err(payload) => panic::resume_unwind(payload),
            },
        }
    }
}

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let why = match self {
            Cancelled::PendingWrite => "pending write",
            Cancelled::PropagatedPanic => "propagated panic",
        };
        f.write_str("cancelled because of ")?;
        f.write_str(why)
    }
}

impl std::error::Error for Cancelled {}

// pub fn on_did_open_custom(
//     snap: StateSnapshot,
//     params: lsp_ext::DidOpenTextDocumentCustomParams,
// ) -> Result<Option<lsp_ext::DiffRangesResponse>> {
//     tracing_to_json_pretty!(&params, "lsp_ext::DidOpenTextDocumentCustom");

//     let relative_stripped_path = Path::new(&params.text_document.uri)
//         .strip_prefix(&self.root_path)
//         .map_err(|err| {
//             tracing::error!("Failed to strip prefix: {err}");
//             ready(Err::<Option<lsp_ext::DiffRangesResponse>, ResponseError>(
//                 ResponseError::new(ErrorCode::REQUEST_FAILED, format!("Failed to strip prefix: {err}")),
//             ))
//         })
//         .unwrap();

//     tracing::debug!(
//         "params.rev: {} - relative_stripped_path: {}",
//         &params.rev,
//         relative_stripped_path.display()
//     );

//     // Handle the Result returned by populate_history
//     if let Err(err) = self.cache_state.populate_history(&params.rev, relative_stripped_path) {
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
//     if let Some((commit_id, version)) = self.cache_state.lookup_version(relative_stripped_path, &params.rev) {
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
