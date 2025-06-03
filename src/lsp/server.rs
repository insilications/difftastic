use std::{
    backtrace::Backtrace,
    borrow::BorrowMut,
    cell::Cell,
    collections::HashMap,
    fmt,
    future::{ready, Future},
    ops::ControlFlow,
    panic,
    panic::{AssertUnwindSafe, UnwindSafe},
    path::{Path, PathBuf},
    sync::{Arc, Once},
};

use anyhow::{bail, Context, Error, Result};
use async_lsp::{router::Router, ClientSocket, ErrorCode, LanguageClient, ResponseError};
use gxhash::gxhash64;
use lsp_types::{
    notification as notif,
    notification::Notification,
    request::{
        Request, {self as req},
    },
    ConfigurationItem, ConfigurationParams, DidChangeConfigurationParams, DidChangeTextDocumentParams,
    DidCloseTextDocumentParams, DidOpenTextDocumentParams, DidSaveTextDocumentParams, InitializeParams,
    InitializeResult, InitializedParams, MessageType, Registration, RegistrationParams, ServerInfo, ShowMessageParams,
    Uri,
};
use tokio::{task, task::JoinHandle};
use tracing::{Instrument, Span};

use crate::{
    diff_for_lsp,
    display::json3::diffresult_to_ranges,
    lsp::{
        cache_git,
        capabilities::{negotiate_capabilities, NegotiatedCapabilities},
        config::{Config, WORKSPACE_CONFIG_KEY},
        lsp_ext,
        uri_ext::UriExt,
        vfs, GXHASH_SEED, LSP_SERVER_NAME, LSP_SERVER_VERSION,
    },
    tracing_to_json, tracing_to_json_pretty,
};

type NotifyResult = ControlFlow<async_lsp::Result<()>>;
struct UpdateConfigEvent(serde_json::Value);

#[derive(Debug, Default)]
struct OpenedFilesData {
    file_name: String,
}

#[derive(Debug)]
pub struct StateSnapshot {
    pub config: Arc<Config>,
    pub cache_state: cache_git::CacheStateShared,
    pub vfs: vfs::Vfs,
    pub root_path: PathBuf,
}

pub struct Server {
    // Immutable (mostly).
    client: ClientSocket,
    // States.
    config: Arc<Config>,
    cache_state: cache_git::CacheStateShared,
    root_path: PathBuf,
    capabilities: NegotiatedCapabilities,
    opened_files: HashMap<Uri, OpenedFilesData>,
    vfs: vfs::Vfs,
}

// TODO: review all usage of `.expect()` and `.unwrap()` in this file and convert them to proper error handling.
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
            // .request::<lsp_ext::DidOpenTextDocumentCustom, _>(Self::on_did_open_custom)
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
            cache_state: cache_git::CacheStateShared::new(),
            root_path: PathBuf::new(),
            capabilities: NegotiatedCapabilities::default(),
            opened_files: HashMap::new(),
            vfs: vfs::Vfs::default(),
        }
    }

    #[tracing::instrument(skip_all)]
    pub fn on_initialize(
        &mut self,
        params: InitializeParams,
    ) -> impl Future<Output = Result<InitializeResult, ResponseError>> {
        tracing_to_json!(&params);

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

        tracing::info!("Workspace Root Path: {}", self.root_path.display());

        tracing_to_json_pretty!(&server_caps, "Server Capabilities");
        tracing_to_json_pretty!(&self.capabilities, "Client Capabilities");

        if let Err(err) = self.cache_state.set_repo(&self.root_path) {
            tracing::error!("Failed to set cache_state repo for {}: {err}", self.root_path.display());
            return ready(Err(ResponseError::new(
                ErrorCode::REQUEST_FAILED,
                format!("Failed to set cache_state repo for {}: {err}", self.root_path.display()),
            )));
        }

        *Arc::get_mut(&mut self.config).expect("No concurrent access yet") = Config::new(self.root_path.clone());

        if let Some(options) = params.initialization_options {
            if options.as_object().filter(|o| !o.is_empty()).is_some() {
                tracing::debug!("initialization_options: {options}");
                #[allow(unused_must_use)]
                self.on_update_config(UpdateConfigEvent(options));
            }
        }

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

    #[tracing::instrument(skip_all)]
    fn on_initialized(&mut self, _params: InitializedParams) -> NotifyResult {
        tracing::debug!("notif::Initialized");

        if self.capabilities.workspace_configuration {
            tokio::spawn({
                let mut client = self.client.clone();
                async move {
                    if let Err(err) = Self::register_did_change_configuration(&mut client).await {
                        client.show_message_ext(
                            MessageType::ERROR,
                            format!("Failed to register DidChangeConfiguration: {err:#}"),
                        );
                    }
                }
                .in_current_span()
            });
        }

        ControlFlow::Continue(())
    }

    #[tracing::instrument(skip_all)]
    fn on_did_open(&mut self, params: DidOpenTextDocumentParams) -> NotifyResult {
        let uri: &Uri = &params.text_document.uri;
        let file_pathbuf = if let Some(cow_path) = uri.to_file_path() {
            cow_path.into_owned()
        } else {
            tracing::error!("Failed to convert URI to file path: {:?}", *uri);
            return ControlFlow::Continue(()); // Return early from on_did_open
        };

        tracing::debug!(
            "notif::DidOpenTextDocument - params.text_document.uri: {} - params.text_document.language_id: {} - params.text_document.version: {}",
            &file_pathbuf.display(),
            &params.text_document.language_id,
            &params.text_document.version
        );

        self.opened_files.insert(uri.clone(), OpenedFilesData::default());

        self.spawn_with_snapshot(move |snap| {
            let ret = with_catch_unwind(
                "on_did_open",
                // Use AssertUnwindSafe to allow catching panics in the closure because the compiler concludes that
                // `lsp_types::Uri`, captured in the closure, is not `RefUnwindSafe` because it (via
                // `fluent_uri::Uri`) contains an `UnsafeCell` for which `RefUnwindSafe` is not
                // implemented. While `Cell<T>` (where `T` is `RefUnwindSafe`) *is*
                // `RefUnwindSafe`, if the `fluent_uri::Uri` type itself doesn't have an explicit
                // (possibly `unsafe`) implementation of `RefUnwindSafe`, the compiler conservatively assumes it's
                // not. This often happens when a crate author hasn't audited their type for
                // unwind safety or hasn't added the `unsafe impl RefUnwindSafe for Uri {}`
                // declaration.
                AssertUnwindSafe(|| {
                    let relative_stripped_path = file_pathbuf
                        .strip_prefix(&snap.root_path)
                        .with_context(|| format!("Failed to strip prefix for {}", &file_pathbuf.display()))?;

                    let blame_highlighting_parent_level: &str = &snap.config.blame_highlighting_parent_level;

                    tracing::debug!(
                        "blame_highlighting_parent_level: {} - relative_stripped_path: {}",
                        &blame_highlighting_parent_level,
                        &relative_stripped_path.display()
                    );

                    // Handle the Result returned by populate_history
                    match snap.cache_state.populate_history(blame_highlighting_parent_level, relative_stripped_path) {
                        Ok(cache_git::PopulateHistoryResult::AlreadyPopulated) => {
                            tracing::debug!(
                                "History already populated for path: {} with revspec: {}",
                                relative_stripped_path.display(),
                                blame_highlighting_parent_level
                            );
                        }
                        Ok(cache_git::PopulateHistoryResult::NewlyPopulated) => {
                            tracing::debug!(
                                "History newly populated for path: {} with revspec: {}",
                                relative_stripped_path.display(),
                                blame_highlighting_parent_level
                            );
                        }
                        Ok(cache_git::PopulateHistoryResult::NoHistory) => {
                            tracing::debug!(
                                "No history to populate for path: {} with revspec: {}",
                                relative_stripped_path.display(),
                                blame_highlighting_parent_level
                            );
                        }
                        Err(err) => {
                            tracing::error!("Failed to populate history: {err}");
                            return Err(Error::new(ResponseError::new(
                                ErrorCode::REQUEST_FAILED,
                                format!("Failed to populate history: {err}"),
                            )));
                        }
                    }

                    let txt_bytes: &[u8] = params.text_document.text.as_bytes();
                    let txt_hash: u64 = gxhash64(txt_bytes, GXHASH_SEED);
                    tracing::debug!("txt_hash: {:#x}", txt_hash);

                    snap.vfs.open(params.text_document.uri, params.text_document.version, &params.text_document.text);

                    // let txt = snap.vfs.get_text(uri).unwrap_or_default();
                    // tracing::debug!("snap.vfs.get_text: {}", txt);

                    // Note: lookup_version now returns an owned FileVersion due to cloning
                    if let Some((commit_id, version)) =
                        snap.cache_state.lookup_version(relative_stripped_path, blame_highlighting_parent_level)
                    {
                        // Arc counts inside the cloned FileVersion will reflect sharing
                        tracing::debug!(
                            "Arc Counts : content: {} - summary: {}",
                            Arc::strong_count(&version.content), // Count on the cloned Arc
                            version
                                .maybe_summary
                                .as_ref()
                                .map_or_else(|| "<no summary>".to_owned(), |s| Arc::strong_count(s).to_string())
                        );
                        tracing::debug!("Path           : {}", &relative_stripped_path.display());
                        tracing::debug!("Revspec        : {}", &blame_highlighting_parent_level);
                        // &commit_id.short() VS commit_id.short()
                        tracing::debug!("Commit         : {}", &commit_id.short());
                        tracing::debug!(
                            "Summary        : {}",
                            version.maybe_summary.as_ref().map_or("<no summary>", |s| s)
                        );
                        tracing::debug!("Content Hash   : {:#x}", &version.content_hash);
                        tracing::debug!("Content Length : {}", &version.content.len());
                    }

                    Ok(())
                }),
            );
            match ret {
                Ok(()) => {
                    // let _: Result<_, _> = client.emit(UpdateDiagnostics(version, diags));
                    tracing::debug!("notif::DidOpenTextDocument - FINISHED");
                }
                // Ignore cancellations caused by editing.
                Err(err) if err.is::<Cancelled>() => {}
                Err(err) => tracing::error!("Failed to update diagnostics: {err:#}"),
            }
        });

        ControlFlow::Continue(())
    }

    #[allow(clippy::needless_pass_by_value)]
    #[allow(clippy::unused_self)]
    fn on_did_close(&mut self, params: DidCloseTextDocumentParams) -> NotifyResult {
        tracing::debug!(
            "notif::DidCloseTextDocument - params.text_document.uri: {}",
            &params.text_document.uri.to_file_path().unwrap_or_default().display()
        );

        self.opened_files.remove(&params.text_document.uri);

        // Must clear all highlights for the file by sending a notification to the client.

        ControlFlow::Continue(())
    }

    #[allow(clippy::needless_pass_by_value)]
    fn on_did_change(&mut self, params: DidChangeTextDocumentParams) -> NotifyResult {
        let uri: &Uri = &params.text_document.uri;
        let file_path = uri.to_file_path().unwrap_or_default();
        let file_path_display = file_path.display();

        let txt1 = self.vfs.get_text(uri).unwrap_or_default();
        // tracing::debug!("notif::DidChangeTextDocument - 1 self.vfs.get_text: {}", txt1);
        let txt1_bytes: &[u8] = txt1.as_bytes();
        let txt1_hash = gxhash64(txt1_bytes, GXHASH_SEED);
        tracing::debug!("notif::DidChangeTextDocument - txt1_hash: {:#x}", txt1_hash);

        tracing::debug!(
            "notif::DidChangeTextDocument - params.text_document.uri: {} - params.text_document.version: {} - params.content_changes.len(): {}",
            file_path_display,
            params.text_document.version,
            params.content_changes.len()
        );
        for c in &params.content_changes {
            tracing::debug!("{} - content_change.range: {:?}", file_path_display, c.range);
            tracing::debug!("{} - content_change.text: {}", file_path_display, c.text);
        }

        if let Err(e) = self.vfs.apply_changes(uri, params.text_document.version, &params.content_changes) {
            tracing::error!("Failed to apply changes for {}: {e:#}", file_path_display);
        }

        let txt2 = self.vfs.get_text(uri).unwrap_or_default();
        // tracing::debug!("notif::DidChangeTextDocument - 2 self.vfs.get_text: {}", txt2);
        let txt2_bytes: &[u8] = txt2.as_bytes();
        let txt2_hash = gxhash64(txt2_bytes, GXHASH_SEED);
        tracing::debug!("notif::DidChangeTextDocument - txt2_hash: {:#x}", txt2_hash);

        ControlFlow::Continue(())
    }

    #[allow(clippy::unused_self)]
    fn on_did_save(&mut self, _params: DidSaveTextDocumentParams) -> NotifyResult {
        tracing::debug!("notif::DidSaveTextDocument");

        ControlFlow::Continue(())
    }

    #[tracing::instrument(skip_all)]
    fn on_did_change_configuration(&mut self, params: DidChangeConfigurationParams) -> NotifyResult {
        tracing_to_json_pretty!(&params);
        self.spawn_reload_config();

        ControlFlow::Continue(())
    }

    #[tracing::instrument(skip_all)]
    fn spawn_reload_config(&self) {
        if self.capabilities.workspace_configuration {
            tokio::spawn({
                let mut client = self.client.clone();
                async move {
                    match client
                        .configuration(ConfigurationParams {
                            items: vec![ConfigurationItem {
                                scope_uri: None,
                                section: Some(WORKSPACE_CONFIG_KEY.into()),
                            }],
                        })
                        .await
                    {
                        Ok(mut v) => {
                            let v = v.pop().unwrap_or_default();
                            tracing::debug!("Updating config: {v:?}");
                            let _: Result<_, _> = client.emit(UpdateConfigEvent(v));
                        }
                        Err(err) => {
                            client
                                .show_message_ext(MessageType::ERROR, format_args!("Failed to update config: {err:#}"));
                        }
                    }
                }
                .in_current_span()
            });
        }
    }

    #[tracing::instrument(skip_all)]
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
            self.client.show_message_ext(MessageType::ERROR, msg);
        }

        ControlFlow::Continue(())
    }

    #[tracing::instrument(skip_all)]
    async fn register_did_change_configuration(client: &mut ClientSocket) -> Result<()> {
        // let settings_json_value = serde_json::to_value(WORKSPACE_CONFIG_KEY)
        //     .with_context(|| format!("Failed to serialize WORKSPACE_CONFIG_KEY ('{WORKSPACE_CONFIG_KEY}') to
        // JSON",))?;

        // let did_change_params_for_registration = DidChangeConfigurationParams {
        //     settings: settings_json_value,
        // };

        // let register_options_json_value = serde_json::to_value(&did_change_params_for_registration)
        //     .context("Failed to serialize DidChangeConfigurationParams to JSON for registration options")?;

        let params = RegistrationParams {
            registrations: vec![Registration {
                id: notif::DidChangeConfiguration::METHOD.into(),
                method: notif::DidChangeConfiguration::METHOD.into(),
                register_options: None,
                // register_options: Some(register_options_json_value),
            }],
        };

        client
            .register_capability(params)
            .await
            .context("Failed to register capability for receiving DidChangeConfiguration")?;

        // if let Err(err) = client.register_capability(params).await {
        //     client.show_message_ext(
        //         MessageType::ERROR,
        //         format!("Failed to register capability for receiving DidChangeConfiguration: {err:#}"),
        //     );
        // }

        tracing::debug!("Registered DidChangeConfiguration");
        Ok(())
    }

    fn spawn_with_snapshot<T: Send + 'static>(
        &self,
        f: impl FnOnce(StateSnapshot) -> T + Send + 'static,
    ) -> JoinHandle<T> {
        let snap = StateSnapshot {
            // analysis: self.host.snapshot(),
            vfs: self.vfs.clone(),
            config: Arc::clone(&self.config),
            cache_state: self.cache_state.clone(),
            // cache_state: cache::CacheStateShared,
            // root_path: PathBuf::from(&self.root_path),
            root_path: self.root_path.clone(),
        };
        let span = Span::current();
        task::spawn_blocking(move || {
            let _enter = span.enter();
            f(snap)
        })
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

trait ClientExt: BorrowMut<ClientSocket> {
    #[inline]
    fn show_message_ext(&mut self, typ: MessageType, msg: impl fmt::Display) {
        match typ {
            MessageType::ERROR => tracing::error!("{msg}"),
            MessageType::WARNING => tracing::warn!("{msg}"),
            MessageType::INFO => tracing::info!("{msg}"),
            MessageType::LOG => tracing::debug!("{msg}"),
            _ => tracing::debug!("{msg}"),
        }
        let _: Result<_, _> = self.borrow_mut().show_message(ShowMessageParams {
            typ,
            message: msg.to_string(),
        });
    }
}

impl ClientExt for ClientSocket {}

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

    // Runs `f`, and catches any cancellation (e.g. from salsa).
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

#[allow(clippy::needless_pass_by_value)]
pub fn on_did_open_custom(
    snap: StateSnapshot,
    params: lsp_ext::DidOpenTextDocumentCustomParams,
) -> Result<Option<lsp_ext::DiffRangesResponse>> {
    tracing_to_json_pretty!(&params, "lsp_ext::DidOpenTextDocumentCustom");

    let relative_stripped_path = Path::new(&params.text_document.uri)
        .strip_prefix(&snap.root_path)
        .map_err(|err| {
            tracing::error!("Failed to strip prefix: {err}");
            ready(Err::<Option<lsp_ext::DiffRangesResponse>, ResponseError>(ResponseError::new(
                ErrorCode::REQUEST_FAILED,
                format!("Failed to strip prefix: {err}"),
            )))
        })
        .unwrap();

    tracing::debug!("params.rev: {} - relative_stripped_path: {}", &params.rev, relative_stripped_path.display());

    // Handle the Result returned by populate_history
    if let Err(err) = snap.cache_state.populate_history(&params.rev, relative_stripped_path) {
        tracing::error!("Failed to populate history: {err}");
        return Err(anyhow::Error::new(ResponseError::new(
            ErrorCode::REQUEST_FAILED,
            format!("Failed to populate history: {err}"),
        )));
    }
    // let response = lsp_ext::DiffRangesResponse { ranges: vec![] };

    // snap.cache_state
    //     .as_ref()
    //     .unwrap()
    //     .iterate_path_versions(&relative_stripped_path);

    let rhs_path_buf = PathBuf::from(&relative_stripped_path);

    // Note: lookup_version now returns an owned FileVersion due to cloning
    if let Some((commit_id, version)) = snap.cache_state.lookup_version(relative_stripped_path, &params.rev) {
        // Arc counts inside the cloned FileVersion will reflect sharing
        tracing::debug!(
            "Arc Counts : content: {} - summary: {}",
            Arc::strong_count(&version.content), // Count on the cloned Arc
            version
                .maybe_summary
                .as_ref()
                .map_or_else(|| "<no summary>".to_owned(), |s| Arc::strong_count(s).to_string())
        );
        tracing::debug!("Path           : {}", &relative_stripped_path.display());
        tracing::debug!("Revspec        : {}", &params.rev);
        // &commit_id.short() VS commit_id.short()
        tracing::debug!("Commit         : {}", &commit_id.short());
        tracing::debug!("Summary        : {}", version.maybe_summary.as_ref().map_or("<no summary>", |s| s));
        tracing::debug!("Content Hash   : {:#x}", &version.content_hash);
        tracing::debug!("Content Length : {}", &version.content.len());

        match diff_for_lsp(&rhs_path_buf, &version.content, &params.text_document.language_id) {
            Ok(diff_result) => {
                if diff_result.has_reportable_change() {
                    Ok(Some(lsp_ext::DiffRangesResponse {
                        ranges: diffresult_to_ranges(&diff_result),
                    }))
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
                    Ok(Some(lsp_ext::DiffRangesResponse {
                        ranges: vec![],
                    }))
                }
            }
            Err(err) => Err(Error::new(ResponseError::new(ErrorCode::REQUEST_FAILED, format!("{err}")))),
        }
    } else {
        tracing::debug!("Version {} not found for path {}", &params.rev, rhs_path_buf.display());
        Err(Error::new(ResponseError::new(
            ErrorCode::REQUEST_FAILED,
            format!("Version {} not found for path {}", &params.rev, rhs_path_buf.display()),
        )))
    }
}

fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[tracing::instrument]
fn test1() {
    let _a = add(5, 6);
    let _b: Vec<_> = [1, 2, 3].iter().map(|item| item).map(|item| item).map(|item| item).collect();
}

fn test2() {
    let _a = add(5, 6);
    let _b: Vec<_> = [1, 2, 3].iter().map(|item| item).map(|item| item).map(|item| item).collect();
}

// #[tracing::instrument]
// fn my_function() {
//     let t = "Hello, world!";
//     spawn(move || {
//         println!("Function executed: {t}");
//     });
// }

// fn spawn(f: impl FnOnce()) {
//     f();
// }

// pub struct MyStruck {
//     uri: String,
//     porra: String,
// }

// fn spawn(f: impl FnOnce()) {
//     f();
// }

// fn main() {
//     let mut my_struck = MyStruck {
//         uri: "uri".into(),
//         porra: "PORRA".into(),
//     };
//     let mut uri1: &mut str = &mut (my_struck.uri);

//     println!("uri1: {}", uri1);
//     my_struck.porra.push('d');
//     my_struck.uri.push('d');
//     println!("my_struck.uri: {}", my_struck.uri);

//     spawn(move || {
//         println!("my_struck.porra: {}", my_struck.porra);
//     });
// }
