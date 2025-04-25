use std::future::ready;
use std::ops::ControlFlow;
use std::time::Duration;

// Change the import to directly access the external crate since custom_lsp_types is its own crate
// use difftastic::custom_lsp_types::{MyInitializeResult, MyServerCapabilities};

use async_lsp::ClientSocket;
use async_lsp::client_monitor::ClientProcessMonitorLayer;
use async_lsp::concurrency::ConcurrencyLayer;
use async_lsp::panic::CatchUnwindLayer;
use async_lsp::router::Router;
use async_lsp::server::LifecycleLayer;
use async_lsp::tracing::TracingLayer;
use lsp_types::{
    Diff, InitializeResult, InitializedParams, MessageType, PositionEncodingKind, Range, ServerCapabilities,
    ServerInfo, ShowMessageParams, TextDocumentSyncCapability, TextDocumentSyncKind, TextDocumentSyncOptions,
    TextDocumentSyncSaveOptions, notification,
    request::{self as req},
};
use tower::ServiceBuilder;
use tracing::{Level, debug, info};

const LSP_SERVER_NAME: &str = "difftastic-lsp";
const LSP_SERVER_VERSION: &str = "0.1.0";

struct ServerState {
    client: ClientSocket,
    counter: i32,
}

struct TickEvent;

pub(crate) async fn start_lsp() {
    let (server, _) = async_lsp::MainLoop::new_server(|client| {
        tokio::spawn({
            let client = client.clone();
            async move {
                let mut interval = tokio::time::interval(Duration::from_secs(1));
                loop {
                    interval.tick().await;
                    if client.emit(TickEvent).is_err() {
                        break;
                    }
                }
            }
        });

        let mut router = Router::new(ServerState {
            client: client.clone(),
            counter: 0,
        });
        router
            .request::<req::Initialize, _>(|_, params| async move {
                // eprintln!("Initialize with {params:?}");
                // info!("Initialize with {params:?}");
                match serde_json::to_string(&params) {
                    Ok(json_params) => info!(params = %json_params, "Initialize with"),
                    Err(_) => debug!(raw_params = ?params, "Raw initialize with"),
                }
                Ok(InitializeResult {
                    capabilities: ServerCapabilities {
                        position_encoding: Some(PositionEncodingKind::UTF16),
                        text_document_sync: Some(TextDocumentSyncCapability::Options(TextDocumentSyncOptions {
                            open_close: Some(true),
                            change: Some(TextDocumentSyncKind::NONE),
                            will_save: None,
                            will_save_wait_until: None,
                            save: Some(TextDocumentSyncSaveOptions::Supported(true)),
                        })),
                        // hover_provider: Some(HoverProviderCapability::Simple(true)),
                        // definition_provider: Some(OneOf::Left(true)),
                        // diff: Some(true),
                        // diff: None,
                        experimental: Some(serde_json::json!({
                            "diff": true,
                        })),
                        ..ServerCapabilities::default()
                    },
                    server_info: Some(ServerInfo {
                        name: LSP_SERVER_NAME.into(),
                        version: Some(LSP_SERVER_VERSION.into()),
                    }),
                    // offset_encoding: Some("utf-8".to_string()),
                    offset_encoding: None,
                })
                // Ok(MyInitializeResult {
                //     capabilities: MyServerCapabilities {
                //         standard: ServerCapabilities {
                //             // hover_provider: Some(HoverProviderCapability::Simple(true)),
                //             // definition_provider: Some(OneOf::Left(true)),
                //             ..ServerCapabilities::default()
                //         },
                //         diff: Some(true),
                //     },
                //     server_info: Some(ServerInfo {
                //         name: "difftastic-lsp".to_string(),
                //         version: Some("0.1.0".to_string()),
                //     }),
                //     offset_encoding: Some("utf-32".to_string()),
                // })
            })
            // .request::<req::DiffRequest, _>(|st, _| {
            //     let client = st.client.clone();
            //     // let counter = st.counter;
            //     async move {
            //         tokio::time::sleep(Duration::from_secs(1)).await;
            //         client
            //             .notify::<notification::ShowMessage>(ShowMessageParams {
            //                 typ: MessageType::INFO,
            //                 message: "request::DiffRequest".into(),
            //             })
            //             .unwrap();
            //         Ok(Some(Diff {
            //             ranges: Some(Vec::from([Range {
            //                 start: lsp_types::Position { line: 0, character: 0 },
            //                 end: lsp_types::Position { line: 1, character: 1 },
            //             }])),
            //         }))
            //     }
            // })
            .request::<req::Shutdown, _>(|_, _| {
                info!("req::Shutdown");
                ready(Ok(()))
            })
            // .request::<request::GotoDefinition, _>(|_, _| async move { unimplemented!("Not yet implemented!") })
            .notification::<notification::Initialized>(|_, _params: InitializedParams| {
                info!("notification::Initialized");
                ControlFlow::Continue(())
            })
            .notification::<notification::Exit>(|_, _| {
                info!("notification::Exit");
                ControlFlow::Break(Ok(()))
            })
            .notification::<notification::DidChangeConfiguration>(|_, _| {
                info!("notification::DidChangeConfiguration");
                ControlFlow::Continue(())
            })
            .notification::<notification::DidOpenTextDocument>(|_, params| {
                info!("notification::DidOpenTextDocument with params: {:?}", params);
                ControlFlow::Continue(())
            })
            .notification::<notification::DidChangeTextDocument>(|_, _| {
                info!("notification::DidChangeTextDocument");
                ControlFlow::Continue(())
            })
            .notification::<notification::DidCloseTextDocument>(|_, _| {
                info!("notification::DidCloseTextDocument");
                ControlFlow::Continue(())
            })
            .notification::<notification::DidSaveTextDocument>(|_, _| {
                info!("notification::DidSaveTextDocument");
                ControlFlow::Continue(())
            })
            .event::<TickEvent>(|st, _| {
                // info!("tick");
                st.counter += 1;
                ControlFlow::Continue(())
            });

        ServiceBuilder::new()
            .layer(TracingLayer::default())
            .layer(LifecycleLayer::default())
            .layer(CatchUnwindLayer::default())
            .layer(ConcurrencyLayer::default())
            .layer(ClientProcessMonitorLayer::new(client))
            .service(router)
    });

    tracing_subscriber::fmt()
        .with_max_level(Level::INFO)
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .init();

    // Prefer truly asynchronous piped stdin/stdout without blocking tasks.
    #[cfg(unix)]
    let (stdin, stdout) = (
        async_lsp::stdio::PipeStdin::lock_tokio().unwrap(),
        async_lsp::stdio::PipeStdout::lock_tokio().unwrap(),
    );
    // Fallback to spawn blocking read/write otherwise.
    #[cfg(not(unix))]
    let (stdin, stdout) = (
        tokio_util::compat::TokioAsyncReadCompatExt::compat(tokio::io::stdin()),
        tokio_util::compat::TokioAsyncWriteCompatExt::compat_write(tokio::io::stdout()),
    );

    server.run_buffered(stdin, stdout).await.unwrap();
}
