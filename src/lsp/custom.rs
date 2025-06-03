use anyhow::{Context, Result};
use async_lsp::{
    client_monitor::ClientProcessMonitorLayer,
    concurrency::ConcurrencyLayer,
    panic::CatchUnwindLayer,
    server::LifecycleLayer,
    // tracing::TracingLayer,
};
use tower::ServiceBuilder;
use tracing::Level;

use crate::lsp::{logging, server::Server, CHRONO_LOCAL};
//  The file length limit. Files larger than this will be rejected from all interactions.
//  The hard limit is `u32::MAX` due to following conditions.
//  - The parser and the `rowan` library uses `u32` based indices.
//  - `vfs::LineMap` uses `u32` based indices.

//  Since large files can cause significant performance issues, also to
//  be away from off-by-n errors, here's an arbitrary chosen limit: 128MiB.

//  If you have any real world usages for files larger than this, please file an issue.
// pub const MAX_FILE_LEN: usize = 128 << 20;

// pub(crate) trait UrlExt: Sized {
// fn to_vfs_path(&self) -> VfsPath;
// fn from_vfs_path(path: &VfsPath) -> Self;
// }

// impl UrlExt for Url {
//     fn to_vfs_path(&self) -> VfsPath {
//         // `Url::to_file_path` doesn't do schema check.
//         if self.scheme() == "file" {
//             if let Ok(path) = self.to_file_path() {
//                 return path.into();
//             }
//             tracing::warn!("Ignore invalid file URI: {self}");
//         }
//         VfsPath::Virtual(self.as_str().to_owned())
//     }

//     fn from_vfs_path(vpath: &VfsPath) -> Self {
//         match vpath {
//             VfsPath::Path(path) => Url::from_file_path(path).expect("VfsPath must be absolute"),
//             VfsPath::Virtual(uri) => uri.parse().expect("Virtual path must be an URI"),
//         }
//     }
// }

pub async fn run_lsp_server_stdio() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_ansi(false)
        .with_writer(std::io::stderr)
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::new(CHRONO_LOCAL.into()))
        .with_target(true)
        // .compact()
        .event_format(logging::CustomEventFormatter::new())
        .init();

    let concurrency = match std::thread::available_parallelism() {
        // Double the concurrency limit since many handlers are blocking anyway.
        Ok(n) => n.saturating_mul(2.try_into().expect("2 is not 0")),
        Err(err) => {
            tracing::error!("Failed to get available parallelism: {err}");
            2.try_into().expect("2 is not 0")
        }
    };
    tracing::debug!("Max concurrent requests: {concurrency}");

    // let stdin = PipeStdin::lock_tokio().context("stdin is not pipe-like")?;
    // let stdout = PipeStdout::lock_tokio().context("stdout is not pipe-like")?;

    // Prefer truly asynchronous piped stdin/stdout without blocking tasks.
    #[cfg(unix)]
    let (stdin, stdout) = (
        async_lsp::stdio::PipeStdin::lock_tokio().context("stdin is not pipe-like")?,
        async_lsp::stdio::PipeStdout::lock_tokio().context("stdout is not pipe-like")?,
    );
    // Fallback to spawn blocking read/write otherwise.
    #[cfg(not(unix))]
    let (stdin, stdout) = (
        tokio_util::compat::TokioAsyncReadCompatExt::compat(tokio::io::stdin()),
        tokio_util::compat::TokioAsyncWriteCompatExt::compat_write(tokio::io::stdout()),
    );

    let (server_main_loop, _) = async_lsp::MainLoop::new_server(|client| {
        ServiceBuilder::new()
            // .layer(TracingLayer::default())
            // .layer(MeterLayer)
            .layer(LifecycleLayer::default())
            .layer(CatchUnwindLayer::default())
            .layer(ConcurrencyLayer::new(concurrency))
            .layer(ClientProcessMonitorLayer::new(client.clone()))
            .service(Server::new_router(client))
    });

    // logging::setup_default_subscriber(client);
    Ok(server_main_loop.run_buffered(stdin, stdout).await?)
}
