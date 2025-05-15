use std::sync::Arc;

use async_lsp::{
    ClientSocket, LanguageClient,
    lsp_types::{LogMessageParams, MessageType},
};
use tracing::{Level, Metadata};
use tracing_subscriber::fmt::MakeWriter;

const CHRONO_LOCAL: &str = "%FT%T";

pub fn setup_default_subscriber(client: ClientSocket) {
    let client_socket_writer = ClientSocketWriterMaker::new(client);

    tracing_subscriber::fmt()
        .with_max_level(Level::DEBUG)
        .with_ansi(false)
        .with_writer(client_socket_writer)
        .with_timer(tracing_subscriber::fmt::time::ChronoLocal::new(CHRONO_LOCAL.into()))
        .compact()
        .init();
}

// pub(crate) fn setup_panic_hook() {
//     // Set up a panic hook
//     std::panic::set_hook(Box::new(|panic_info| {
//         // Extract the panic message
//         let payload = panic_info.payload();
//         let message = if let Some(s) = payload.downcast_ref::<&str>() {
//             *s
//         } else if let Some(s) = payload.downcast_ref::<String>() {
//             &s[..]
//         } else {
//             "Unknown panic message"
//         };

//         // Get the location of the panic if available
//         let location = if let Some(location) = panic_info.location() {
//             format!(" at {}:{}", location.file(), location.line())
//         } else {
//             String::from("Unknown location")
//         };

//         // Capture the backtrace
//         let backtrace = Backtrace::capture();

//         // Log the panic information and backtrace
//         tracing::error!("Panic occurred{}: {}\nBacktrace:\n{:?}", location, message, backtrace);
//     }));
// }

pub struct ClientSocketWriterMaker {
    pub client_socket: Arc<ClientSocket>,
}

impl ClientSocketWriterMaker {
    pub fn new(client_socket: ClientSocket) -> Self {
        ClientSocketWriterMaker {
            client_socket: Arc::new(client_socket),
        }
    }
}

pub struct ClientSocketWriter {
    client_socket: Arc<ClientSocket>,
    typ: MessageType,
}

impl std::io::Write for ClientSocketWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        // Compute how many bytes to *actually* emit:
        // drop a trailing `\n`, and if that leaves a trailing `\r`, drop that too.
        let mut emit = buf.len();
        if emit > 0 && buf[emit - 1] == b'\n' {
            emit -= 1;
            if emit > 0 && buf[emit - 1] == b'\r' {
                emit -= 1;
            }
        }

        if emit > 0 {
            let message = String::from_utf8_lossy(&buf[..emit]).to_string();
            let mut client_socket = self.client_socket.as_ref();
            _ = client_socket.log_message(LogMessageParams { typ: self.typ, message });
        }

        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for ClientSocketWriterMaker {
    type Writer = ClientSocketWriter;

    fn make_writer(&'a self) -> Self::Writer {
        ClientSocketWriter {
            client_socket: self.client_socket.clone(),
            typ: MessageType::LOG,
        }
    }

    fn make_writer_for(&'a self, meta: &Metadata<'_>) -> Self::Writer {
        let typ = match *meta.level() {
            Level::ERROR => MessageType::ERROR,
            Level::WARN => MessageType::WARNING,
            Level::INFO => MessageType::INFO,
            Level::DEBUG | Level::TRACE => MessageType::LOG,
        };

        ClientSocketWriter {
            client_socket: self.client_socket.clone(),
            typ,
        }
    }
}
