use std::sync::Arc;

use async_lsp::{
    ClientSocket, LanguageClient,
    lsp_types::{LogMessageParams, MessageType},
};
use tracing::Metadata;
use tracing_core::{Event, Level, Subscriber};
use tracing_subscriber::fmt::format::Writer as FmtWrite; // To avoid conflict with std::io::Write
use tracing_subscriber::{
    fmt::{
        FmtContext, FormatEvent, FormatFields, MakeWriter,
        time::{ChronoLocal, FormatTime},
    },
    registry::LookupSpan,
};

use crate::lsp::CHRONO_LOCAL;

#[derive(Debug, Clone)]
pub struct CustomEventFormatter {
    timer: ChronoLocal,
}

impl CustomEventFormatter {
    pub fn new() -> Self {
        Self {
            timer: ChronoLocal::new(CHRONO_LOCAL.into()),
        }
    }
}

impl<S, N> FormatEvent<S, N> for CustomEventFormatter
where
    S: Subscriber + for<'a> LookupSpan<'a>,
    N: for<'a> FormatFields<'a> + 'static, // N will be the field formatter from the subscriber builder
{
    fn format_event(
        &self,
        ctx: &FmtContext<'_, S, N>, // Context providing span info and field formatter
        mut writer: FmtWrite<'_>,   // Special writer from tracing_subscriber
        event: &Event<'_>,          // The event to format
    ) -> std::fmt::Result {
        // 1. Write time (e.g., 025-05-20 02:00:25)
        self.timer.format_time(&mut writer)?;
        // if self.timer.format_time(&mut writer).is_err() {
        //     writer.write_str("<unknown time>")?;
        // }

        // 2. Write space
        writer.write_char(' ')?;

        let meta = event.metadata();
        // 3. Write level and target with a final `::` (e.g., INFO my_crate::my_module::)
        write!(writer, " {} {}::", meta.level().as_str(), meta.target())?;

        // 4. Write Span Context (e.g., parent_span:current_span)
        if let Some(scope) = ctx.event_scope() {
            // Iterate from the outermost span to the current one
            let mut first = true;
            for span_ref in scope.from_root() {
                if first {
                    first = false;
                } else {
                    // Separator between span names
                    writer.write_char(':')?;
                }
                write!(writer, "{}", span_ref.name())?;
            }
        }

        // 5. Ensure a ` - ` before the message content
        writer.write_str(" - ")?;

        // 6. Write the event's message and other fields
        // This uses the default field formatter (N) configured on the subscriber,
        ctx.format_fields(writer.by_ref(), event)?;

        // Add a newline at the end of the log entry
        writeln!(writer)
    }
}

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
            _ = client_socket.log_message(LogMessageParams {
                typ: self.typ,
                message,
            });
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
