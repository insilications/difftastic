mod cache_git;
mod capabilities;
mod config;
pub mod custom;
mod logging;
mod lsp_ext;
mod meter;
mod server;
mod uri_ext;
mod vfs;

pub const CHRONO_LOCAL: &str = "%F %T";
pub const LSP_SERVER_NAME: &str = "difftastic-lsp";
pub const LSP_SERVER_VERSION: &str = "0.1.0";
pub const GXHASH_SEED: i64 = 0x87FA_3129;

/// `tracing_to_json` (compact JSON)
#[macro_export]
macro_rules! tracing_to_json {
    ($value:expr, $label:literal $(,)?) => {{
        if ::tracing::enabled!(::tracing::Level::DEBUG) {
            let __ttj_val = &$value;

            match ::serde_json::to_string(__ttj_val) {
                Ok(__ttj_json) => {
                    ::tracing::debug!("{} - {}: {}", $label, ::core::stringify!($value), __ttj_json);
                }
                Err(__err) => {
                    ::tracing::debug!(
                        "Failed to serialise `{}` ({}): {:?}",
                        ::core::stringify!($value),
                        __err,
                        __ttj_val
                    );
                }
            }
        }
    }};

    ($value:expr $(,)?) => {{
        if ::tracing::enabled!(::tracing::Level::DEBUG) {
            let __ttj_val = &$value;

            match ::serde_json::to_string(__ttj_val) {
                Ok(__ttj_json) => {
                    ::tracing::debug!("{}: {}", ::core::stringify!($value), __ttj_json);
                }
                Err(__err) => {
                    ::tracing::debug!(
                        "Failed to serialise `{}` ({}): {:?}",
                        ::core::stringify!($value),
                        __err,
                        __ttj_val
                    );
                }
            }
        }
    }};
}

/// `tracing_to_json_pretty` (pretty-printed JSON)
#[macro_export]
macro_rules! tracing_to_json_pretty {
    ($value:expr, $label:literal $(,)?) => {{
        if ::tracing::enabled!(::tracing::Level::DEBUG) {
            let __ttj_val = &$value;

            match ::serde_json::to_string_pretty(__ttj_val) {
                Ok(__ttj_json) => {
                    ::tracing::debug!("{} - {}: {}", $label, ::core::stringify!($value), __ttj_json);
                }
                Err(__err) => {
                    ::tracing::debug!(
                        "Failed to serialise `{}` ({}): {:?}",
                        ::core::stringify!($value),
                        __err,
                        __ttj_val
                    );
                }
            }
        }
    }};

    ($value:expr $(,)?) => {{
        if ::tracing::enabled!(::tracing::Level::DEBUG) {
            let __ttj_val = &$value;

            match ::serde_json::to_string_pretty(__ttj_val) {
                Ok(__ttj_json) => {
                    ::tracing::debug!("{}: {}", ::core::stringify!($value), __ttj_json);
                }
                Err(__err) => {
                    ::tracing::debug!(
                        "Failed to serialise `{}` ({}): {:?}",
                        ::core::stringify!($value),
                        __err,
                        __ttj_val
                    );
                }
            }
        }
    }};
}
