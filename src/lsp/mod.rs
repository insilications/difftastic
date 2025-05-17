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

#[doc(hidden)]
#[macro_export] // Implementation: label is *any* expression
macro_rules! __tracing_to_json_impl {
    ($value:expr, $label:expr) => {{
        // Evaluate the expression only once
        if ::tracing::enabled!(::tracing::Level::DEBUG) {
            let __ttj_val = &$value;

            match ::serde_json::to_string(__ttj_val) {
                Ok(__ttj_json) => {
                    ::tracing::debug!(
                        /* ── structured fields ── */
                        label = $label,
                        expr  = ::core::stringify!($value),
                        json  = %__ttj_json,
                        /* ── human-readable message ── */
                        "{} - {}: {}",
                        $label,
                        ::core::stringify!($value),
                        __ttj_json
                    );
                }
                Err(__err) => {
                    ::tracing::debug!(
                        label = $label,
                        expr  = ::core::stringify!($value),
                        error = %__err,
                        value = ?__ttj_val,
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

#[macro_export] // Public entry point (label optional)
macro_rules! tracing_to_json {
    // ❶ Original “value + literal label” form keeps the old syntax intact
    ($value:expr, $label:literal $(,)?) => {
        $crate::__tracing_to_json_impl!($value, $label)
    };

    // ❷ NEW form: only the value – we generate a fallback label automatically
    ($value:expr $(,)?) => {
        $crate::__tracing_to_json_impl!($value, ::core::stringify!($value))
    };
}

// ──────────────────────────────────────────────────────────────────────────────
// tracing_to_json_pretty (DEBUG, pretty-printed JSON)
// ──────────────────────────────────────────────────────────────────────────────

#[doc(hidden)]
#[macro_export]
macro_rules! __tracing_to_json_pretty_impl {
    ($value:expr, $label:expr) => {{
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
}

#[macro_export]
macro_rules! tracing_to_json_pretty {
    // ❶ Explicit label
    ($value:expr, $label:literal $(,)?) => {
        $crate::__tracing_to_json_pretty_impl!($value, $label)
    };

    // ❷ Implicit label = token stringification of the value
    ($value:expr $(,)?) => {
        $crate::__tracing_to_json_pretty_impl!($value, ::core::stringify!($value))
    };
}

// #[macro_export]
// macro_rules! tracing_to_json {
//     // Single arm handles all cases: &ident, ident, expr
//     ($value:expr, $label:literal $(,)?) => {{
//         // Optimization: Check log level *before* doing any work
//         if ::tracing::enabled!(::tracing::Level::DEBUG) {
//             // Evaluate the expression once, only if the level is enabled
//             let __ttj_val = &$value;

//             match ::serde_json::to_string(__ttj_val) {
//                 Ok(__ttj_json) => {
//                     // Optimization: Use structured logging fields
//                     ::tracing::debug!(
//                         // --- Structured Fields (Sigils % and ? work here) ---
//                         label = $label,
//                         // Use stringify! directly, handles &ident correctly
//                         expr = ::core::stringify!($value),
//                         // Use % formatting for the JSON string (which implements Display)
//                         json = %__ttj_json,
//                         // Message template using fields
//                         "{} - {}: {}",
//                        $label, // Corresponds to the first {}
//                         ::core::stringify!($value), // Corresponds to the second {}
//                         __ttj_json // Corresponds to the third {} (using Display)
//                     );
//                 }
//                 Err(__err) => {
//                     // Log failure using structured fields
//                     // Combine error and fallback into one event for efficiency
//                     ::tracing::debug!(
//                         // --- Structured Fields (Sigils % and ? work here) ---
//                         label = $label,
//                         expr = ::core::stringify!($value),
//                         // Use % formatting for the error (which implements Display)
//                         error = %__err,
//                         // Use ? formatting for the Debug representation
//                         value = ?__ttj_val,
//                         // Combined message template
//                         "Failed to serialise `{}` ({}): {:?}",
//                         ::core::stringify!($value), // Corresponds to the first {}
//                         __err, // Corresponds to the second {} (using Display)
//                         __ttj_val // Corresponds to the third {:?} (using Debug)
//                     );
//                 }
//             }
//         }
//     }};
// }

// #[macro_export]
// macro_rules! tracing_to_json_pretty {
//     // Single arm handles all cases: &ident, ident, expr
//     ($value:expr, $label:literal $(,)?) => {{
//         // Optimization: Check log level *before* doing any work
//         if ::tracing::enabled!(::tracing::Level::DEBUG) {
//             // Evaluate the expression once, only if the level is enabled
//             let __ttj_val = &$value;

//             match ::serde_json::to_string_pretty(__ttj_val) {
//                 Ok(__ttj_json) => {
//                     ::tracing::debug!(
//                         "{} - {}: {}",
//                         $label,                     // Corresponds to the first {}
//                         ::core::stringify!($value), // Corresponds to the second {}
//                         __ttj_json                  // Corresponds to the third {} (using Display)
//                     );
//                 }
//                 Err(__err) => {
//                     // Log failure
//                     // Combine error and fallback into one event for efficiency
//                     ::tracing::debug!(
//                         "Failed to serialise `{}` ({}): {:?}",
//                         ::core::stringify!($value), // Corresponds to the first {}
//                         __err,                      // Corresponds to the second {} (using Display)
//                         __ttj_val                   // Corresponds to the third {:?} (using Debug)
//                     );
//                 }
//             }
//         }
//     }};
// }
