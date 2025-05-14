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

#[macro_export]
macro_rules! tracing_to_json {
    // Single arm handles all cases: &ident, ident, expr
    ($value:expr, $label:literal $(,)?) => {{
        // Optimization: Check log level *before* doing any work
        if ::tracing::enabled!(::tracing::Level::DEBUG) {
            // Evaluate the expression once, only if the level is enabled
            let __ttj_val = &$value;

            match ::serde_json::to_string(__ttj_val) {
                Ok(__ttj_json) => {
                    // Optimization: Use structured logging fields
                    ::tracing::debug!(
                        // --- Structured Fields (Sigils % and ? work here) ---
                        label = $label,
                        // Use stringify! directly, handles &ident correctly
                        expr = ::core::stringify!($value),
                        // Use % formatting for the JSON string (which implements Display)
                        json = %__ttj_json,
                        // Message template using fields
                        "{} - {}: {}",
                       $label, // Corresponds to the first {}
                        ::core::stringify!($value), // Corresponds to the second {}
                        __ttj_json // Corresponds to the third {} (using Display)
                    );
                }
                Err(__err) => {
                    // Log failure using structured fields
                    // Combine error and fallback into one event for efficiency
                    ::tracing::debug!(
                        // --- Structured Fields (Sigils % and ? work here) ---
                        label = $label,
                        expr = ::core::stringify!($value),
                        // Use % formatting for the error (which implements Display)
                        error = %__err,
                        // Use ? formatting for the Debug representation
                        value = ?__ttj_val,
                        // Combined message template
                        "Failed to serialise `{}` ({}): {:?}",
                        ::core::stringify!($value), // Corresponds to the first {}
                        __err, // Corresponds to the second {} (using Display)
                        __ttj_val // Corresponds to the third {:?} (using Debug)
                    );
                }
            }
        }
    }};
}

#[macro_export]
macro_rules! tracing_to_json_pretty {
    // Single arm handles all cases: &ident, ident, expr
    ($value:expr, $label:literal $(,)?) => {{
        // Optimization: Check log level *before* doing any work
        if ::tracing::enabled!(::tracing::Level::DEBUG) {
            // Evaluate the expression once, only if the level is enabled
            let __ttj_val = &$value;

            match ::serde_json::to_string_pretty(__ttj_val) {
                Ok(__ttj_json) => {
                    ::tracing::debug!(
                        // // --- Structured Fields (Sigils % and ? work here) ---
                        // label = $label,
                        // // Use stringify! directly, handles &ident correctly
                        // expr = ::core::stringify!($value),
                        // // Use % formatting for the JSON string (which implements Display)
                        // json = %__ttj_json,
                        // Message template using fields
                        "{} - {}: {}",
                        $label,                     // Corresponds to the first {}
                        ::core::stringify!($value), // Corresponds to the second {}
                        __ttj_json                  // Corresponds to the third {} (using Display)
                    );
                }
                Err(__err) => {
                    // Log failure
                    // Combine error and fallback into one event for efficiency
                    ::tracing::debug!(
                        // // --- Structured Fields (Sigils % and ? work here) ---
                        // label = $label,
                        // expr = ::core::stringify!($value),
                        // // Use % formatting for the error (which implements Display)
                        // error = %__err,
                        // // Use ? formatting for the Debug representation
                        // value = ?__ttj_val,
                        // Combined message template
                        "Failed to serialise `{}` ({}): {:?}",
                        ::core::stringify!($value), // Corresponds to the first {}
                        __err,                      // Corresponds to the second {} (using Display)
                        __ttj_val                   // Corresponds to the third {:?} (using Debug)
                    );
                }
            }
        }
    }};
}
