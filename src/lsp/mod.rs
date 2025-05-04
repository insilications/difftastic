mod cache;
mod capabilities;
mod config;
pub mod custom;
mod lsp_ext;
mod meter;
mod server;
mod uri_ext;

// #[macro_export]
// macro_rules! tracing_to_json {
//     // Allow an optional trailing comma so the macro looks natural.
//     ($value:expr, $ok_fmt:expr, $err_fmt:expr $(,)?) => {{
//         // Take a reference *once* so we never evaluate `$value` twice.
//         let __val = &$value;

//         // Fully-qualified paths keep the macro usable without explicit `use`s.
//         match ::serde_json::to_string_pretty(__val) {
//             Ok(__json) => {
//                 ::tracing::info!($ok_fmt, __json);
//             }
//             Err(__err) => {
//                 // 1. Log the fallback.
//                 ::tracing::debug!("Failed to serialize value to JSON: {}", __err);
//                 ::tracing::debug!($err_fmt, __val);
//             }
//         }
//     }};
// }

// #[macro_export]
// macro_rules! tracing_to_json {
//     // ------------------------------------------------------------------
//     // 1. `&ident`  ⇒  use `ident` as the field key
//     // ------------------------------------------------------------------
//     ( & $val:ident , $ok_msg:expr, $err_msg:expr $(,)? ) => {{
//         let __ttj_val = &$val;                         // evaluate once
//         match ::serde_json::to_string(__ttj_val) {
//             Ok(__ttj_json) => {
//                 ::tracing::info!($val = %__ttj_json, $ok_msg);
//             }
//             Err(_) => {
//                 ::tracing::debug!($val = ?__ttj_val,  $err_msg);
//             }
//         }
//     }};

//     // ------------------------------------------------------------------
//     // 2. bare `ident`  (no leading `&`)
//     // ------------------------------------------------------------------
//     ( $val:ident , $ok_msg:expr, $err_msg:expr $(,)? ) => {{
//         let __ttj_val = &$val;
//         match ::serde_json::to_string(__ttj_val) {
//             Ok(__ttj_json) => {
//                 ::tracing::info!($val = %__ttj_json, $ok_msg);
//             }
//             Err(_) => {
//                 ::tracing::debug!($val = ?__ttj_val,  $err_msg);
//             }
//         }
//     }};

//     // ------------------------------------------------------------------
//     // 3. Fallback for any other expression  ⇒  fixed field name `value`
//     // ------------------------------------------------------------------
//     ( $val:expr , $ok_msg:expr, $err_msg:expr $(,)? ) => {{
//         let __ttj_val = &$val;
//         match ::serde_json::to_string(__ttj_val) {
//             Ok(__ttj_json) => {
//                 ::tracing::info!(value = %__ttj_json, $ok_msg);
//             }
//             Err(_) => {
//                 ::tracing::debug!(value = ?__ttj_val, $err_msg);
//             }
//         }
//     }};
// }

#[macro_export]
macro_rules! tracing_to_json {
    // ────────────────────────────────────────────────────────────────
    // Example: tracing_to_json!(&server_caps, "Server Capabilities"); -> "Server Capabilities - &  server_caps: {...}"
    // ────────────────────────────────────────────────────────────────
    (& $value:ident, $label:literal $(,)?) => {{
        let __ttj_val = &$value; // Evaluate once

        match ::serde_json::to_string_pretty(__ttj_val) {
            Ok(__ttj_json) => {
                ::tracing::debug!(concat!($label, " - &", stringify!($value), ": {}"), __ttj_json);
            }
            Err(__err) => {
                ::tracing::debug!(concat!("Failed to serialise `&", stringify!($value), "`: {}"), __err);
                ::tracing::debug!(concat!($label, " - &", stringify!($value), ": {:?}"), __ttj_val);
            }
        }
    }};

    // ────────────────────────────────────────────────────────────────
    // Example: tracing_to_json!(server_caps, "Server Capabilities"); -> "Server Capabilities - server_caps: {...}"
    // ────────────────────────────────────────────────────────────────
    ($value:ident, $label:literal $(,)?) => {{
        let __ttj_val = &$value; // Evaluate once

        match ::serde_json::to_string_pretty(__ttj_val) {
            Ok(__ttj_json) => {
                ::tracing::debug!(concat!($label, " - ", stringify!($value), ": {}"), __ttj_json);
            }
            Err(__err) => {
                ::tracing::debug!(concat!("Failed to serialise `", stringify!($value), "`: {}"), __err);
                ::tracing::debug!(concat!($label, " - ", stringify!($value), ": {:?}"), __ttj_val);
            }
        }
    }};

    // ────────────────────────────────────────────────────────────────
    // Example: tracing_to_json!(&self.config.server_caps(), "Server Capabilities");
    // -> "Server Capabilities - &self.config.server_caps(): {...}"
    // ────────────────────────────────────────────────────────────────
    ($value:expr, $label:literal $(,)?) => {{
        let __ttj_val = &$value; // Evaluate once

        match ::serde_json::to_string_pretty(__ttj_val) {
            Ok(__ttj_json) => {
                ::tracing::debug!(concat!($label, " - ", stringify!($value), ": {}"), __ttj_json);
            }
            Err(__err) => {
                ::tracing::debug!(concat!("Failed to serialise `", stringify!($value), "`: {}"), __err);
                ::tracing::debug!(concat!($label, " - ", stringify!($value), ": {:?}"), __ttj_val);
            }
        }
    }};
}
