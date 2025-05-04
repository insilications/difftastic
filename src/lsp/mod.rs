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

#[macro_export]
macro_rules! tracing_to_json {
    // ────────────────────────────────────────────────────────────────
    // 1. Simplest form ─ just the value. We auto-derive a label from the type name so the log line is still
    //    identifiable. Example: tracing_to_json!(&foo);
    // ────────────────────────────────────────────────────────────────
    ($value:expr $(,)?) => {{
        let __val = &$value;
        let __type = ::core::any::type_name::<_>(); // e.g. "my_crate::Foo"

        match ::serde_json::to_string_pretty(__val) {
            Ok(__json) => {
                ::tracing::debug!("{}: {}", __type, __json);
            }
            Err(__err) => {
                ::tracing::debug!("Failed to serialise {}: {}", __type, __err);
                ::tracing::debug!("{}: {:?}", __type, __val);
            }
        }
    }};

    // ────────────────────────────────────────────────────────────────
    // 2. Value + *one* string literal label. Example: tracing_to_json!(&foo, "foo"); -> "foo: { ... }"  or  "foo: Foo {
    //    .. }"
    // ────────────────────────────────────────────────────────────────
    ($value:expr, $label:literal $(,)?) => {{
        let __val = &$value;

        match ::serde_json::to_string_pretty(__val) {
            Ok(__json) => {
                ::tracing::debug!(concat!($label, ": {}"), __json);
            }
            Err(__err) => {
                ::tracing::debug!(concat!("Failed to serialise ", $label, ": {}"), __err);
                ::tracing::debug!(concat!($label, ": {:?}"), __val);
            }
        }
    }};

    // ────────────────────────────────────────────────────────────────
    // 3. Full control - value + success format + error format (your original signature, now the “catch-all” arm).
    // ────────────────────────────────────────────────────────────────
    ($value:expr, $ok_fmt:expr, $err_fmt:expr $(,)?) => {{
        let __val = &$value;

        match ::serde_json::to_string_pretty(__val) {
            Ok(__json) => {
                ::tracing::debug!($ok_fmt, __json);
            }
            Err(__err) => {
                ::tracing::debug!("Failed to serialize value to JSON: {}", __err);
                ::tracing::debug!($err_fmt, __val);
            }
        }
    }};
}
