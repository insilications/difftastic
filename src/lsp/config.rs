use std::path::PathBuf;

use anyhow::Result;
use phf::phf_map;
use serde_json::Value;

pub const WORKSPACE_CONFIG_KEY: &str = "blameHighlightingSettings";

/// Your log‐level enum; still `Default` ⇒ `Info`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LspLogLevel {
    Debug,
    #[default]
    Info,
    Warning,
    Error,
}

/// A perfect‐hash map from strings → enum.
pub static LSP_LOG_LEVEL_FROM_STRING: phf::Map<&'static str, LspLogLevel> = phf_map! {
    "Debug"   => LspLogLevel::Debug,
    "Info"    => LspLogLevel::Info,
    "Warning" => LspLogLevel::Warning,
    "Error"   => LspLogLevel::Error,
};

/// # A macro that:
/// 1. Defines your struct with its fields and defaults in `new`.
/// 2. Generates an `update(&mut self, mut v: Value, errors: &mut Vec<String>)` which:
///    + walks each JSON pointer you declared.
///    + attempts `serde_json::from_value(...)` – on success, either runs your custom parser or else assigns directly.
///    + on *any* error, pushes a `(pointer, message)` into `errors`.
macro_rules! define_config {
    (
        $(#[$meta:meta])*
        $vis:vis struct $config:ident {
            $(
              // If you wrote #[parse(...)] then we capture:
              $(#[parse($pointer:literal $(, default = $default:expr)? $(, parse = $parse:path)?)])?
              $field_vis:vis $field:ident : $field_ty:ty,
            )*
        }
    ) => {
        // 1) The struct itself
        $(#[$meta])*
        $vis struct $config {
            $(
            $field_vis $field : $field_ty,
            )*
        }

        impl $config {
            pub fn new(root_path: std::path::PathBuf) -> Self {
                assert!(root_path.is_absolute());
                Self {
                    root_path,
                    $(
                        $(
                            $field : define_config!(@default $($default)?),
                        )?
                    )*
                }
            }

            /* ------------------- update ----------------------- */

            pub fn update(&mut self,
                          mut v: serde_json::Value,
                          errors: &mut Vec<String>)
            {
                $(
                    $(
                        if let Some(slot) = v.pointer_mut($pointer) {
                            let raw = slot.take();
                            define_config!(
                                @apply_parse
                                    self, raw, errors, $pointer,
                                    $field, $field_ty $(, $parse)?
                            );
                        }
                    )?
                )*
            }
        }
    };

    // Helpers

  (@default) => { Default::default() };
  (@default $expr:expr) => { $expr };

    /* … field has a custom parser … */
    (@apply_parse
        $self:ident, $raw:expr, $errs:ident, $ptr:expr,
        $field:ident, $field_ty:ty, $parse:path
    ) => {{
        match ::serde_json::from_value::<String>($raw) {        // force String
            Ok(s)  => match $parse($self, &s) {                 // borrow it
                Ok(v)  => $self.$field = v,
                Err(e) => $errs.push(format!(
                    "invalid value for `{}`: {e}",
                    $ptr.trim_start_matches('/').replace('/', "."),
                )),
            },
            Err(e) => $errs.push(format!(
                "failed to deserialize `{}`: {e}",
                $ptr.trim_start_matches('/').replace('/', "."),
            )),
        }
    }};

    /* … plain assignment, no parser … */
    (@apply_parse
        $self:ident, $raw:expr, $errs:ident, $ptr:expr,
        $field:ident, $field_ty:ty
    ) => {{
        match ::serde_json::from_value::<$field_ty>($raw) {
            Ok(v)  => $self.$field = v,
            Err(e) => $errs.push(format!(
                "failed to deserialize `{}`: {e}",
                $ptr.trim_start_matches('/').replace('/', "."),
            )),
        }
    }};
}

// {
//   "blameHighlightingSettings": {
//     "blameHighlightingOnChange": 1000,
//     "blameHighlightingParentLevel": "1",
//     "blameHighlightingShowStatus": true,
//     "blameHighlightinglogLevel": "Info"
//   }
// }

#[macro_rules_attribute::apply(define_config!)]
#[derive(Debug, Clone)]
pub struct Config {
    pub root_path: PathBuf,

    #[parse("/blameHighlightingSettings/blameHighlightingOnChange", default = 1000)]
    pub blame_highlighting_on_change: u32,

    #[parse("/blameHighlightingSettings/blameHighlightingParentLevel", default = 1)]
    pub blame_highlighting_parent_level: u32,

    // No custom parser, no default. `bool::default()` already returns false.
    #[parse("/blameHighlightingSettings/blameHighlightingShowStatus")]
    pub blame_highlighting_show_status: bool,

    #[parse("/blameHighlightingSettings/blameHighlightingLogLevel", default = LspLogLevel::Info, parse = Config::log_level_from_str)]
    pub blame_highlighting_log_level: LspLogLevel,
}

impl Config {
    /// Parse a `&str` into an `LspLogLevel`.
    ///
    /// Accepts a *borrow* only; we do not need to own the intermediate
    /// `String` that `serde_json` produced.
    fn log_level_from_str(&self, v: &str) -> Result<LspLogLevel, String> {
        LSP_LOG_LEVEL_FROM_STRING
            .get(v)
            .copied()
            .ok_or_else(|| format!("unrecognized logLevel `{v}`"))
    }
}
