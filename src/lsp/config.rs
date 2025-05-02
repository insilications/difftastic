use std::path::PathBuf;

use anyhow::Result;
use phf::phf_map;

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
#[macro_export]
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
            /// `new` merely asserts the path is absolute and fills in defaults.
            pub fn new(root_path: PathBuf) -> Self {
                assert!(root_path.is_absolute());
                Self {
                    root_path,
                    $($(
                        $field : define_config!(@default $($default)?),
                    )?)*
                }
            }

            /// Walk the JSON object, update any pointer‐declared fields, and accumulate *all* errors into `errors`.
            pub fn update(&mut self, mut v: serde_json::Value, errors: &mut Vec<String>)
            {
                $(
                $(
                    if let Some(slot) = v.pointer_mut($pointer) {
                        let raw = slot.take();
                        match serde_json::from_value(raw) {
                            Ok(parsed) => {
                                // Dispatch to either your parser or a direct‐assign arm
                                define_config!(@apply_parse
                                    self, parsed, $field, errors, $pointer $(, $parse)?);
                            }
                            Err(err) => {
                                errors.push(format!(
                                    "failed to deserialize `{}`: {}",
                                    $pointer[1..].replace('/', "."),
                                    err
                                ));
                            }
                        }
                    }
                )?
                )*
            }
        }
    };

    // Helper: pick your default or else Default::default().
    (@default) => { Default::default() };
    (@default $expr:expr) => { $expr };

    // Helper: two arms of apply_parse.
    // This has a custom parser path.
    (@apply_parse $self:ident,
                  $parsed:expr,
                  $field:ident,
                  $errors:ident,
                  $pointer:expr,
                  $parse:path)
    => {
        match $parse($self, &$parsed) {
            Ok(value) => $self.$field = value,
            Err(msg) => $errors.push(format!(
                "invalid value of `{}`: {}",
                $pointer[1..].replace('/', "."),
                msg
            )),
        }
    };

    // This without a parser, so just assign the deserialized 📦.
    (@apply_parse $self:ident,
                  $parsed:expr,
                  $field:ident,
                  $errors:ident,
                  $pointer:expr)
    => {
        $self.$field = $parsed;
    };
}

#[macro_rules_attribute::apply(define_config!)]
#[derive(Debug, Clone)]
pub struct Config {
    pub root_path: PathBuf,

    // No custom parser, no default→ bool::default() == false.
    #[parse("/blameHighlightingSettings/blameHighlightingShowStatus")]
    pub blame_highlighting_show_status: bool,

    // default = Info, custom parser from String→LspLogLevel.
    #[parse("/blameHighlightingSettings/blameHighlightingLogLevel", default = LspLogLevel::Info, parse = Config::blame_highlighting_log_level_from_string)]
    pub blame_highlighting_log_level: LspLogLevel,
}

impl Config {
    /// Returns `Result<Field, String>`, so our macro can match on Err(msg) and push it into `errors`.
    fn blame_highlighting_log_level_from_string(&self, v: &str) -> Result<LspLogLevel, String> {
        LSP_LOG_LEVEL_FROM_STRING
            .get(v)
            .copied()
            .ok_or_else(|| format!("unrecognized logLevel {}", v))
    }
}
