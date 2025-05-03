use std::{convert::TryFrom, path::PathBuf};

use phf::phf_map;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LspLogLevel {
    Debug,
    #[default]
    Info,
    Warning,
    Error,
}

pub static LSP_LOG_LEVEL_FROM_STRING: phf::Map<&'static str, LspLogLevel> = phf_map! {
    "Debug"   => LspLogLevel::Debug,
    "Info"    => LspLogLevel::Info,
    "Warning" => LspLogLevel::Warning,
    "Error"   => LspLogLevel::Error,
};

#[derive(Debug, Clone)]
pub struct Config {
    pub root_path: PathBuf,
    pub blame_highlighting_on_change: u32,
    pub blame_highlighting_parent_level: u32,
    pub blame_highlighting_show_status: bool,
    pub blame_highlighting_log_level: LspLogLevel,
}

// ----------------------------------------------------------------
// recursive helper macro: handles one field per arm, then recurses.
// ----------------------------------------------------------------
macro_rules! parse_config_obj {
    // u32‐case, with trailing comma + more fields
    ($self:ident, $obj:ident, $errs:ident, $disp:expr,
     $field:ident : u32 => $key:expr , $($rest:tt)* ) => {
        if let Some(raw) = $obj.get($key) {
            match raw.as_u64().and_then(|n| u32::try_from(n).ok()) {
                Some(v) => $self.$field = v,
                None    => $errs.push(
                    format!("invalid integer for `{}.{} `", $disp, $key)
                ),
            }
        }
        parse_config_obj!($self, $obj, $errs, $disp, $($rest)*);
    };

    // bool‐case, with trailing comma + more fields
    ($self:ident, $obj:ident, $errs:ident, $disp:expr,
     $field:ident : bool => $key:expr , $($rest:tt)* ) => {
        if let Some(raw) = $obj.get($key) {
            match raw.as_bool() {
                Some(v) => $self.$field = v,
                None    => $errs.push(
                    format!("invalid boolean for `{}.{} `", $disp, $key)
                ),
            }
        }
        parse_config_obj!($self, $obj, $errs, $disp, $($rest)*);
    };

    // enum‐case, with trailing comma + more fields
    ($self:ident, $obj:ident, $errs:ident, $disp:expr,
     $field:ident : enum($map:expr, $msg:expr) => $key:expr , $($rest:tt)* ) => {
        if let Some(raw) = $obj.get($key) {
            if let Some(s) = raw.as_str() {
                match $map.get(s).copied() {
                    Some(v) => $self.$field = v,
                    None    => $errs.push(
                        format!("{} `{}` for `{}.{} `", $msg, s, $disp, $key)
                    ),
                }
            } else {
                $errs.push(
                    format!("invalid string for `{}.{} `", $disp, $key)
                );
            }
        }
        parse_config_obj!($self, $obj, $errs, $disp, $($rest)*);
    };

    // last u32‐case (no trailing comma)
    ($self:ident, $obj:ident, $errs:ident, $disp:expr,
     $field:ident : u32 => $key:expr ) => {
        if let Some(raw) = $obj.get($key) {
            match raw.as_u64().and_then(|n| u32::try_from(n).ok()) {
                Some(v) => $self.$field = v,
                None    => $errs.push(
                    format!("invalid integer for `{}.{} `", $disp, $key)
                ),
            }
        }
    };

    // last bool
    ($self:ident, $obj:ident, $errs:ident, $disp:expr,
     $field:ident : bool => $key:expr ) => {
        if let Some(raw) = $obj.get($key) {
            match raw.as_bool() {
                Some(v) => $self.$field = v,
                None    => $errs.push(
                    format!("invalid boolean for `{}.{} `", $disp, $key)
                ),
            }
        }
    };

    // last enum
    ($self:ident, $obj:ident, $errs:ident, $disp:expr,
     $field:ident : enum($map:expr, $msg:expr) => $key:expr ) => {
        if let Some(raw) = $obj.get($key) {
            if let Some(s) = raw.as_str() {
                match $map.get(s).copied() {
                    Some(v) => $self.$field = v,
                    None    => $errs.push(
                        format!("{} `{}` for `{}.{} `", $msg, s, $disp, $key)
                    ),
                }
            } else {
                $errs.push(
                    format!("invalid string for `{}.{} `", $disp, $key)
                );
            }
        }
    };

    // done
    ($self:ident, $obj:ident, $errs:ident, $disp:expr,) => {};
}

impl Config {
    pub fn new(root_path: PathBuf) -> Self {
        assert!(root_path.is_absolute());
        Self {
            root_path,
            blame_highlighting_on_change: 1000,
            blame_highlighting_parent_level: 1,
            blame_highlighting_show_status: Default::default(),
            blame_highlighting_log_level: LspLogLevel::default(),
        }
    }

    pub fn update(&mut self, settings: &serde_json::Value, errors: &mut Vec<String>) {
        const JSON_PREFIX: &str = "/blameHighlightingSettings";
        const DISP_PREFIX: &str = "blameHighlightingSettings";

        // 1) Only one JSON‐pointer walk
        let obj: &serde_json::Map<String, serde_json::Value> =
            match settings.pointer(JSON_PREFIX).and_then(serde_json::Value::as_object) {
                Some(o) => o,
                None => return,
            };

        // 2) Munch through each field.  Add a new line here to add a new field.
        parse_config_obj!(self, obj, errors, DISP_PREFIX,
            blame_highlighting_on_change    : u32  => "blameHighlightingOnChange",
            blame_highlighting_parent_level : u32  => "blameHighlightingParentLevel",
            blame_highlighting_show_status  : bool => "blameHighlightingShowStatus",
            blame_highlighting_log_level    :
                enum(LSP_LOG_LEVEL_FROM_STRING, "unrecognized logLevel")
                                             => "blameHighlightingLogLevel",
        );
    }
}
