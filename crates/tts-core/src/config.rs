//! Settings that outlive one command: which engine, which voice, where the weights are.
//!
//! There are three ways to say the same thing and they have to agree on precedence, so it
//! is written down once here: **flag > environment > `dream-tts.json` in the install root > user
//! config > built-in default**. Nothing in this module knows an engine id — resolving
//! `references/<id>/weights` is the registry's job, and this only supplies the directory
//! those conventions hang off.
//!
//! Every name here is prefixed `dream-tts` / `DREAM_TTS_`. A bare `tts.json` or `TTS_ROOT`
//! is a name this app has no claim to on a machine that runs anything else.
//!
//! Two roots, deliberately distinct:
//!
//! * **root** is the installation — binaries, `voices/`, `scripts/`. Small, replaced
//!   wholesale by an upgrade.
//! * **data_dir** is the downloads — checkpoints and fixtures, 4 to 13 GB. Survives an
//!   upgrade, and is the thing someone wants to put on an external disk.
//!
//! Both default to the current directory, which is exactly what every path in this repo
//! meant before this module existed. Relocation is opt-in and nothing changes for someone
//! who never writes a config file.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// A settings file. Every field optional: a config that sets one thing should not have to
/// restate the rest.
///
/// Unknown keys are an error rather than a shrug. A misspelled key that silently does
/// nothing is the worst outcome available — the user believes they configured something.
#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Settings {
    /// Engine id. Omitted means the registry's default.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine: Option<String>,
    /// Voice asset directory. Relative paths resolve against the install root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<PathBuf>,
    /// Weight format, engine-specific. `tts engines` lists what each accepts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quant: Option<String>,
    /// Where checkpoints and fixtures live. Relative paths resolve against the install
    /// root, so `"data_dir": "."` is the default and `"/Volumes/ssd/tts"` is not.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_dir: Option<PathBuf>,
    /// Segment length budget in characters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_chars: Option<usize>,
    /// Hold the GPU lock during synthesis. Default true; see [`crate::lock`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gpu_lock: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gaps: Option<GapSettings>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub serve: Option<ServeSettings>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GapSettings {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segment_ms: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub paragraph_ms: Option<usize>,
}

#[derive(Clone, Debug, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ServeSettings {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub host: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub port: Option<u16>,
    /// Per-request character ceiling.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_chars: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub segment_chars: Option<usize>,
}

/// Where a value came from. Printed by `tts config`, which is the only thing that makes a
/// precedence chain debuggable instead of mysterious.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Origin {
    Default,
    UserConfig,
    ProjectConfig,
    Environment,
}

impl Origin {
    pub fn label(self) -> &'static str {
        match self {
            Origin::Default => "default",
            Origin::UserConfig => "user config",
            Origin::ProjectConfig => PROJECT_CONFIG,
            Origin::Environment => "environment",
        }
    }
}

/// The resolved settings, plus enough provenance to explain them.
#[derive(Clone, Debug)]
pub struct Config {
    /// The installation directory. `DREAM_TTS_ROOT`, else the current directory.
    pub root: PathBuf,
    /// Absolute path to the checkpoints and fixtures.
    pub data_dir: PathBuf,
    pub data_dir_origin: Origin,
    /// The file these settings were read from, if any.
    pub source: Option<PathBuf>,
    pub settings: Settings,
}

/// Filename looked for in the install root. Namespaced, like every other name this app
/// puts on a user's machine: `tts` alone belongs to whoever installed it first.
pub const PROJECT_CONFIG: &str = "dream-tts.json";

fn env_path(key: &str) -> Option<PathBuf> {
    match std::env::var_os(key) {
        Some(v) if !v.is_empty() => Some(PathBuf::from(v)),
        _ => None,
    }
}

/// `~/.config/dream-tts/config.json`, for settings that should follow the user across
/// installs. XDG's variable is honoured because someone who sets it means it.
pub fn user_config_path() -> Option<PathBuf> {
    if let Some(base) = env_path("XDG_CONFIG_HOME") {
        return Some(base.join("dream-tts/config.json"));
    }
    env_path("HOME").map(|h| h.join(".config/dream-tts/config.json"))
}

/// Drop keys beginning with `//`, recursively.
///
/// JSON has no comments and a settings file people edit by hand needs them, so `"//": "…"`
/// is the convention — the one `dream-tts.example.json` uses to explain itself. Stripping
/// them here rather than relaxing `deny_unknown_fields` keeps the typo protection intact:
/// `"engien"` is still an error, and no plausible typo of a real key starts with `//`.
fn strip_comments(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(map) => {
            map.retain(|k, _| !k.starts_with("//"));
            for v in map.values_mut() {
                strip_comments(v);
            }
        }
        serde_json::Value::Array(items) => items.iter_mut().for_each(strip_comments),
        _ => {}
    }
}

fn read_settings(path: &Path) -> Result<Settings> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("reading config {}", path.display()))?;
    let mut value: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("{} is not valid JSON", path.display()))?;
    strip_comments(&mut value);
    // serde_json's own message carries the offending key, which is what makes a typo in a
    // hand-written config findable.
    serde_json::from_value(value).with_context(|| {
        format!(
            "parsing config {} — every key is optional, but unknown keys are rejected so a \
             typo cannot silently do nothing (keys starting with `//` are comments)",
            path.display()
        )
    })
}

impl Config {
    /// Load the config chain. `explicit` is a `--config` flag: given, it is the only file
    /// consulted, and a missing one is an error rather than a silent fall-through.
    pub fn load(explicit: Option<&Path>) -> Result<Self> {
        let root = env_path("DREAM_TTS_ROOT")
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("."));

        let (settings, source) = match explicit.or(env_path("DREAM_TTS_CONFIG").as_deref()) {
            Some(p) => {
                anyhow::ensure!(p.exists(), "no config file at {}", p.display());
                (read_settings(p)?, Some(p.to_path_buf()))
            }
            None => {
                let project = root.join(PROJECT_CONFIG);
                if project.is_file() {
                    (read_settings(&project)?, Some(project))
                } else {
                    match user_config_path().filter(|p| p.is_file()) {
                        Some(p) => (read_settings(&p)?, Some(p)),
                        None => (Settings::default(), None),
                    }
                }
            }
        };

        let origin = match &source {
            Some(p) if p.ends_with(PROJECT_CONFIG) => Origin::ProjectConfig,
            Some(_) => Origin::UserConfig,
            None => Origin::Default,
        };

        let (data_dir, data_dir_origin) = match env_path("DREAM_TTS_DATA_DIR") {
            Some(p) => (p, Origin::Environment),
            None => match settings.data_dir.clone() {
                Some(p) => (p, origin),
                None => (PathBuf::from("."), Origin::Default),
            },
        };

        Ok(Self {
            data_dir: absolutise(&root, &data_dir),
            data_dir_origin,
            root,
            source,
            settings,
        })
    }

    /// A conventional path under the data directory — where `references/<id>/weights` and
    /// `fixtures/<id>` live. An absolute input is returned untouched.
    pub fn data_path(&self, relative: impl AsRef<Path>) -> PathBuf {
        absolutise(&self.data_dir, relative.as_ref())
    }

    /// A path that belongs to the installation rather than the downloads: `voices/…`.
    pub fn root_path(&self, relative: impl AsRef<Path>) -> PathBuf {
        absolutise(&self.root, relative.as_ref())
    }

    /// Resolve a path the *user* supplied, leniently.
    ///
    /// As typed if that exists, otherwise against the install root. The working directory
    /// wins, so nothing shipped can shadow a local file — but `--voice voices/…`, the form
    /// every doc and every script uses, keeps working when `dream-tts` is invoked from
    /// somewhere else through a symlink on PATH. Without this, the documented commands are
    /// correct only from inside the install directory.
    pub fn locate(&self, p: &Path) -> Option<PathBuf> {
        if p.exists() {
            return Some(p.to_path_buf());
        }
        let rooted = self.root_path(p);
        rooted.exists().then_some(rooted)
    }

    /// [`Self::locate`], with an error that names both places that were tried. A "no such
    /// file" naming only what the user typed sends them looking in the wrong directory.
    pub fn locate_or_err(&self, p: &Path, what: &str) -> Result<PathBuf> {
        self.locate(p).with_context(|| {
            format!(
                "no {what} at {} or {}",
                p.display(),
                self.root_path(p).display()
            )
        })
    }

    pub fn gpu_lock(&self) -> bool {
        self.settings.gpu_lock.unwrap_or(true)
    }

    /// Where the advisory GPU lock lives. In the data directory, not `/tmp`, so two
    /// installs pointed at one set of weights contend and two independent ones do not.
    pub fn lock_path(&self) -> PathBuf {
        self.data_path(".dream-tts-gpu.lock")
    }
}

/// Join unless already absolute. `Path::join` does this too, but silently — being explicit
/// about it is what keeps an absolute `data_dir` from being quietly reinterpreted.
fn absolutise(base: &Path, p: &Path) -> PathBuf {
    if p.is_absolute() {
        p.to_path_buf()
    } else if p == Path::new(".") {
        // `base.join(".")` is correct and prints as `/install/.`, which reads like a bug
        // in every path this crate reports.
        base.to_path_buf()
    } else {
        base.join(p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_config_changes_nothing() {
        let c = Config {
            root: PathBuf::from("/install"),
            data_dir: PathBuf::from("/install"),
            data_dir_origin: Origin::Default,
            source: None,
            settings: Settings::default(),
        };
        // The pre-config behaviour: conventions hang off the install directory.
        assert_eq!(
            c.data_path("references/qwen3tts/weights"),
            PathBuf::from("/install/references/qwen3tts/weights")
        );
        assert_eq!(
            c.root_path("voices/cosy-default-qwen3tts"),
            PathBuf::from("/install/voices/cosy-default-qwen3tts")
        );
        assert!(c.gpu_lock());
    }

    #[test]
    fn a_dot_data_dir_prints_as_the_root() {
        assert_eq!(
            absolutise(Path::new("/install"), Path::new(".")),
            PathBuf::from("/install")
        );
    }

    #[test]
    fn absolute_data_dir_is_not_rejoined() {
        assert_eq!(
            absolutise(Path::new("/install"), Path::new("/Volumes/ssd/tts")),
            PathBuf::from("/Volumes/ssd/tts")
        );
    }

    #[test]
    fn locate_prefers_the_working_directory_then_the_root() {
        let dir = std::env::temp_dir().join(format!("dream-tts-locate-{}", std::process::id()));
        let root = dir.join("install");
        std::fs::create_dir_all(root.join("voices/shipped")).unwrap();
        std::fs::create_dir_all(dir.join("voices/local")).unwrap();
        let c = Config {
            root: root.clone(),
            data_dir: root.clone(),
            data_dir_origin: Origin::Default,
            source: None,
            settings: Settings::default(),
        };
        // Found under the install root even though the cwd knows nothing about it.
        assert_eq!(
            c.locate(Path::new("voices/shipped")),
            Some(root.join("voices/shipped"))
        );
        // Absent in both: `None`, and the error names both.
        assert!(c.locate(Path::new("voices/nope")).is_none());
        let err = c
            .locate_or_err(Path::new("voices/nope"), "voice asset")
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("voices/nope") && err.contains("install"),
            "{err}"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn unknown_keys_are_rejected() {
        let err = serde_json::from_str::<Settings>(r#"{"engien":"qwen3tts"}"#)
            .expect_err("a typo must not be accepted");
        assert!(err.to_string().contains("engien"), "{err}");
    }

    /// The example config ships with `//` notes in it, so this is the property that keeps
    /// it from being an example that does not work.
    #[test]
    fn comment_keys_are_stripped_but_typos_are_not() {
        let mut v: serde_json::Value = serde_json::from_str(
            r#"{"//": "a note", "//data_dir": "why", "quant": "f16",
                "serve": {"//": "nested too", "port": 3003}}"#,
        )
        .unwrap();
        strip_comments(&mut v);
        let s: Settings = serde_json::from_value(v).expect("comments must not be fields");
        assert_eq!(s.quant.as_deref(), Some("f16"));
        assert_eq!(s.serve.and_then(|x| x.port), Some(3003));

        let mut bad: serde_json::Value = serde_json::from_str(r#"{"engien": "x"}"#).unwrap();
        strip_comments(&mut bad);
        assert!(serde_json::from_value::<Settings>(bad).is_err());
    }

    #[test]
    fn a_partial_config_leaves_the_rest_unset() {
        let s: Settings = serde_json::from_str(r#"{"quant":"f16"}"#).unwrap();
        assert_eq!(s.quant.as_deref(), Some("f16"));
        assert!(s.engine.is_none() && s.voice.is_none() && s.data_dir.is_none());
    }

    /// Round-tripping matters because `tts config --write` emits this.
    #[test]
    fn serialising_omits_unset_fields() {
        let s = Settings {
            engine: Some("audio8".into()),
            ..Default::default()
        };
        assert_eq!(serde_json::to_string(&s).unwrap(), r#"{"engine":"audio8"}"#);
    }
}
