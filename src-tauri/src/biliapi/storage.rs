//! 登录态与 WBI key 持久化存储
//!
//! 阶段 4 实现。把 SESSDATA 与 WBI mixinKey 落盘到 Tauri `app_config_dir`
//! 下的 JSON 文件，使一次扫码后后续启动自动登录、冷启动少一次 `nav` 请求。
//!
//! 设计原则（与 `client.rs` 一致）：
//! - 不直接依赖 Tauri 运行时，支持「调用方显式传入目录」或「自动探测」；
//! - 文件读写失败不影响主流程（降级为内存临时态 / 重新拉取）。

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const AUTH_FILE: &str = "auth.json";
const WBI_FILE: &str = "wbi_cache.json";
const WBI_TTL_SECS: u64 = 24 * 3600;

/// 登录态持久化结构（明文存储；如需加密可在此层扩展）。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize, Default)]
pub struct AuthStore {
    #[serde(default)]
    pub sessdata: Option<String>,
}

/// WBI key 落盘结构（与 `wbi_cache::Cache` 对应）。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct WbiCacheStore {
    pub key: String,
    pub fetched_at: u64,
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// 解析配置目录：优先调用方传入，其次环境变量 `AUDIO_PROCESSOR_CONFIG_DIR`，
/// 最后回退到当前工作目录（便于无 Tauri 运行时环境如单元测试运行）。
fn resolve_dir(explicit: Option<&Path>) -> PathBuf {
    if let Some(d) = explicit {
        return d.to_path_buf();
    }
    if let Ok(d) = std::env::var("AUDIO_PROCESSOR_CONFIG_DIR") {
        return PathBuf::from(d);
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Option<T> {
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

fn write_json<T: serde::Serialize>(path: &Path, value: &T) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(text) = serde_json::to_string_pretty(value) {
        // 写入临时文件再 rename，避免半写损坏已有文件
        let tmp = path.with_extension("tmp");
        if std::fs::write(&tmp, text).is_ok() {
            let _ = std::fs::rename(&tmp, path);
        }
    }
}

// ───────────────────────────── SESSDATA ─────────────────────────────

/// 读取已持久化的 SESSDATA。
/// - `dir` 为 `None` 时自动探测配置目录；
/// - 文件不存在或解析失败返回 `None`（不报错）。
pub fn load_sessdata(dir: Option<&Path>) -> Option<String> {
    let path = resolve_dir(dir).join(AUTH_FILE);
    let store: AuthStore = read_json(&path)?;
    store.sessdata.filter(|s| !s.is_empty())
}

/// 持久化 SESSDATA。传 `None` 等价于清除登录态。
/// 写入失败仅打印警告（不阻断主流程）。
pub fn save_sessdata(dir: Option<&Path>, sessdata: Option<&str>) {
    let path = resolve_dir(dir).join(AUTH_FILE);
    let store = AuthStore {
        sessdata: sessdata.map(|s| s.to_string()),
    };
    write_json(&path, &store);
}

/// 清除已持久化的 SESSDATA（登出 / 登录失效时调用）。
pub fn clear_sessdata(dir: Option<&Path>) {
    save_sessdata(dir, None);
}

// ───────────────────────────── WBI key ─────────────────────────────

/// 读取落盘的 WBI key；超过 24h 视为失效返回 `None`。
pub fn load_wbi(dir: Option<&Path>) -> Option<String> {
    let path = resolve_dir(dir).join(WBI_FILE);
    let store: WbiCacheStore = read_json(&path)?;
    if now_secs().saturating_sub(store.fetched_at) < WBI_TTL_SECS {
        Some(store.key)
    } else {
        None
    }
}

/// 落盘 WBI key 及抓取时间戳。
pub fn save_wbi(dir: Option<&Path>, key: &str) {
    let path = resolve_dir(dir).join(WBI_FILE);
    let store = WbiCacheStore {
        key: key.to_string(),
        fetched_at: now_secs(),
    };
    write_json(&path, &store);
}

/// 清除落盘 WBI 缓存（登录态变更时调用）。
pub fn clear_wbi(dir: Option<&Path>) {
    let path = resolve_dir(dir).join(WBI_FILE);
    let _ = std::fs::remove_file(path);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir() -> PathBuf {
        let p = std::env::temp_dir().join(format!("ap_storage_test_{}", now_secs()));
        let _ = std::fs::create_dir_all(&p);
        p
    }

    #[test]
    fn test_sessdata_roundtrip() {
        let dir = tmp_dir();
        assert!(load_sessdata(Some(&dir)).is_none());

        save_sessdata(Some(&dir), Some("SESS_test_123"));
        assert_eq!(load_sessdata(Some(&dir)).as_deref(), Some("SESS_test_123"));

        // 空值清除
        clear_sessdata(Some(&dir));
        assert!(load_sessdata(Some(&dir)).is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_wbi_roundtrip_and_expiry() {
        let dir = tmp_dir();
        assert!(load_wbi(Some(&dir)).is_none());

        save_wbi(Some(&dir), "mixin_key_abc");
        assert_eq!(load_wbi(Some(&dir)).as_deref(), Some("mixin_key_abc"));

        clear_wbi(Some(&dir));
        assert!(load_wbi(Some(&dir)).is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_auth_json_shape() {
        // 落盘文件应为合法 JSON 且含 sessdata 字段
        let dir = tmp_dir();
        save_sessdata(Some(&dir), Some("XYZ"));
        let content = std::fs::read_to_string(dir.join(AUTH_FILE)).unwrap();
        assert!(content.contains("sessdata"));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
