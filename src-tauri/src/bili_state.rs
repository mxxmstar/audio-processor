//! B 站功能的应用级共享状态（供 Tauri 命令层使用）
//!
//! 阶段 5 引入。持有：
//! - 登录态持久化目录（= Tauri `app_config_dir`）；
//! - 已解析/进行中的下载任务列表（供 `list_tasks` 查询、后台下载任务更新）。
//!
//! `BiliState` 本身通过内部 `Arc` 包裹以实现 `Clone`，
//! 这样后台 `tokio` 任务中的进度回调可以把状态 `move` 进去（要求 `'static`），
//! 同时满足 Tauri `State` 的 `Send + Sync`。

use crate::biliapi::task::DownloadTask;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Inner {
    /// 登录态 / WBI 缓存落盘目录（Tauri 启动时注入）
    config_dir: Mutex<Option<PathBuf>>,
    /// 当前任务列表（含状态/进度/错误）
    tasks: Mutex<Vec<DownloadTask>>,
}

#[derive(Default, Clone)]
pub struct BiliState {
    inner: Arc<Inner>,
}

impl BiliState {
    pub fn new() -> Self {
        Self::default()
    }

    /// 注入配置目录并同步给 WBI 缓存层
    pub fn init_config_dir(&self, dir: PathBuf) {
        *self.inner.config_dir.lock().unwrap() = Some(dir.clone());
        crate::biliapi::wbi_cache::set_cache_dir(Some(&dir));
    }

    pub fn config_dir_opt(&self) -> Option<PathBuf> {
        self.inner.config_dir.lock().unwrap().clone()
    }

    /// 用一批解析得到的任务替换当前任务列表
    pub fn set_tasks(&self, tasks: Vec<DownloadTask>) {
        *self.inner.tasks.lock().unwrap() = tasks;
    }

    /// 读取任务列表快照（前端展示用）
    pub fn snapshot_tasks(&self) -> Vec<DownloadTask> {
        self.inner.tasks.lock().unwrap().clone()
    }

    /// 用后台下载结果回写任务列表（按 id 匹配）
    pub fn apply_results(&self, updated: &[DownloadTask]) {
        let mut guard = self.inner.tasks.lock().unwrap();
        for u in updated {
            if let Some(slot) = guard.iter_mut().find(|t| t.id == u.id) {
                slot.status = u.status;
                slot.error = u.error.clone();
            }
        }
    }
}
