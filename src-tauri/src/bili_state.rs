//! B 站功能的应用级共享状态（供 Tauri 命令层使用）
//!
//! 阶段 5 引入。持有：
//! - 登录态持久化目录（= Tauri `app_config_dir`）；
//! - 已解析/进行中的下载任务列表（供 `list_tasks` 查询、后台下载任务更新）。
//!
//! `BiliState` 本身通过内部 `Arc` 包裹以实现 `Clone`，
//! 这样后台 `tokio` 任务中的进度回调可以把状态 `move` 进去（要求 `'static`），
//! 同时满足 Tauri `State` 的 `Send + Sync`。

use crate::biliapi::task::{DownloadControl, DownloadTask};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Default)]
struct Inner {
    /// 登录态 / WBI 缓存落盘目录（Tauri 启动时注入）
    config_dir: Mutex<Option<PathBuf>>,
    /// 当前任务列表（含状态/进度/错误）
    tasks: Mutex<Vec<DownloadTask>>,
    /// 下载控制句柄（暂停 / 停止信号）。每次启动下载时重建。
    control: Mutex<DownloadControl>,
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
                *slot = u.clone();
            }
        }
    }

    /// 获取当前的下载控制句柄（用于启动下载时传递给 run_batch）。
    pub fn download_control(&self) -> DownloadControl {
        self.inner.control.lock().unwrap().clone()
    }

    /// 重置（重建）下载控制句柄。在启动一次新下载前调用，
    /// 确保上一轮 pause/stop 信号已被清空。
    pub fn reset_download_control(&self) -> DownloadControl {
        let ctrl = DownloadControl::default();
        *self.inner.control.lock().unwrap() = ctrl.clone();
        ctrl
    }

    /// 置位暂停信号：当前进行中的任务下载完当前文件后进入 `Paused`（保留部分）。
    pub fn pause_download(&self) {
        self.inner
            .control
            .lock()
            .unwrap()
            .pause
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// 置位停止信号：立即中断并删除已下载部分，状态置 `Cancelled`。
    pub fn stop_download(&self) {
        let g = self.inner.control.lock().unwrap();
        g.stop.store(true, std::sync::atomic::Ordering::SeqCst);
        g.pause.store(true, std::sync::atomic::Ordering::SeqCst);
    }
}
