//! biliapi：B 站 API 封装模块
//!
//! 把 `bilidownload/server/bilibili/` 中调用 B 站的接口封装为异步 Rust API。
//! 底层复用 `http_client` 模块（WBI 签名、直连、Mozilla UA）。
//!
//! 子模块：
//! - `client`  - BiliClient 上下文与统一请求构造
//! - `types`   - B 站响应/业务数据结构（对齐 Go 版 type.go）
//! - `video`   - 视频/番剧/收藏夹/合集/热门 API
//! - `stream`  - 音视频直链挑选（按 Codecid/清晰度优先级）
//! - `task`    - 下载任务模型与编排（AudioOnly/VideoOnly/Merge + 并发）
//! - `login`   - 扫码登录 / 登录态检查 / 持久化封装
//! - `storage` - SESSDATA / WBI key 落盘存储（阶段 4）
//! - `wbi_cache` - WBI mixinKey 拉取与 24h 内存+落盘缓存
//! - `error`   - 错误类型

pub mod client;
pub mod error;
pub mod login;
pub mod storage;
pub mod stream;
pub mod task;
pub mod types;
pub mod video;
pub mod wbi_cache;

pub use client::BiliClient;
pub use error::{BiliApiError, Result};
