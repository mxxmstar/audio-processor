//! 阶段 5：B 站下载相关 Tauri 命令
//!
//! 命令层把 `biliapi` 的能力暴露给前端：
//! - 解析 URL（视频 / 合集 / 番剧）→ 展开任务
//! - 登录态（生成二维码 / 轮询 / 校验 / 登出）
//! - 启动并发下载，进度经 Tauri 事件 `download-progress` 推送到前端

use crate::bili_state::BiliState;
use crate::biliapi::client::BiliClient;
use crate::biliapi::login;
use crate::biliapi::task::{DownloadMode, DownloadTask};
use crate::biliapi::types::{MediaFormat, QrInfo};
use crate::biliapi::video;
use serde::Serialize;
use std::sync::Arc;
use tauri::{AppHandle, Emitter, Manager, State};

/// 解析请求参数
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ResolveInput {
    /// 原始输入：BV 号 / AV 号 / 链接 / 合集(ss) / 番剧(ep)
    pub input: String,
    /// 下载模式：audio / video / merge（缺省 audio）
    #[serde(default)]
    pub mode: Option<String>,
    /// 优先清晰度（如 "1080P"、"720P"），缺省 "1080P"
    #[serde(default)]
    pub prefer_format: Option<String>,
    /// 输出目录（缺省用配置目录）
    #[serde(default)]
    pub output_dir: Option<String>,
}

/// 启动下载请求
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StartDownloadInput {
    #[serde(default)]
    pub output_dir: Option<String>,
    #[serde(default)]
    pub concurrency: Option<usize>,
}

/// 生成的登录二维码
#[derive(Debug, Clone, Serialize)]
pub struct LoginQr {
    /// 二维码 SVG（data URL 可直接 `<img src>` 渲染）
    pub qr_svg: String,
    pub qr_key: String,
}

/// 轮询二维码登录结果
#[derive(Debug, Clone, Serialize)]
pub struct LoginState {
    pub authed: bool,
    pub message: String,
}

/// 进度事件（推送到前端）
#[derive(Debug, Clone, Serialize)]
pub struct ProgressEvent {
    pub task_id: String,
    pub title: String,
    pub status: String,
    pub percent: f64,
    pub downloaded: u64,
    pub total: u64,
    pub speed: u64,
    pub error: Option<String>,
}

fn parse_mode(s: &str) -> DownloadMode {
    match s.to_lowercase().as_str() {
        "video" => DownloadMode::VideoOnly,
        "merge" => DownloadMode::Merge,
        _ => DownloadMode::AudioOnly,
    }
}

/// 清晰度 label → code（用于构建 prefer_format）
fn parse_format(label: &str) -> i64 {
    let l = label.trim().to_uppercase();
    let m = match l.as_str() {
        "360P" => MediaFormat::Q_360P,
        "720P" => MediaFormat::Q_720P,
        "1080P" | "1080" => MediaFormat::Q_1080P,
        "1080P+" | "1080P_PLUS" => MediaFormat::Q_1080P_PLUS,
        "4K" => MediaFormat::Q_4K,
        "DOLBY" => MediaFormat::Q_DOLBY,
        "HDR" => MediaFormat::Q_HDR,
        "8K" => MediaFormat::Q_8K,
        _ => MediaFormat::Q_1080P,
    };
    m.0
}

/// 从用户输入识别目标并解析为下载任务（不立即下载）
#[tauri::command]
pub async fn bili_resolve(input: ResolveInput, state: State<'_, BiliState>) -> Result<Vec<DownloadTask>, String> {
    let sessdata = login::load_and_check(state.config_dir_opt().as_deref())
        .await
        .map_err(|e| e.to_string())?
        .ok_or_else(|| "未登录或登录态已失效，请先扫码登录".to_string())?;

    let mode = parse_mode(input.mode.as_deref().unwrap_or("audio"));
    let prefer = parse_format(input.prefer_format.as_deref().unwrap_or("1080P"));
    let client = BiliClient::new(&sessdata);

    // 识别目标类型并分发解析
    let target = identify(&input.input);
    let results = match target {
        Target::Bv(bvid) => vec![video::resolve_video(&client, &bvid, prefer)
            .await
            .map_err(|e| e.to_string())?],
        Target::Av(aid) => {
            // AV 号需先转 BV 号；复用 view 接口拿 bvid
            let info = video::get_video_info(&client, &bv_from_aid(aid))
                .await
                .map_err(|e| e.to_string())?;
            vec![video::resolve_video(&client, &info.bvid, prefer)
                .await
                .map_err(|e| e.to_string())?]
        }
        Target::Collection(mid, sid) => video::resolve_collection(&client, &mid, &sid, prefer)
            .await
            .map_err(|e| e.to_string())?,
        Target::Season(ssid) => video::resolve_season(&client, &ssid, prefer)
            .await
            .map_err(|e| e.to_string())?,
    };

    let root = input.output_dir.clone().unwrap_or_else(|| {
        state
            .config_dir_opt()
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|| ".".to_string())
    });

    let tasks: Vec<DownloadTask> = DownloadTask::from_resolves(&results, mode, &root);
    if tasks.is_empty() {
        return Err("解析成功，但未找到可下载的音视频流".to_string());
    }

    state.set_tasks(tasks.clone());
    Ok(tasks)
}

/// 输入目标类型
enum Target {
    Bv(String),
    Av(i64),
    Collection(String, String), // (mid, season_id)
    Season(String),             // ssid
}

/// 从用户输入字符串识别目标：BV / AV / 链接中的 bvid / 合集 / 番剧
fn identify(input: &str) -> Target {
    let s = input.trim();
    // 纯 BV 号
    if let Some(rest) = s.strip_prefix("BV").or_else(|| s.strip_prefix("bv")) {
        if !rest.is_empty() {
            return Target::Bv(format!("BV{}", rest));
        }
    }
    // 纯 AV 号
    if let Some(rest) = s.strip_prefix("av").or_else(|| s.strip_prefix("AV")) {
        if let Ok(aid) = rest.parse::<i64>() {
            return Target::Av(aid);
        }
    }
    // 链接：提取 query 参数
    if s.contains("bilibili.com") {
        if let Some(ssid) = extract_query(s, "ssid").or_else(|| {
            extract_path_token(s, "ss")
        }) {
            return Target::Season(ssid);
        }
        if let Some(ep) = extract_query(s, "ep_id").or_else(|| extract_path_token(s, "ep")) {
            // ep 也需要 season 信息，但 resolve_season 仅接受 ssid；
            // 简化：ep 直接当作 ss 不可用，这里回退用 bvid 解析
            if let Some(bvid) = extract_query(s, "bvid") {
                return Target::Bv(bvid);
            }
            // ep 单独出现：尝试从链接拿 bvid
            if let Some(bvid) = extract_bvid_from_url(s) {
                return Target::Bv(bvid);
            }
            // 退化为 Season（ssid 缺失时由 video 层报错）
            return Target::Season(ep);
        }
        if let Some(bvid) = extract_query(s, "bvid").or_else(|| extract_bvid_from_url(s)) {
            return Target::Bv(bvid);
        }
        if let Some(mid) = extract_query(s, "mid") {
            if let Some(sid) = extract_query(s, "sid") {
                return Target::Collection(mid, sid);
            }
        }
    }
    // 兜底：当作 BV 号
    Target::Bv(s.to_string())
}

/// 从 URL 路径提取形如 /ss123/ 或 /ep123/ 的 token
fn extract_path_token(url: &str, prefix: &str) -> Option<String> {
    let pat = format!("/{}/", prefix);
    if let Some(pos) = url.find(&pat) {
        let after = &url[pos + pat.len()..];
        let end = after.find('/').unwrap_or(after.len());
        let tok = &after[..end];
        if !tok.is_empty() {
            return Some(tok.to_string());
        }
    }
    None
}

/// 从 query 提取参数值
fn extract_query(url: &str, key: &str) -> Option<String> {
    let marker = format!("{}=", key);
    let idx = url.find(&marker)?;
    let after = &url[idx + marker.len()..];
    let end = after
        .find(['&', '#', '/'])
        .unwrap_or(after.len());
    let v = &after[..end];
    if v.is_empty() {
        None
    } else {
        Some(v.to_string())
    }
}

/// 从 URL 提取 BV 号（形如 BV1xx...）
fn extract_bvid_from_url(url: &str) -> Option<String> {
    let idx = url.find("BV")?;
    let after = &url[idx..];
    let end = after
        .find(['/', '?', '#', '&'])
        .unwrap_or(after.len());
    let cand = &after[..end];
    if cand.len() > 2 {
        Some(cand.to_string())
    } else {
        None
    }
}

/// AV 号转 BV 号（B 站 base58 算法；仅用于链接场景）
fn bv_from_aid(aid: i64) -> String {
    // 简化：直接用 av 号回退；实际解析由 view 接口处理。
    // 这里我们返回空串再让 get_video_info 失败更明确，但为兼容，
    // video 层 get_video_info 仅接受 bvid。故这里做标准转换。
    const TABLE: &[u8] = b"fcCAfED2BxB7H9KIvMNT3nrS8Lp5gG4a1doe0j6mZQuyOhiktJWzlYwVPUX5sR";
    let xor = 23442827791579u64;
    let base = 58u64;
    let mut bv = [b'B', b'V', 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let mut tmp = aid as u64 ^ xor;
    let mut i = 0;
    while tmp > 0 {
        let rem = (tmp % base) as usize;
        bv[11 - i] = TABLE[rem];
        tmp /= base;
        i += 1;
    }
    // 按 B 站换位表重排
    let swap = |arr: &mut [u8], a: usize, b: usize| {
        arr.swap(a, b);
    };
    swap(&mut bv, 0, 9);
    swap(&mut bv, 3, 10);
    swap(&mut bv, 6, 4);
    swap(&mut bv, 7, 2);
    swap(&mut bv, 8, 1);
    String::from_utf8_lossy(&bv).into_owned()
}

/// 启动已解析任务的并发下载（后台运行，进度经事件推送）
#[tauri::command]
pub async fn bili_start_download(
    input: StartDownloadInput,
    app: AppHandle,
    state: State<'_, BiliState>,
) -> Result<Vec<String>, String> {
    let tasks = state.snapshot_tasks();
    if tasks.is_empty() {
        return Err("没有可下载的任务，请先调用 bili_resolve".to_string());
    }
    let concurrency = input.concurrency.unwrap_or(3).max(1);
    let client = Arc::new(crate::http_client::client::HttpClient::new());

    let app_for_cb = app.clone();
    let prog_cb: Option<Arc<dyn Fn(&DownloadTask, crate::http_client::types::Progress) + Send + Sync>> =
        Some(Arc::new(move |task: &DownloadTask, p: crate::http_client::types::Progress| {
            // 通过 AppHandle 取共享状态（AppHandle 为 'static，闭包内安全）
            let st = app_for_cb.state::<BiliState>();
            st.apply_results(std::slice::from_ref(task));
            let payload = ProgressEvent {
                task_id: task.id.clone(),
                title: task.title.clone(),
                status: status_label(task.status),
                percent: p.percent,
                downloaded: p.downloaded,
                total: p.total.unwrap_or(0),
                speed: p.speed,
                error: task.error.clone(),
            };
            let _ = app_for_cb.emit("download-progress", payload);
        }));

    let mut tasks_ref = tasks.clone();
    // 若下载命令显式指定了目录，覆盖各任务的输出目录（优先于解析时设定的值）
    if let Some(dir) = input.output_dir.clone().filter(|d| !d.is_empty()) {
        for t in tasks_ref.iter_mut() {
            t.output_dir = dir.clone();
        }
    }
    let app2 = app.clone();
    tauri::async_runtime::spawn(async move {
        let results = crate::biliapi::task::run_batch(client, &mut tasks_ref, concurrency, prog_cb).await;
        let failed: Vec<_> = results.iter().enumerate().filter(|(_, r)| r.is_err()).collect();
        let _ = app2.emit(
            "download-finished",
            serde_json::json!({ "ok": failed.is_empty(), "failed": failed.len() }),
        );
    });

    Ok(tasks.iter().map(|t| t.id.clone()).collect())
}

/// 查询当前任务列表与状态
#[tauri::command]
pub fn bili_list_tasks(state: State<'_, BiliState>) -> Result<Vec<DownloadTask>, String> {
    Ok(state.snapshot_tasks())
}

/// 生成登录二维码
#[tauri::command]
pub async fn bili_login_qr() -> Result<LoginQr, String> {
    let info: QrInfo = login::new_qr_info().await.map_err(|e| e.to_string())?;
    let svg = login::generate_qr_svg(&info.url).map_err(|e| e.to_string())?;
    Ok(LoginQr {
        qr_svg: format!("data:image/svg+xml;charset=utf-8,{}", urlencoding(&svg)),
        qr_key: info.qrcode_key,
    })
}

/// 对 SVG 字符串做 URL 编码以嵌入 data URL（避免 `#`/`"` 等破坏属性）
fn urlencoding(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => {
                out.push('%');
                out.push_str(&format!("{:02X}", b));
            }
        }
    }
    out
}

/// 轮询登录二维码状态；成功则自动持久化 SESSDATA
#[tauri::command]
pub async fn bili_login_poll(qr_key: String, state: State<'_, BiliState>) -> Result<LoginState, String> {
    let (status, sessdata) = login::get_qr_status(&qr_key).await.map_err(|e| e.to_string())?;
    if status.code == crate::biliapi::types::QR_SUCCESS {
        login::persist_login(state.config_dir_opt().as_deref(), sessdata.as_deref());
        Ok(LoginState {
            authed: true,
            message: status.message,
        })
    } else {
        Ok(LoginState {
            authed: false,
            message: status.message,
        })
    }
}

/// 校验当前是否已登录
#[tauri::command]
pub async fn bili_check_login(state: State<'_, BiliState>) -> Result<bool, String> {
    Ok(login::load_and_check(state.config_dir_opt().as_deref())
        .await
        .map_err(|e| e.to_string())?
        .is_some())
}

/// 登出：清除持久化登录态
#[tauri::command]
pub fn bili_logout(state: State<'_, BiliState>) -> Result<(), String> {
    let dir = state.config_dir_opt();
    crate::biliapi::storage::clear_sessdata(dir.as_deref());
    crate::biliapi::storage::clear_wbi(dir.as_deref());
    crate::biliapi::wbi_cache::clear_cache();
    state.set_tasks(Vec::new());
    Ok(())
}

/// 将 `DownloadStatus` 转为可读中文标签
fn status_label(s: crate::biliapi::task::DownloadStatus) -> String {
    use crate::biliapi::task::DownloadStatus;
    match s {
        DownloadStatus::Pending => "等待中",
        DownloadStatus::Downloading => "下载中",
        DownloadStatus::Completed => "已完成",
        DownloadStatus::Failed => "失败",
    }
    .to_string()
}
