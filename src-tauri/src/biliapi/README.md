# biliapi 模块说明

> 路径：`src-tauri/src/biliapi/`
> 定位：把 `bilidownload/server/bilibili/` 中调用 B 站的接口封装为异步 Rust API。
> 底层依赖：`http_client`（WBI 签名、强制直连、Mozilla UA）。

## 1. 模块结构

```
biliapi/
├── mod.rs       # 导出子模块，重导出 BiliClient / BiliApiError / Result
├── client.rs    # BiliClient：持有 HttpClient + SESSDATA，统一请求构造与发送
├── types.rs     # B 站响应/业务数据结构（对齐 Go 版 type.go）
├── video.rs     # 视频/番剧/收藏夹/合集/热门 API
├── login.rs     # 扫码登录 + 登录态检查
├── wbi_cache.rs # WBI mixinKey 拉取与 24h 内存缓存
└── error.rs     # BiliApiError 错误类型
```

## 2. 已封装的 B 站 API（对齐 bilidownload）

| 功能 | 函数 | 对应 Go 接口 | WBI 签名 |
|---|---|---|---|
| 视频信息 | `video::get_video_info(client, bvid)` | `x/web-interface/wbi/view` | ✅ |
| 番剧信息 | `video::get_season_info(client, epid, ssid)` | `pgc/view/web/season` | ❌ |
| 播放直链 | `video::get_play_info(client, bvid, cid, format)` | `x/player/playurl` | ✅ |
| 收藏夹列表 | `video::get_fav_list(client, media_id, pn, ps)` | `x/v3/fav/resource/list` | ❌ |
| 合集首个视频 | `video::get_seasons_archives_first_bvid(client, mid, season_id)` | `x/polymer/web-space/seasons_archives_list` | ❌ |
| 热门视频 | `video::get_popular_videos(client, pn, ps)` | `x/web-interface/popular` | ❌ |
| 登录二维码 | `login::new_qr_info()` | `passport-login/web/qrcode/generate` | ❌ |
| 扫码状态 | `login::get_qr_status(qr_key)` | `passport-login/web/qrcode/poll` | ❌ |
| 登录检查 | `login::check_login(sessdata)` | `x/space/myinfo` | ❌ |

## 3. 设计要点

### 3.1 登录态注入（解耦数据库）
`BiliClient::new(sessdata)` 由调用方注入 `SESSDATA`，模块本身**不依赖 SQLite**，
便于与 `audio-processor` 主项目的存储层（如 Tauri 状态、配置文件）解耦。

### 3.2 标准请求头
`BiliClient::get(url)` 统一注入：
- `Cookie: SESSDATA=...`（非空时）
- `Referer: https://www.bilibili.com`
- `User-Agent: Mozilla/5.0`（由 `http_client` 默认提供）

### 3.3 WBI 签名自动注入
需要签名的接口（view / playurl）调用 `client.with_wbi(cfg)`：
1. 从 `wbi_cache` 获取 mixinKey（首次从 `x/web-interface/nav` 拉取，缓存 24h）；
2. 对已有 query 参数追加 `wts` + `w_rid`（`http_client::wbi::sign`）。

### 3.4 错误统一
所有接口返回 `biliapi::Result<T>`，`BaseRes<T>::into_result()` 在 `code != 0` 时
转为 `BiliApiError::Api { code, message }`。

## 4. 使用示例

```rust
use crate::biliapi::{BiliClient, video, login};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let sessdata = std::env::var("SESSDATA").unwrap_or_default();
    let client = BiliClient::new(sessdata);

    // 视频信息（自动 WBI 签名）
    let info = video::get_video_info(&client, "BV1xx411c7XD").await?;
    println!("标题: {}", info.title);

    // 播放直链
    if let Some(cid) = info.pages.first().map(|p| p.cid) {
        let play = video::get_play_info(&client, &info.bvid, cid, 80).await?;
        if let Some(dash) = play.dash {
            println!("视频流数: {}", dash.video.len());
        }
    }

    // 扫码登录
    let qr = login::new_qr_info().await?;
    println!("扫码 URL: {}", qr.url);
    Ok(())
}
```

## 5. 测试

```bash
cargo test --lib biliapi
```

覆盖：`BaseRes` 反序列化（成功/错误路径）、WBI 常量、二维码状态常量。

## 6. 待补充（对照 bilidownload 完整能力）

- `getPlayInfo` 后挑选音视频直链（Go 版 `GetVideoURL`/`GetAudioURL`，按 Codecid 优先级）；
- 批量解析（合集、番剧多分集）的遍历逻辑；
- `audio`/`video`/`merge` 三种下载模式的音视频拉流（依赖 `http_client` 流式下载，见 http_client 文档 P2）。
