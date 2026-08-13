use thiserror::Error;

/// 应用统一的错误类型。
/// 使用 `thiserror` 宏为每种错误自动生成 `Display` 实现，
/// 前端拿到的是已经本地化（中文）的错误描述字符串。
#[derive(Error, Debug)]
pub enum AppError {
    #[error("无法打开音频文件: {0}")]
    OpenFile(String),

    #[error("音频解码失败: {0}")]
    Decode(String),

    #[error("生成指纹失败: {0}")]
    Fingerprint(String),

    #[error("AcoustID 请求失败: {0}")]
    AcoustidRequest(String),

    #[error("AcoustID 未识别到任何歌曲")]
    NoMatch,

    #[error("MusicBrainz 请求失败: {0}")]
    MusicBrainzRequest(String),
}

/// 统一结果别名，简化各模块函数签名。
pub type Result<T> = std::result::Result<T, AppError>;
