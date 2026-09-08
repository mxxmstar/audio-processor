//! biliapi 模块错误类型

use thiserror::Error;

/// B 站常见业务错误码的中文说明。
///
/// 措辞有意保持「或」式并列而不下断言：同一个错误码常对应多种成因，例如 `-404`
/// 既可能是稿件不存在/已删除，也可能是大会员专享或地区限制，不能只报其中一种。
/// 未收录的码回退到通用文案，原始 `code` / `message` 始终保留在末尾供排查。
fn describe_api_error(code: i64) -> &'static str {
    match code {
        -404 => "视频不存在、已删除，或需要大会员 / 受地区限制",
        -403 => "没有访问权限，可能需要登录后重试",
        -412 => "请求被风控拦截，请稍后重试，或登录后再试",
        -101 => "尚未登录，请先扫码登录后再试",
        _ => "B 站接口返回错误",
    }
}

#[derive(Debug, Error)]
pub enum BiliApiError {
    #[error("HTTP 请求失败: {0}")]
    Http(#[from] crate::http_client::error::HttpClientError),

    #[error("JSON 解析失败: {0}")]
    Json(#[from] serde_json::Error),

    #[error("{}（B 站错误码 {}：{}）", describe_api_error(*code), code, message)]
    Api { code: i64, message: String },

    #[error("WBI 签名所需 SESSDATA 为空")]
    EmptySessdata,

    #[error("未找到 Cookie: {0}")]
    CookieNotFound(String),

    #[error("二维码状态: {0}")]
    QrStatus(String),

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, BiliApiError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_codes_render_friendly_text() {
        let cases = [
            (-404, "啥都木有", "需要大会员"),
            (-412, "请求被拦截", "风控"),
            (-403, "访问权限不足", "没有访问权限"),
            (-101, "用户未登录", "尚未登录"),
        ];
        for (code, upstream, expected_fragment) in cases {
            let err = BiliApiError::Api {
                code,
                message: upstream.to_string(),
            };
            let text = err.to_string();
            assert!(
                text.contains(expected_fragment),
                "code={code} 应展示友好文案，实际：{text}"
            );
            // 原始信息必须保留，便于用户反馈与排查
            assert!(text.contains(&code.to_string()), "应保留原始错误码：{text}");
            assert!(text.contains(upstream), "应保留 B 站原始 message：{text}");
        }
    }

    #[test]
    fn unknown_code_falls_back_to_generic_text() {
        let err = BiliApiError::Api {
            code: -123456,
            message: "未知业务错误".to_string(),
        };
        let text = err.to_string();
        assert!(text.contains("-123456"), "{text}");
        assert!(text.contains("未知业务错误"), "{text}");
        // 未收录的码不应套用任何具体成因的措辞
        assert!(!text.contains("大会员"), "{text}");
        assert!(!text.contains("风控"), "{text}");
    }
}
