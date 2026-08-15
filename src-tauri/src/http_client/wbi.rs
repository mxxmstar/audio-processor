//! B 站 WBI 签名实现
//!
//! 对应 Go 版 `bilidownload/server/bilibili/wbi.go`。
//!
//! B 站部分接口（如 `x/player/playurl`、`x/web-interface/wbi/view`）要求请求携带
//! 经过 WBI（Web 接口）签名的 `w_rid` 与 `wts` 参数，否则返回 `-404` / `-412`。
//!
//! 签名流程：
//! 1. 调用 `x/web-interface/nav` 获取 `img_url` / `sub_url`，提取出 `img_key` / `sub_key`；
//! 2. 用内置混淆表 `MIXIN_KEY_ENC_TAB` 对 `sub_key + img_key` 重排，取前 32 位得到 `mixin_key`；
//! 3. 将业务参数排序后加入时间戳 `wts`，`url::form_urlencoded` 编码（注意 `+` 需替换为 `%20`），
//!    末尾拼接 `mixin_key`，取 MD5（`md5` crate）得到 `w_rid`。
//!
//! MD5 使用成熟的 `md5` crate，避免手写实现。

use md5;
use std::collections::BTreeMap;

/// 混淆表（64 位），用于重排 wbi key。
/// 与 B 站官方及 Go 版 wbi.go 保持一致（标准 64 元素）。
const MIXIN_KEY_ENC_TAB: [usize; 64] = [
    46, 47, 18, 2, 53, 8, 23, 32, 15, 50, 10, 31, 58, 3, 45, 35, 27, 43, 5, 49, 33, 9, 42, 19, 29,
    28, 14, 39, 12, 38, 41, 13, 37, 48, 7, 16, 24, 55, 40, 61, 26, 17, 0, 1, 57, 22, 45, 34, 44,
    52, 59, 6, 60, 25, 54, 11, 36, 21, 56, 51, 62, 20, 4, 30,
];

/// 从 nav 接口返回的 `img_url` / `sub_url` 提取 key 片段。
///
/// 例如 `https://i0.hdslb.com/bfs/wbi/abc123.png` -> `abc123`。
fn extract_key_from_url(url: &str) -> String {
    let path = url.rsplit('/').next().unwrap_or("");
    path.trim_end_matches(".png")
        .trim_end_matches(".jpg")
        .trim_end_matches(".webp")
        .to_string()
}

/// 生成 mixinKey：将 `sub_key + img_key` 按混淆表重排，取前 32 位。
pub fn get_mixin_key(img_key: &str, sub_key: &str) -> String {
    let raw = format!("{}{}", sub_key, img_key);
    let bytes = raw.as_bytes();
    let mut mixin = String::with_capacity(32);
    for &i in MIXIN_KEY_ENC_TAB.iter() {
        if i < bytes.len() {
            mixin.push(bytes[i] as char);
        }
        if mixin.len() == 32 {
            break;
        }
    }
    mixin
}

/// 计算 MD5 十六进制字符串（使用 `md5` crate 0.7 的 `compute`）。
fn md5_hex(input: &str) -> String {
    let digest = md5::compute(input.as_bytes());
    format!("{:x}", digest)
}

/// 对已传入的参数集合进行 WBI 签名。
///
/// # 参数
/// * `params` - 业务参数（不含 `wts` / `w_rid`，函数内部自动添加 `wts`）。
/// * `mixin_key` - 由 [`get_mixin_key`] 生成的混淆 key。
///
/// # 返回值
/// 返回完整的、已带 `wts` 和 `w_rid` 的查询参数列表，可直接用于 `RequestConfig::query`。
pub fn sign(params: &[(&str, &str)], mixin_key: &str) -> Vec<(String, String)> {
    let wts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // 排序（按 key 字典序）
    let mut sorted: BTreeMap<String, String> = BTreeMap::new();
    for (k, v) in params {
        sorted.insert((*k).to_string(), (*v).to_string());
    }
    sorted.insert("wts".to_string(), wts.to_string());

    // urlencode，并将 '+' 替换为 '%20'（B 站要求）
    let mut query = String::new();
    for (i, (k, v)) in sorted.iter().enumerate() {
        if i > 0 {
            query.push('&');
        }
        query.push_str(&format!("{}={}", url_encode(k), url_encode(v).replace('+', "%20")));
    }

    let to_sign = format!("{}&{}", query, mixin_key);
    let w_rid = md5_hex(&to_sign);

    let mut out: Vec<(String, String)> = sorted.into_iter().collect();
    out.push(("w_rid".to_string(), w_rid));
    out
}

/// 对单个字符串做百分号编码（保留 RFC 3986 未保留字符）。
fn url_encode(input: &str) -> String {
    let mut out = String::with_capacity(input.len() * 3);
    for byte in input.bytes() {
        match byte {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(byte as char);
            }
            _ => {
                out.push_str(&format!("%{:02X}", byte));
            }
        }
    }
    out
}

/// 从 nav 接口响应中提取 img_key / sub_key 并生成 mixin_key。
///
/// # 参数
/// * `img_url` - nav 响应的 `wbi_img.img_url`
/// * `sub_url` - nav 响应的 `wbi_img.sub_url`
pub fn mixin_key_from_nav(img_url: &str, sub_url: &str) -> String {
    let img_key = extract_key_from_url(img_url);
    let sub_key = extract_key_from_url(sub_url);
    get_mixin_key(&img_key, &sub_key)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mixin_key_from_nav() {
        // 使用长度足够的 key（模拟真实 B 站 key，各 16 位，拼接后 32 位）
        let mk = mixin_key_from_nav(
            "https://i0.hdslb.com/bfs/wbi/abcdefghijklmnop.png",
            "https://i0.hdslb.com/bfs/wbi/qrstuvwxyz012345.jpg",
        );
        assert_eq!(mk.len(), 32);
        assert!(mk.chars().all(|c| c.is_ascii_alphanumeric()));
    }

    #[test]
    fn test_sign_adds_wts_and_w_rid() {
        let mk = "abcdefghijklmnopqrstuvwxyz012345";
        let signed = sign(&[("bvid", "BV1xx411c7XD")], mk);
        let keys: Vec<&String> = signed.iter().map(|(k, _)| k).collect();
        assert!(keys.contains(&&"wts".to_string()));
        assert!(keys.contains(&&"w_rid".to_string()));
    }

    #[test]
    fn test_md5_known_vector() {
        assert_eq!(md5_hex(""), "d41d8cd98f00b204e9800998ecf8427e");
        assert_eq!(md5_hex("abc"), "900150983cd24fb0d6963f7d28e17f72");
    }
}
