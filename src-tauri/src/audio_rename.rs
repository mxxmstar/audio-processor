//! 音频文件命名、冲突处理和最终落盘。
//!
//! 本模块只处理本地字符串和文件系统，不调用识别服务。识别编排由下载命令层负责，
//! 这样命名规则可以在不依赖网络的情况下单元测试。

use crate::recognizer::SongInfo;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// 自动重命名配置。
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct AutoRenameConfig {
    pub enabled: bool,
    pub confidence_threshold: f64,
    /// 当前首期支持 `{title}` 和 `{artist}`。
    pub template: String,
    /// 最终输出扩展名。首期固定为 mp3，但保留字段方便后续扩展。
    pub extension: String,
}

impl Default for AutoRenameConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            confidence_threshold: 70.0,
            template: "{title}-{artist}".into(),
            extension: "mp3".into(),
        }
    }
}

/// 进程内串行化“检查目标 + 移动源文件”，避免批量任务竞争同一个序号。
static RENAME_LOCK: Mutex<()> = Mutex::new(());

/// 清理一个文件名组件，避免标题或艺术家改变目录结构。
pub fn sanitize_component(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut previous_space = false;
    for c in input.chars() {
        let invalid = c == '\\'
            || c == '/'
            || c == ':'
            || c == '*'
            || c == '?'
            || c == '"'
            || c == '<'
            || c == '>'
            || c == '|'
            || c.is_control();
        if invalid {
            out.push('_');
            previous_space = false;
        } else if c.is_whitespace() {
            if !previous_space {
                out.push(' ');
            }
            previous_space = true;
        } else {
            out.push(c);
            previous_space = false;
        }
    }

    let mut cleaned = out.trim_matches(|c: char| c == ' ' || c == '.').to_string();
    // Windows 对这些名称不允许直接作为文件名基础名。
    let reserved = cleaned
        .rsplit_once('.')
        .map(|(base, _)| base)
        .unwrap_or(&cleaned)
        .to_ascii_uppercase();
    if matches!(reserved.as_str(), "CON" | "PRN" | "AUX" | "NUL")
        || (reserved.len() == 4
            && (reserved.starts_with("COM") || reserved.starts_with("LPT"))
            && reserved.as_bytes()[3].is_ascii_digit())
    {
        cleaned.insert(0, '_');
    }

    // 给扩展名和目录路径预留空间；按 Unicode 标量值截断，不切断 UTF-8。
    cleaned = cleaned.chars().take(160).collect();
    if cleaned.is_empty() {
        "未命名音频".into()
    } else {
        cleaned
    }
}

/// 根据识别结果生成文件名基础名。
pub fn recognized_stem(info: &SongInfo, source_title: &str, template: &str) -> String {
    let title = sanitize_component(&info.title);
    let artist = sanitize_component(&info.artist);
    let source = sanitize_component(source_title);
    let rendered = template
        .replace(
            "{title}",
            if title == "未命名音频" {
                &source
            } else {
                &title
            },
        )
        .replace(
            "{artist}",
            if artist == "未命名音频" {
                ""
            } else {
                &artist
            },
        );
    let rendered = rendered.trim().trim_matches('-').trim().to_string();
    sanitize_component(if rendered.is_empty() {
        &source
    } else {
        &rendered
    })
}

/// 使用来源标题生成兜底文件名基础名。
pub fn fallback_stem(source_title: &str) -> String {
    sanitize_component(source_title)
}

/// 为下载任务生成稳定的临时文件路径。
pub fn staging_path(output_dir: &Path, task_id: &str, source_ext: &str) -> PathBuf {
    let key = sanitize_component(task_id).replace(' ', "_");
    output_dir.join(format!(".audio-processor-{}.{}.part", key, source_ext))
}

/// 将源文件移动到不冲突的目标文件名，绝不主动覆盖已有文件。
pub fn move_to_unique(
    source: &Path,
    output_dir: &Path,
    stem: &str,
    extension: &str,
) -> Result<PathBuf, String> {
    if !source.exists() {
        return Err(format!("待移动的音频文件不存在: {}", source.display()));
    }
    let _guard = RENAME_LOCK
        .lock()
        .map_err(|_| "获取文件重命名锁失败".to_string())?;
    let extension = extension.trim_start_matches('.');
    let base = sanitize_component(stem);
    for index in 0..10_000usize {
        let suffix = if index == 0 {
            String::new()
        } else {
            format!(" ({index})")
        };
        let candidate = output_dir.join(format!("{base}{suffix}.{extension}"));
        if candidate.exists() {
            continue;
        }
        match std::fs::rename(source, &candidate) {
            Ok(()) => return Ok(candidate),
            Err(_e) if candidate.exists() => continue,
            Err(e) => return Err(format!("移动音频文件失败: {e}")),
        }
    }
    Err("同名音频文件过多，无法分配新的文件名".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn song(title: &str, artist: &str) -> SongInfo {
        SongInfo {
            title: title.into(),
            artist: artist.into(),
            album: None,
            album_date: None,
            confidence: 95.0,
        }
    }

    #[test]
    fn test_default_title_artist_template() {
        let result = recognized_stem(&song("Song", "Artist"), "Source", "{title}-{artist}");
        assert_eq!(result, "Song-Artist");
    }

    #[test]
    fn test_sanitize_windows_chars_and_reserved_name() {
        assert_eq!(sanitize_component(" CON: test? "), "CON_ test_");
        assert_eq!(sanitize_component("CON"), "_CON");
        assert_eq!(sanitize_component("a/b"), "a_b");
    }

    #[test]
    fn test_move_to_unique_does_not_overwrite() {
        let dir = std::env::temp_dir().join(format!("audio_rename_{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        let source = dir.join("source.part");
        fs::write(&source, b"audio").unwrap();
        fs::write(dir.join("Song-Artist.mp3"), b"existing").unwrap();

        let target = move_to_unique(&source, &dir, "Song-Artist", "mp3").unwrap();
        assert_eq!(target.file_name().unwrap(), "Song-Artist (1).mp3");
        assert_eq!(fs::read(target).unwrap(), b"audio");
        let _ = fs::remove_dir_all(&dir);
    }
}
