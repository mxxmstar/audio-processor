//! 通用历史记录模块（阶段 D）
//!
//! 统一管理「音频识别」与「B站下载」两类历史，避免重复实现。
//! 数据存于本机 SQLite 文件 `history.db`（位于应用配置目录）。

use rusqlite::{Connection, Result as SqlResult};
use serde::Serialize;

/// 历史记录的种类，区分不同业务来源。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HistoryKind {
    /// 音频识别
    Recognize,
    /// B站下载
    Download,
}

impl HistoryKind {
    /// 存库用的字符串标识。
    pub fn as_str(&self) -> &'static str {
        match self {
            HistoryKind::Recognize => "recognize",
            HistoryKind::Download => "download",
        }
    }

    /// 展示用中文标签。
    pub fn label(&self) -> &'static str {
        match self {
            HistoryKind::Recognize => "音频识别",
            HistoryKind::Download => "B站下载",
        }
    }

    /// 从字符串解析（未知值回退为 `Recognize`）。
    pub fn from_str(s: &str) -> Self {
        match s {
            "download" => HistoryKind::Download,
            _ => HistoryKind::Recognize,
        }
    }
}

/// 历史记录行（返回给前端的 JSON 结构）
#[derive(Debug, Clone, Serialize)]
pub struct HistoryItem {
    pub id: i64,
    pub kind: String,
    pub title: String,
    pub subtitle: String,
    /// 业务详情的 JSON 字符串（识别为 SongInfo、下载为 DownloadTask）
    pub payload: String,
    /// 关联文件本地绝对路径（下载类记录为输出文件，识别类记录为空）
    pub file_path: String,
    pub created_at: String,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    kind        TEXT NOT NULL,
    title       TEXT NOT NULL,
    subtitle    TEXT NOT NULL DEFAULT '',
    payload     TEXT NOT NULL,
    file_path   TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_history_created ON history(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_history_kind ON history(kind);
";

/// 打开（或创建）历史数据库。
/// `dir` 为应用配置目录；若为空则用当前目录作为兜底。
pub fn open_db(dir: &std::path::Path) -> SqlResult<Connection> {
    let path = dir.join("history.db");
    let conn = Connection::open(&path)?;
    conn.execute_batch(SCHEMA)?;
    // 兼容旧库：若已存在的历史表缺少 file_path 列则补加（CREATE TABLE 不会自动变更已有表结构）
    let has_col: bool = conn
        .query_row(
            "SELECT COUNT(*) FROM pragma_table_info('history') WHERE name = 'file_path'",
            [],
            |r| r.get::<_, i64>(0).map(|c| c > 0),
        )
        .unwrap_or(false);
    if !has_col {
        conn.execute_batch("ALTER TABLE history ADD COLUMN file_path TEXT NOT NULL DEFAULT ''")?;
    }
    Ok(conn)
}

/// 插入一条历史记录。
pub fn insert(
    conn: &Connection,
    kind: HistoryKind,
    title: &str,
    subtitle: &str,
    payload: &str,
    file_path: &str,
) -> SqlResult<()> {
    conn.execute(
        "INSERT INTO history (kind, title, subtitle, payload, file_path) VALUES (?1, ?2, ?3, ?4, ?5)",
        (kind.as_str(), title, subtitle, payload, file_path),
    )?;
    Ok(())
}

/// 查询历史列表。
/// `kind` 为 `None` 时返回全部；`limit` 默认 200。
pub fn list(
    conn: &Connection,
    kind: Option<&str>,
    limit: usize,
) -> SqlResult<Vec<HistoryItem>> {
    let rows = match kind {
        Some(k) => {
            let mut stmt = conn.prepare(
                "SELECT id, kind, title, subtitle, payload, file_path, created_at
                 FROM history WHERE kind = ?1 ORDER BY created_at DESC LIMIT ?2",
            )?;
            let x = stmt
                .query_map(rusqlite::params![k, limit as i64], map_row)?
                .collect::<SqlResult<Vec<_>>>()?;
            x
        }
        None => {
            let mut stmt = conn.prepare(
                "SELECT id, kind, title, subtitle, payload, file_path, created_at
                 FROM history ORDER BY created_at DESC LIMIT ?1",
            )?;
            let x = stmt
                .query_map(rusqlite::params![limit as i64], map_row)?
                .collect::<SqlResult<Vec<_>>>()?;
            x
        }
    };
    Ok(rows)
}

/// 按 id 取单条。
pub fn get(conn: &Connection, id: i64) -> SqlResult<Option<HistoryItem>> {
    let mut stmt = conn.prepare(
        "SELECT id, kind, title, subtitle, payload, file_path, created_at FROM history WHERE id = ?1",
    )?;
    let mut rows = stmt.query_map([id], map_row)?;
    match rows.next() {
        Some(row) => Ok(Some(row?)),
        None => Ok(None),
    }
}

/// 按 id 删除单条。
pub fn delete(conn: &Connection, id: i64) -> SqlResult<()> {
    conn.execute("DELETE FROM history WHERE id = ?1", [id])?;
    Ok(())
}

/// 按种类清空（None 表示全部清空，调试用）。
pub fn clear(conn: &Connection, kind: Option<&str>) -> SqlResult<()> {
    match kind {
        Some(k) => conn.execute("DELETE FROM history WHERE kind = ?1", [k])?,
        None => conn.execute("DELETE FROM history", [])?,
    };
    Ok(())
}

/// 行映射辅助。
fn map_row(r: &rusqlite::Row) -> SqlResult<HistoryItem> {
    Ok(HistoryItem {
        id: r.get(0)?,
        kind: r.get(1)?,
        title: r.get(2)?,
        subtitle: r.get(3)?,
        payload: r.get(4)?,
        file_path: r.get(5)?,
        created_at: r.get(6)?,
    })
}
