//! 识别历史记录的 SQLite 持久化（阶段 B）
//!
//! 把每次音频识别结果落盘到本机 SQLite，供前端「历史记录」子菜单回顾。
//! 数据库文件位于应用配置目录（`app_config_dir`）下的 `recognize_history.db`。

use crate::recognizer::SongInfo;
use rusqlite::{Connection, Result as SqlResult};
use serde::Serialize;

/// 历史记录行（返回给前端的 JSON 结构）
#[derive(Debug, Clone, Serialize)]
pub struct HistoryRecord {
    pub id: i64,
    pub title: String,
    pub artist: String,
    pub album: Option<String>,
    pub album_date: Option<String>,
    pub confidence: f64,
    pub file_path: String,
    pub created_at: String,
}

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS recognize_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    title       TEXT NOT NULL,
    artist      TEXT NOT NULL,
    album       TEXT,
    album_date  TEXT,
    confidence  REAL NOT NULL,
    file_path   TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now','localtime'))
);
CREATE INDEX IF NOT EXISTS idx_history_created ON recognize_history(created_at DESC);
";

/// 打开（或创建）历史数据库。
/// `dir` 为应用配置目录；若为空则用当前目录作为兜底。
pub fn open_db(dir: &std::path::Path) -> SqlResult<Connection> {
    let path = dir.join("recognize_history.db");
    let conn = Connection::open(&path)?;
    conn.execute_batch(SCHEMA)?;
    Ok(conn)
}

/// 插入一条识别记录；`created_at` 由数据库默认值填充。
pub fn insert_record(conn: &Connection, info: &SongInfo, file_path: &str) -> SqlResult<()> {
    conn.execute(
        "INSERT INTO recognize_history
         (title, artist, album, album_date, confidence, file_path)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        (
            &info.title,
            &info.artist,
            &info.album,
            &info.album_date,
            info.confidence,
            file_path,
        ),
    )?;
    Ok(())
}

/// 按时间倒序返回最近 `limit` 条记录（默认 100）。
pub fn list_records(conn: &Connection, limit: usize) -> SqlResult<Vec<HistoryRecord>> {
    let mut stmt = conn.prepare(
        "SELECT id, title, artist, album, album_date, confidence, file_path, created_at
         FROM recognize_history ORDER BY created_at DESC LIMIT ?1",
    )?;
    let rows = stmt.query_map([limit], |r| {
        Ok(HistoryRecord {
            id: r.get(0)?,
            title: r.get(1)?,
            artist: r.get(2)?,
            album: r.get(3)?,
            album_date: r.get(4)?,
            confidence: r.get(5)?,
            file_path: r.get(6)?,
            created_at: r.get(7)?,
        })
    })?;
    rows.collect()
}

/// 按 id 取单条记录。
pub fn get_record(conn: &Connection, id: i64) -> SqlResult<Option<HistoryRecord>> {
    let mut stmt = conn.prepare(
        "SELECT id, title, artist, album, album_date, confidence, file_path, created_at
         FROM recognize_history WHERE id = ?1",
    )?;
    let mut rows = stmt.query_map([id], |r| {
        Ok(HistoryRecord {
            id: r.get(0)?,
            title: r.get(1)?,
            artist: r.get(2)?,
            album: r.get(3)?,
            album_date: r.get(4)?,
            confidence: r.get(5)?,
            file_path: r.get(6)?,
            created_at: r.get(7)?,
        })
    })?;
    match rows.next() {
        Some(row) => Ok(Some(row?)),
        None => Ok(None),
    }
}

/// 按 id 删除单条记录。
pub fn delete_record(conn: &Connection, id: i64) -> SqlResult<()> {
    conn.execute("DELETE FROM recognize_history WHERE id = ?1", [id])?;
    Ok(())
}

/// 清空全部历史（调试用）。
pub fn clear_all(conn: &Connection) -> SqlResult<()> {
    conn.execute("DELETE FROM recognize_history", [])?;
    Ok(())
}
