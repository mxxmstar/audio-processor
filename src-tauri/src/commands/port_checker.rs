//! 端口查询相关 Tauri 命令

use crate::port_checker::{self, PortInfo, PortQuery};

/// 查询端口占用情况
///
/// 支持按端口号、协议、状态、PID 过滤。
/// 不传过滤条件则返回所有端口信息。
#[tauri::command]
pub fn port_query(filter: Option<PortQuery>) -> Result<Vec<PortInfo>, String> {
    port_checker::query_ports(filter.as_ref())
}

/// 快速检查指定端口是否已被占用
///
/// 尝试建立 TCP 连接来判断，仅适用于 TCP 协议。
#[tauri::command]
pub fn port_check(port: u16) -> Result<bool, String> {
    port_checker::check_port(port)
}

/// 查询指定进程占用的所有端口
#[tauri::command]
pub fn port_query_by_pid(pid: u32) -> Result<Vec<PortInfo>, String> {
    port_checker::query_ports_by_pid(pid)
}

/// 终止指定 PID 的进程
#[tauri::command]
pub fn port_kill_process(pid: u32) -> Result<String, String> {
    port_checker::kill_process(pid)
}

/// 终止占用指定端口的进程
///
/// 自动查找占用该端口的监听进程并强制终止，返回被终止的进程信息。
#[tauri::command]
pub fn port_kill_by_port(port: u16) -> Result<PortInfo, String> {
    port_checker::kill_process_by_port(port)
}