//! 端口占用查询模块
//!
//! 提供查询系统端口占用情况的功能，支持：
//! - 查询所有正在监听的端口
//! - 查询指定端口是否被占用
//! - 按进程 ID 查询关联的端口

use std::collections::HashMap;
use std::net::TcpStream;
use std::time::Duration;

/// 端口信息
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PortInfo {
    /// 协议（TCP / UDP）
    pub protocol: String,
    /// 本地 IP 地址
    pub local_ip: String,
    /// 本地端口
    pub local_port: u16,
    /// 远程 IP 地址
    pub remote_ip: String,
    /// 远程端口
    pub remote_port: u16,
    /// 连接状态（LISTENING / ESTABLISHED 等）
    pub state: String,
    /// 占用进程的 PID
    pub pid: u32,
    /// 占用进程名称（可能为空）
    pub process_name: String,
}

/// 端口查询条件
#[derive(Debug, Clone, serde::Deserialize)]
pub struct PortQuery {
    /// 按端口号精确查找
    pub port: Option<u16>,
    /// 按协议过滤（tcp / udp）
    pub protocol: Option<String>,
    /// 按状态过滤（如 LISTENING）
    pub state: Option<String>,
    /// 按 PID 过滤
    pub pid: Option<u32>,
}

/// 查询端口占用情况
///
/// 在 Windows 上使用 `netstat -ano` 命令获取系统端口信息，
/// 并通过 `tasklist` 获取进程名称。
pub fn query_ports(filter: Option<&PortQuery>) -> Result<Vec<PortInfo>, String> {
    let output = run_netstat()?;
    let mut ports = parse_netstat_output(&output)?;

    // 获取进程名映射
    let pid_names = get_process_names();

    // 填充进程名
    for port in &mut ports {
        if let Some(name) = pid_names.get(&port.pid) {
            port.process_name = name.clone();
        }
    }

    // 应用过滤条件
    if let Some(f) = filter {
        ports.retain(|p| {
            if let Some(port) = f.port {
                if p.local_port != port {
                    return false;
                }
            }
            if let Some(ref protocol) = f.protocol {
                if !p.protocol.eq_ignore_ascii_case(protocol) {
                    return false;
                }
            }
            if let Some(ref state) = f.state {
                if !p.state.eq_ignore_ascii_case(state) {
                    return false;
                }
            }
            if let Some(pid) = f.pid {
                if p.pid != pid {
                    return false;
                }
            }
            true
        });
    }

    Ok(ports)
}

/// 快速检查指定端口是否已被占用
///
/// 尝试建立 TCP 连接来判断端口是否被占用。
/// 仅适用于 TCP 协议，适合快速判断。
pub fn check_port(port: u16) -> Result<bool, String> {
    let addr = format!("127.0.0.1:{port}");
    let socket_addr: std::net::SocketAddr = addr
        .parse()
        .map_err(|e: std::net::AddrParseError| format!("地址解析失败: {e}"))?;
    match TcpStream::connect_timeout(&socket_addr, Duration::from_millis(500)) {
        Ok(_) => Ok(true),
        Err(e) => {
            if e.kind() == std::io::ErrorKind::ConnectionRefused {
                Ok(false)
            } else {
                Err(format!("检查端口 {port} 失败: {e}"))
            }
        }
    }
}

/// 终止指定 PID 的进程
///
/// 在 Windows 上使用 `taskkill /F /PID <pid>` 强制终止进程。
/// 返回被杀进程的进程名（如能获取到）。
pub fn kill_process(pid: u32) -> Result<String, String> {
    let process_name = get_process_names().get(&pid).cloned().unwrap_or_default();

    let output = std::process::Command::new("taskkill")
        .args(["/F", "/PID", &pid.to_string()])
        .output()
        .map_err(|e| format!("执行 taskkill 失败: {e}"))?;

    if output.status.success() {
        Ok(process_name)
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // taskkill 中文输出可能包含 "成功" 但退出码非零，检查 stdout
        let stdout = String::from_utf8_lossy(&output.stdout);
        if stdout.contains("成功") || stderr.contains("成功") {
            Ok(process_name)
        } else {
            let msg = if !stderr.is_empty() { stderr } else { stdout };
            Err(format!("终止进程 {pid} 失败: {msg}"))
        }
    }
}

/// 终止占用指定端口的进程
///
/// 自动查找占用该端口的第一个进程并终止。
/// 返回被终止的进程信息（PID、进程名、协议）。
pub fn kill_process_by_port(port: u16) -> Result<PortInfo, String> {
    let filter = PortQuery {
        port: Some(port),
        protocol: None,
        state: Some("LISTENING".to_string()),
        pid: None,
    };
    let ports = query_ports(Some(&filter))?;

    if ports.is_empty() {
        return Err(format!("端口 {port} 未被占用"));
    }

    // 优先杀 LISTENING 状态的 TCP 进程
    let target = ports
        .iter()
        .find(|p| p.protocol == "TCP" && p.state == "LISTENING")
        .or_else(|| ports.first())
        .ok_or_else(|| format!("端口 {port} 未被占用"))?;

    let pid = target.pid;
    if pid == 0 || pid == 4 {
        return Err(format!("端口 {port} 被系统进程 (PID={pid}) 占用，无法终止"));
    }

    kill_process(pid)?;
    Ok(target.clone())
}

/// 查询指定进程占用的端口
pub fn query_ports_by_pid(pid: u32) -> Result<Vec<PortInfo>, String> {
    let filter = PortQuery {
        port: None,
        protocol: None,
        state: None,
        pid: Some(pid),
    };
    query_ports(Some(&filter))
}

/// 运行 netstat 命令并返回输出
fn run_netstat() -> Result<String, String> {
    let output = std::process::Command::new("netstat")
        .args(["-ano"])
        .output()
        .map_err(|e| format!("执行 netstat 失败: {e}"))?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(format!("netstat 执行失败: {stderr}"));
    }

    String::from_utf8(output.stdout).map_err(|e| format!("netstat 输出编码错误: {e}"))
}

/// 解析 netstat -ano 的输出
///
/// 输出格式示例：
/// ```
/// Proto  Local Address          Foreign Address        State           PID
/// TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1056
/// TCP    [::]:135               [::]:0                 LISTENING       1056
/// UDP    0.0.0.0:500            *:*                                    1234
/// ```
fn parse_netstat_output(output: &str) -> Result<Vec<PortInfo>, String> {
    let mut ports = Vec::new();

    for line in output.lines() {
        let line = line.trim();
        // 跳过空行和表头
        if line.is_empty() || line.starts_with("Proto") || line.starts_with("活动连接") {
            continue;
        }

        // 按空白分割
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() < 4 {
            continue;
        }

        let protocol = parts[0].to_uppercase();
        if protocol != "TCP" && protocol != "UDP" {
            continue;
        }

        let local = parse_address(parts[1])?;
        let remote = parse_address(parts[2])?;

        let state = if protocol == "TCP" && parts.len() > 3 {
            // TCP 第4列是状态，第5列是 PID
            if parts.len() > 4 {
                parts[3].to_string()
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        let pid: u32 = if protocol == "TCP" {
            if parts.len() > 4 {
                parts[4].parse().unwrap_or(0)
            } else {
                0
            }
        } else {
            // UDP: PID 是第4列
            if parts.len() > 3 {
                parts[3].parse().unwrap_or(0)
            } else {
                0
            }
        };

        ports.push(PortInfo {
            protocol,
            local_ip: local.0,
            local_port: local.1,
            remote_ip: remote.0,
            remote_port: remote.1,
            state,
            pid,
            process_name: String::new(),
        });
    }

    Ok(ports)
}

/// 解析地址字符串 "0.0.0.0:135" 或 "[::]:135" 为 (ip, port)
///
/// 特殊值 `*:*` 表示任意地址任意端口（常见于 UDP 行），返回("*", 0)。
fn parse_address(addr: &str) -> Result<(String, u16), String> {
    if addr == "*:*" {
        return Ok(("*".to_string(), 0));
    }

    // 处理 IPv6 地址 [::]:port
    if addr.starts_with('[') {
        if let Some(bracket_end) = addr.find(']') {
            let ip = addr[..=bracket_end].to_string();
            let port_str = &addr[bracket_end + 2..]; // 跳过 "]:"
            let port: u16 = port_str
                .parse()
                .map_err(|_| format!("无法解析端口: {port_str}"))?;
            return Ok((ip, port));
        }
    }

    // 处理 IPv4 地址 ip:port
    if let Some(colon_pos) = addr.rfind(':') {
        let ip = addr[..colon_pos].to_string();
        let port_str = &addr[colon_pos + 1..];
        let port: u16 = port_str
            .parse()
            .map_err(|_| format!("无法解析端口: {port_str}"))?;
        return Ok((ip, port));
    }

    Err(format!("无法解析地址: {addr}"))
}

/// 获取 PID 到进程名的映射（通过 tasklist 命令）
fn get_process_names() -> HashMap<u32, String> {
    let mut map = HashMap::new();

    let output = match std::process::Command::new("tasklist")
        .args(["/FO", "CSV", "/NH"])
        .output()
    {
        Ok(o) => o,
        Err(_) => return map,
    };

    let stdout = match String::from_utf8(output.stdout) {
        Ok(s) => s,
        Err(_) => return map,
    };

    // tasklist CSV 格式: "映像名称","PID","会话名","会话#","内存使用"
    for line in stdout.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() < 2 {
            continue;
        }
        // 去掉引号
        let name = parts[0].trim_matches('"').to_string();
        let pid_str = parts[1].trim_matches('"');
        if let Ok(pid) = pid_str.parse::<u32>() {
            map.insert(pid, name);
        }
    }

    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_address_ipv4() {
        let (ip, port) = parse_address("0.0.0.0:135").unwrap();
        assert_eq!(ip, "0.0.0.0");
        assert_eq!(port, 135);
    }

    #[test]
    fn test_parse_address_ipv6() {
        let (ip, port) = parse_address("[::]:135").unwrap();
        assert_eq!(ip, "[::]");
        assert_eq!(port, 135);
    }

    #[test]
    fn test_parse_netstat_output() {
        let sample = "\
Proto  Local Address          Foreign Address        State           PID
TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1056
TCP    0.0.0.0:445            0.0.0.0:0              LISTENING       4
TCP    127.0.0.1:3306         0.0.0.0:0              LISTENING       1234
UDP    0.0.0.0:500            *:*                                    1234
";
        let ports = parse_netstat_output(sample).unwrap();
        assert_eq!(ports.len(), 4);
        assert_eq!(ports[0].protocol, "TCP");
        assert_eq!(ports[0].local_port, 135);
        assert_eq!(ports[0].state, "LISTENING");
        assert_eq!(ports[0].pid, 1056);
        assert_eq!(ports[3].protocol, "UDP");
        assert_eq!(ports[3].local_port, 500);
    }
}