# Aria2 下载器集成实施计划

## 1. 项目概述

### 1.1 目标
在现有 Tauri + Vue 3 + Ant Design Vue 应用中集成 aria2 下载器功能，提供图形化界面管理下载任务。

### 1.2 功能需求
- 在左侧菜单栏添加 "aria2 下载" 和 "下载历史" 菜单项
- 封装 aria2 功能，提供图形化界面
- 支持 HTTP/HTTPS、FTP、BT、磁力链接等下载
- 任务管理：添加、暂停、恢复、删除、重试
- 下载历史记录查看
- 实时进度显示

### 1.3 技术栈
- **前端**: Vue 3 + TypeScript + Ant Design Vue
- **后端**: Tauri (Rust)
- **下载引擎**: aria2 1.37.0 (C:\Program Files\aria2-1.37.0-win-64bit-build1)

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────┐
│         Vue 3 前端界面                    │
│  ┌──────────────┐  ┌──────────────┐    │
│  │ Aria2View    │  │ HistoryView  │    │
│  │ (下载界面)    │  │ (历史记录)    │    │
│  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────┘
              ↓ invoke
┌─────────────────────────────────────────┐
│      Tauri 后端 (Rust)                  │
│  ┌────────────────────────────────┐    │
│  │ aria2 命令模块                  │    │
│  │ - aria2_start                  │    │
│  │ - aria2_pause                  │    │
│  │ - aria2_remove                 │    │
│  │ - aria2_get_status             │    │
│  │ - aria2_list_tasks             │    │
│  └────────────────────────────────┘    │
└─────────────────────────────────────────┘
              ↓ 进程调用
┌─────────────────────────────────────────┐
│      aria2 RPC 服务                     │
│  - JSON-RPC 接口                        │
│  - WebSocket 实时通信                   │
└─────────────────────────────────────────┘
```

### 2.2 aria2 运行模式

采用 **aria2 RPC 模式**：
- 启动 aria2 作为后台服务，监听 JSON-RPC 端口
- Tauri 后端通过 HTTP/WebSocket 与 aria2 通信
- 前端通过 Tauri invoke 调用后端命令

---

## 3. 文件结构规划

### 3.1 前端文件

```
src/
├── Aria2View.vue              # aria2 下载主界面
├── Aria2HistoryView.vue       # aria2 下载历史界面（可复用 HistoryView）
└── App.vue                    # 更新菜单配置
```

### 3.2 后端文件

```
src-tauri/src/
├── aria2/
│   ├── mod.rs                 # aria2 模块入口
│   ├── client.rs              # aria2 RPC 客户端
│   ├── types.rs               # 数据类型定义
│   └── manager.rs             # 任务管理器
├── commands/
│   └── aria2.rs               # aria2 相关 Tauri 命令
└── lib.rs                     # 注册命令
```

### 3.3 aria2 可执行文件

```
bin/
└── aria2/
    ├── aria2c.exe             # 从 C 盘复制
    └── aria2.conf             # aria2 配置文件
```

---

## 4. 详细实现方案

### 4.1 aria2 部署与配置

#### 4.1.1 复制 aria2 到项目
- 将 `C:\Program Files\aria2-1.37.0-win-64bit-build1\aria2c.exe` 复制到 `bin/aria2/`
- 创建 aria2 配置文件 `bin/aria2/aria2.conf`

#### 4.1.2 aria2 配置
```conf
# RPC 配置
enable-rpc=true
rpc-listen-port=6800
rpc-secret=your_secret_token
rpc-allow-origin-all=true

# 下载配置
dir=./downloads
max-concurrent-downloads=5
max-connection-per-server=16
min-split-size=1M
split=16

# BT 配置
enable-dht=true
enable-peer-exchange=true

# 日志
log=aria2.log
log-level=info
```

### 4.2 后端实现 (Rust)

#### 4.2.1 数据类型定义 (types.rs)

```rust
// 任务状态
pub enum Aria2TaskStatus {
    Active,
    Waiting,
    Paused,
    Complete,
    Error,
    Removed,
}

// 下载任务
pub struct Aria2Task {
    pub gid: String,           // aria2 任务 ID
    pub name: String,          // 文件名
    pub url: String,           // 下载链接
    pub status: Aria2TaskStatus,
    pub total_length: u64,     // 总大小
    pub completed_length: u64, // 已完成大小
    pub download_speed: u64,   // 下载速度
    pub upload_speed: u64,     // 上传速度
    pub progress: f64,         // 进度百分比
    pub error_message: Option<String>,
    pub created_at: i64,       // 创建时间戳
    pub output_dir: String,    // 输出目录
}

// 添加任务请求
pub struct AddTaskRequest {
    pub urls: Vec<String>,
    pub output_dir: Option<String>,
    pub file_name: Option<String>,
    pub options: Option<HashMap<String, String>>,
}
```

#### 4.2.2 aria2 RPC 客户端 (client.rs)

实现 aria2 JSON-RPC 通信：
- `start_service()`: 启动 aria2 服务进程
- `stop_service()`: 停止 aria2 服务
- `add_uri()`: 添加 HTTP/FTP 下载
- `add_torrent()`: 添加 BT 种子
- `pause()`: 暂停任务
- `resume()`: 恢复任务
- `remove()`: 删除任务
- `get_status()`: 获取任务状态
- `tell_active()`: 获取所有活跃任务
- `tellWaiting()`: 获取等待中的任务
- `tellStopped()`: 获取已停止的任务

#### 4.2.3 任务管理器 (manager.rs)

- 维护任务列表缓存
- 定期轮询 aria2 状态（每秒）
- 通过 Tauri 事件系统推送进度更新
- 持久化下载历史记录到本地数据库/文件

#### 4.2.4 Tauri 命令 (commands/aria2.rs)

```rust
#[tauri::command]
pub async fn aria2_start_service() -> Result<(), String>

#[tauri::command]
pub async fn aria2_stop_service() -> Result<(), String>

#[tauri::command]
pub async fn aria2_add_task(req: AddTaskRequest) -> Result<String, String>

#[tauri::command]
pub async fn aria2_pause_task(gid: String) -> Result<(), String>

#[tauri::command]
pub async fn aria2_resume_task(gid: String) -> Result<(), String>

#[tauri::command]
pub async fn aria2_remove_task(gid: String) -> Result<(), String>

#[tauri::command]
pub async fn aria2_list_tasks() -> Result<Vec<Aria2Task>, String>

#[tauri::command]
pub async fn aria2_get_history() -> Result<Vec<Aria2Task>, String>
```

### 4.3 前端实现 (Vue)

#### 4.3.1 Aria2View.vue 界面布局

```
┌─────────────────────────────────────────┐
│  aria2 下载器                            │
├─────────────────────────────────────────┤
│  下载链接输入框                          │
│  [支持 HTTP/HTTPS/FTP/BT/磁力链接]      │
├─────────────────────────────────────────┤
│  输出目录: [选择目录]  默认: ./downloads │
├─────────────────────────────────────────┤
│  [添加下载] [全部暂停] [全部继续]        │
├─────────────────────────────────────────┤
│  下载任务列表                            │
│  ┌─────────────────────────────────┐   │
│  │ 文件名.mp4                      │   │
│  │ ████████░░░░ 65%  2.3MB/s      │   │
│  │ [暂停] [删除] [打开文件夹]      │   │
│  └─────────────────────────────────┘   │
│  ┌─────────────────────────────────┐   │
│  │ 文件名2.mp3                     │   │
│  │ ████████████ 100%  已完成       │   │
│  │ [删除] [打开文件夹]             │   │
│  └─────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

#### 4.3.2 主要功能

1. **添加下载**
   - 输入框支持多个链接（换行分隔）
   - 选择输出目录
   - 点击添加按钮

2. **任务管理**
   - 实时显示下载进度、速度
   - 单个任务：暂停/恢复/删除
   - 批量操作：全部暂停/全部继续

3. **状态显示**
   - 使用 Ant Design 的 Progress 组件显示进度
   - 使用 Tag 组件显示状态
   - 实时更新下载速度

4. **历史记录**
   - 复用现有 HistoryView 组件
   - 显示已完成的下载记录

#### 4.3.3 实时更新机制

- 使用 Tauri 事件监听 `aria2-progress-update`
- 每秒接收后端推送的任务状态更新
- 使用 Vue 的响应式系统自动更新界面

### 4.4 菜单更新 (App.vue)

在左侧菜单添加：

```typescript
const items = [
  {
    key: "download-group",
    icon: h(DownloadOutlined),
    label: "B站下载",
    children: [
      { key: "download", icon: h(DownloadOutlined), label: "下载" },
      { key: "download-history", icon: h(HistoryOutlined), label: "历史记录" },
    ],
  },
  {
    key: "aria2-group",
    icon: h(CloudDownloadOutlined),
    label: "aria2 下载",
    children: [
      { key: "aria2-download", icon: h(CloudDownloadOutlined), label: "下载" },
      { key: "aria2-history", icon: h(HistoryOutlined), label: "历史记录" },
    ],
  },
  {
    key: "recognize-group",
    icon: h(AudioOutlined),
    label: "音频识别",
    children: [
      { key: "recognize", icon: h(AudioOutlined), label: "识别" },
      { key: "history", icon: h(HistoryOutlined), label: "历史记录" },
    ],
  },
];
```

---

## 5. 实施步骤

### 阶段 1: 环境准备 (Day 1)
- [ ] 复制 aria2c.exe 到 `bin/aria2/`
- [ ] 创建 aria2 配置文件
- [ ] 测试 aria2 命令行启动

### 阶段 2: 后端开发 (Day 2-3)
- [ ] 创建 `src-tauri/src/aria2/` 模块结构
- [ ] 实现 aria2 RPC 客户端 (client.rs)
- [ ] 实现任务管理器 (manager.rs)
- [ ] 实现 Tauri 命令 (commands/aria2.rs)
- [ ] 在 lib.rs 中注册命令

### 阶段 3: 前端开发 (Day 4-5)
- [ ] 创建 Aria2View.vue 组件
- [ ] 实现下载链接输入和目录选择
- [ ] 实现任务列表展示
- [ ] 实现实时更新监听
- [ ] 实现任务操作按钮

### 阶段 4: 集成测试 (Day 6)
- [ ] 更新 App.vue 菜单
- [ ] 测试 aria2 服务启动/停止
- [ ] 测试添加下载任务
- [ ] 测试暂停/恢复/删除功能
- [ ] 测试实时进度更新

### 阶段 5: 历史记录 (Day 7)
- [ ] 实现下载历史持久化
- [ ] 创建 Aria2HistoryView 或复用 HistoryView
- [ ] 测试历史记录查询

---

## 6. 技术要点

### 6.1 aria2 进程管理
- 使用 Tauri 的 `Command` API 启动 aria2 进程
- 监听进程退出，自动重启
- 应用退出时优雅关闭 aria2

### 6.2 RPC 通信
- 使用 `reqwest` 库发送 HTTP JSON-RPC 请求
- 可选：使用 WebSocket 实现实时推送

### 6.3 数据持久化
- 下载历史记录存储到 SQLite 或 JSON 文件
- 存储位置：Tauri 应用数据目录

### 6.4 错误处理
- aria2 启动失败处理
- RPC 连接失败重试
- 下载失败错误提示

---

## 7. 风险与应对

| 风险 | 影响 | 应对措施 |
|------|------|----------|
| aria2 进程启动失败 | 无法下载 | 检查端口占用，提示用户 |
| RPC 通信超时 | 状态不更新 | 增加重试机制，超时提示 |
| 下载路径权限问题 | 下载失败 | 提供目录选择，检查权限 |
| BT/磁力链接无种子 | 下载失败 | 提示用户检查链接有效性 |

---

## 8. 后续优化

- [ ] 支持批量导入链接（从文件）
- [ ] 支持下载优先级调整
- [ ] 支持下载速度限制设置
- [ ] 支持下载完成后通知
- [ ] 支持自动分类下载文件

---

## 9. 验收标准

1. ✅ 左侧菜单显示 "aria2 下载" 和 "历史记录"
2. ✅ 可以添加 HTTP/HTTPS/FTP 下载任务
3. ✅ 可以添加 BT/磁力链接下载
4. ✅ 实时显示下载进度和速度
5. ✅ 可以暂停/恢复/删除下载任务
6. ✅ 下载历史记录可查看
7. ✅ aria2 服务随应用启动/停止

---

**文档版本**: v1.0  
**创建日期**: 2026-08-29  
**最后更新**: 2026-08-29
