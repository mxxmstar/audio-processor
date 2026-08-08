# audio-processor 音频识别调试记录

本文档记录了从零搭建音频指纹识别工具 `recognizer.py` 过程中遇到的所有问题、根因分析与最终解决方案。
技术栈：Python + `pyacoustid`(acoustid) + `musicbrainzngs`，指纹计算依赖 Chromaprint 的 `fpcalc.exe`。

---

## 0. 环境准备

- 项目目录：`k:/code_mx/audio-processor`
- 音频指纹工具：`bin/fpcalc.exe`（Chromaprint，已置于项目 `bin/` 目录）
- 目标：用 `acoustid` 库对音频做指纹识别，并通过 MusicBrainz 查询歌曲信息

### 0.1 创建独立虚拟环境（避免污染根 Python）

```powershell
cd k:/code_mx/audio-processor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

> 注意：`K:` 盘为 reparse point（符号链接式挂载），在此盘创建 `.venv` 时 `venv` 自带的 safe-delete 清理步骤会失败，
> 但虚拟环境本身已可用，只需手动补建根目录下的 `pyvenv.cfg` 即可（本机基于 Python 3.10.11）。

### 0.2 安装依赖

```powershell
pip install pyacoustid musicbrainzngs
pip freeze > requirements.txt
```

> **坑 1（包名陷阱）**：脚本里写的是 `import acoustid`，但 PyPI 上的发行包名是 `pyacoustid`。
> 若直接 `pip install acoustid` 会报 `Could not find a version that satisfies the requirement acoustid (from versions: none)`，
> 这是**包名错误**，不是网络问题（可用 `pip install requests` 验证 PyPI 网络是否通畅）。

最终 `requirements.txt` 内容（节选）：

```
pyacoustid==1.3.1
musicbrainzngs==0.7.1
audioread==3.1.0
requests==2.34.2
...
```

---

## 1. 问题一：basedpyright 报 "无法解析导入 acoustid"

**现象**：编辑器红线提示 `无法解析导入 "acoustid"`，来源 `basedpyright`。
**根因**：acoustid 尚未安装到运行环境（纯静态检查器报错，非代码逻辑错误）。
**解决**：参见 0.2 安装步骤。装完后该报错消失。

> 剩余的 `reportUnknownMemberType` / `reportUnknownVariableType` 等 WARNING 是类型检查器对无 stub 文件的
> 第三方库（pyacoustid / musicbrainzngs）的误报，**不影响实际运行**。

---

## 2. 问题二：fpcalc 路径配置无效

**最初写法（错误）**：

```python
acoustid.fpcalc = r"bin\fpcalc.exe"   # 也曾尝试 acoustid.fpcalc = r"C:\Tools\fpcalc\fpcalc.exe"
```

**现象**：运行时报 `acoustid.NoBackendError: fpcalc not found`。
**排查**：阅读 `pyacoustid` 源码（`acoustid.py`）发现：

```python
44:  FPCALC_COMMAND = "fpcalc"
309: fpcalc = os.environ.get(FPCALC_ENVVAR, FPCALC_COMMAND)   # FPCALC_ENVVAR = "FPCALC"
```

`acoustid.fpcalc` 这个**模块属性在代码里根本没有被读取**，`pyacoustid` 实际只通过**环境变量 `FPCALC`** 定位可执行文件。
**解决**：改用环境变量设置（基于脚本目录，保证任意 CWD 都能定位）：

```python
os.environ["FPCALC"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "fpcalc.exe")
```

> 关于相对路径：Python `subprocess` 支持相对路径（基于 CWD 解析），但为稳妥应使用 `__file__` 构造绝对路径，
> 避免"在别的目录运行脚本"导致找不到。

---

## 3. 问题三：中文/日文文件名导致 fpcalc 无法打开文件

**现象**：即使 fpcalc 路径正确，仍报 `fpcalc exited with status 2` / `Could not open the input file`。

**分步定位过程**：

1. 直接用 `fpcalc.exe` 打开该中文名 mp3 → 同样 `status 2`（No such file or directory）。
2. 把该 mp3 复制成 ASCII 文件名 `test_ascii.mp3` 再调用 → **成功**（rc=0，返回 FINGERPRINT）。
3. 改用 8.3 短路径方案 → 失败（`K:` 卷是 reparse point，短路径 API 返回原中文路径，不可用）。
4. 将临时副本放在**系统 TEMP 目录**（含中文用户名）→ 仍失败。
5. 将临时副本放在**项目目录** `k:/code_mx/audio-processor`（纯 ASCII 路径）→ **成功**。

**真正的根因（两层）**：

- **A. fpcalc 是 C++ 程序**，通过子进程接收命令行参数时，含非 ASCII 字符（中文/日文）的路径会乱码，导致打不开文件。
  解决方案：识别前先把音频复制到一个 ASCII 命名的临时文件，再交给 fpcalc，识别后删除。
- **B. PowerShell 传参截断**：在 PowerShell 中把 `01. 風の声を聴きながら.mp3` 作为命令行参数传给 Python 时，
  非 ASCII 部分被丢弃，Python 收到的 `sys.argv[1]` 实际是 `'01.'`（纯 ASCII）。
  这导致后来加的 `if not file_path.isascii():` 判断为 **False**，跳过了复制逻辑，直接把 `'01.'` 喂给 fpcalc → 失败。

**解决（两层都修）**：

- 新增 `_resolve_path()`：当传入路径不存在时，用其作为**前缀**在目录中模糊匹配真实文件名
  （例如 `'01.'` 匹配到 `'01. 風の声を聴きながら.mp3'`）。
- 复制逻辑改为：`if not file_path.isascii():` 时，用 `tempfile.mkstemp(dir=脚本目录)` 在 **ASCII 路径的脚本目录**下
  生成临时副本（避免系统 TEMP 含中文用户名），识别完毕在 `finally` 中清理。

```python
def _resolve_path(file_path):
    if os.path.exists(file_path):
        return file_path
    search_dir = os.path.dirname(file_path) or "."
    prefix = os.path.basename(file_path)
    name_no_ext, ext = os.path.splitext(prefix)
    matches = []
    for entry in os.listdir(search_dir):
        if entry.startswith(prefix) or (
            name_no_ext and entry.startswith(name_no_ext) and entry.lower().endswith(ext.lower())
        ):
            matches.append(os.path.join(search_dir, entry))
    return matches[0] if matches else file_path
```

```python
if not file_path.isascii():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fd, tmp_path = tempfile.mkstemp(
        suffix=os.path.splitext(file_path)[1], prefix="tmp_audio_", dir=base_dir
    )
    os.close(fd)
    shutil.copy(file_path, tmp_path)
    work_path = tmp_path
```

---

## 4. 问题四：`results` 是生成器，不能下标访问

**现象**：`TypeError: 'generator' object is not subscriptable`（或 `if not results:` 始终为 False）。
**根因**：`acoustid.match()` 返回的是**生成器**（generator），不是 list。
**解决**：消费前转成 list：

```python
results = list(results)
if not results:
    print("未识别到任何歌曲")
    return
best_score, recording_id, title, artist = results[0]
```

---

## 5. 问题五：AcoustID Web 服务返回 `invalid API key`

**现象**：指纹生成成功后，查询阶段报 `acoustid.WebServiceError: status: error`。
**排查**：直接打印 `acoustid.lookup()` 原始响应，服务端返回：

```python
{'error': {'code': 4, 'message': 'invalid API key'}, 'status': 'error'}
```

**根因**：`ACOUSTID_KEY` 使用了无效的密钥（误用了 User API key / 占位 key）。
**解决**：从 https://acoustid.org/api-key 获取正确的 **Application API key**（用于查询，参数名 `client`），
替换 `recognizer.py` 第 9 行的 `ACOUSTID_KEY`。

> 注意区分两种 key：
> - **Application key**：用于查询指纹（`client` 参数），本项目使用。
> - **User API key**：用于提交指纹，本项目不需要。

---

## 6. 最终验证结果

```powershell
.\.venv\Scripts\Activate.ps1
python recognizer.py "01. 風の声を聴きながら.mp3"
```

输出：

```
识别结果 (置信度: 100.0%)
标题: 風の声を聴きながら
艺术家: 三月のパンタシア
艺术家(MB): 三月のパンタシア
专辑: 風の声を聴きながら (2018-01-21)
```

完整链路验证通过：venv 依赖 → FPCALC 环境变量 → 中文文件名模糊匹配 + ASCII 临时副本 → fpcalc 生成指纹
→ AcoustID 查询 → MusicBrainz 补充专辑信息。

---

## 7. 快速排错速查表

| 报错 | 根因 | 解决 |
|------|------|------|
| 无法解析导入 "acoustid" | 未安装依赖 | `pip install pyacoustid` |
| `pip install acoustid` 报 from versions: none | 包名错误，应为 `pyacoustid` | 改用 `pyacoustid` |
| `NoBackendError: fpcalc not found` | 设了 `acoustid.fpcalc` 无效 | 改为设环境变量 `FPCALC` |
| `fpcalc exited with status 2` | 中文路径乱码 / PowerShell 截断参数 | `_resolve_path` 模糊匹配 + ASCII 临时副本 |
| `'generator' object is not subscriptable` | `match()` 返回生成器 | `list(results)` |
| `WebServiceError: status: error` (code 4) | API key 无效 | 换正确的 Application API key |

---

## 8. 运行方式

```powershell
cd k:/code_mx/audio-processor
.\.venv\Scripts\Activate.ps1
python recognizer.py <音频文件路径>
```

> 提示：若文件名含中文/日文，直接用 PowerShell 传参会被截断，脚本已内置模糊匹配兜底；
> 为获得最稳定体验，可传入完整绝对路径，或拖拽文件到终端。

### 8.1 关键踩坑：必须先激活 venv 再运行

**现象**：跳过激活直接运行脚本，报：

```text
ModuleNotFoundError: No module named 'acoustid'
```

**根因**：`acoustid` / `musicbrainzngs` 等依赖只装在了项目的 `.venv` 里。不激活 venv 时，`python` 指向的是系统（根）Python，
那里并没有这些包，于是找不到模块。激活后命令行提示符前会出现 `(.venv)` 前缀，表示已切换到项目环境。

**正确顺序（务必先激活）**：

```powershell
cd k:/code_mx/audio-processor
.\.venv\Scripts\Activate.ps1        # 激活后提示符前缀变为 (.venv)
python .\recognizer.py "K:\code_mx\audio-processor\01. 風の声を聴きながら.mp3"
```

> 激活后建议用**绝对路径**传参（如上面示例），避免 PowerShell 对中文/日文文件名做参数截断（见 问题三-B）。

### 8.2 激活失败：PowerShell 执行策略拦截

若执行 `Activate.ps1` 时报类似 `无法加载文件 ... 因为在此系统上禁止运行脚本` 的错，是 PowerShell 默认执行策略（`Restricted`）阻止了脚本运行。

**解决**：为当前用户放宽策略（只需执行一次）：

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

设置完成后重新运行 `.\.venv\Scripts\Activate.ps1` 即可激活。

### 8.3 复制粘贴命令的注意事项

把多行命令从聊天窗口复制到 PowerShell 时，偶尔会出现两行被拼成一行、中间丢失换行符的情况（例如
`...mp3"python .\recognizer.py...`），这会导致命令解析异常并仍然报 `ModuleNotFoundError`。

**解决**：复制后确认每条命令单独成行，或逐行手动输入，确保 `Activate.ps1` 单独执行成功（看到 `(.venv)` 前缀）后再运行
`python` 命令。
