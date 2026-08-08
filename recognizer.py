import acoustid
import musicbrainzngs
import sys
import os
import shutil
import tempfile

# === 配置区 ===
ACOUSTID_KEY = "rheYMWAd4x"
# pyacoustid 通过环境变量 FPCALC 定位 fpcalc 可执行文件
os.environ["FPCALC"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin", "fpcalc.exe")

musicbrainzngs.set_useragent("MySongRecognizer", "0.1")

def _resolve_path(file_path):
    """解析音频文件路径。

    PowerShell 在把含非 ASCII 字符(中文/日文)的文件名作为命令行参数
    传给外部进程时，经常会把非 ASCII 部分丢掉(得到如 '01.' 这样的残缺名)。
    因此当传入的路径不存在时，尝试用其作为前缀/包含子串，在所在目录中
    模糊匹配真实的文件名。
    """
    if os.path.exists(file_path):
        return file_path
    candidate = file_path
    # 去掉可能被截掉后缀的情况，先尝试用传入字符串作为前缀匹配
    search_dir = os.path.dirname(candidate) or "."
    prefix = os.path.basename(candidate)
    # 若含扩展名则用前缀(去掉扩展名后的部分)匹配
    name_no_ext, ext = os.path.splitext(prefix)
    matches = []
    try:
        for entry in os.listdir(search_dir):
            if entry.startswith(prefix):
                matches.append(os.path.join(search_dir, entry))
            elif name_no_ext and entry.startswith(name_no_ext) and entry.lower().endswith(ext.lower()):
                matches.append(os.path.join(search_dir, entry))
    except OSError:
        pass
    if matches:
        return matches[0]
    return file_path


def identify_song(file_path):
    file_path = _resolve_path(file_path)
    # fpcalc (C++ 程序) 在 Windows 下通过子进程接收命令行参数时，
    # 含非 ASCII 字符(如中文)的路径会乱码导致无法打开文件。
    # 因此先把音频复制到一个 ASCII 命名的临时文件再识别。
    tmp_path = None
    work_path = file_path
    if not file_path.isascii():
        # 临时副本放在脚本所在目录(ASCII路径)，避免系统 TEMP 含中文用户名导致 fpcalc 打不开
        base_dir = os.path.dirname(os.path.abspath(__file__))
        fd, tmp_path = tempfile.mkstemp(
            suffix=os.path.splitext(file_path)[1], prefix="tmp_audio_", dir=base_dir
        )
        os.close(fd)
        shutil.copy(file_path, tmp_path)
        work_path = tmp_path
    try:
        results = acoustid.match(ACOUSTID_KEY, work_path, force_fpcalc=True)
    except acoustid.NoBackendError:
        print("错误: 找不到 fpcalc，请检查路径配置")
        return
    except acoustid.AcoustidError as e:
        print(f"AcoustID 请求失败: {e}")
        return
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    results = list(results)
    if not results:
        print("未识别到任何歌曲")
        return

    best_score, recording_id, title, artist = results[0]
    confidence = best_score * 100

    print(f"识别结果 (置信度: {confidence:.1f}%)")
    print(f"标题: {title}")
    print(f"艺术家: {artist}")

    # 通过 MusicBrainz 获取专辑等额外信息
    try:
        rec = musicbrainzngs.get_recording_by_id(
            recording_id, includes=["artists", "releases"]
        )
        recording = rec["recording"]
        if "artist-credit" in recording:
            artists = [c["artist"]["name"] for c in recording["artist-credit"]]
            print(f"艺术家(MB): {', '.join(artists)}")
        if "release-list" in recording:
            rel = recording["release-list"][0]
            print(f"专辑: {rel['title']} ({rel.get('date','未知')})")
    except Exception as e:
        print(f"获取详细信息失败: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python recognizer.py <音频文件>")
    else:
        identify_song(sys.argv[1])