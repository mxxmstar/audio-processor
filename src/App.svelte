<script lang="ts">
  import { invoke } from "@tauri-apps/api/core";
  import { open } from "@tauri-apps/plugin-dialog";

  interface SongInfo {
    title: string;
    artist: string;
    album: string | null;
    album_date: string | null;
    confidence: number;
  }

  let info: SongInfo | null = null;
  let errorMsg: string | null = null;
  let loading = false;

  async function pickAndIdentify() {
    const selected = await open({
      multiple: false,
      filters: [
        {
          name: "音频文件",
          extensions: ["mp3", "flac", "wav", "m4a", "aac"],
        },
      ],
    });
    if (!selected || Array.isArray(selected)) return;
    await runIdentify(selected as string);
  }

  async function runIdentify(path: string) {
    loading = true;
    errorMsg = null;
    info = null;
    try {
      info = await invoke<SongInfo>("identify", { path });
    } catch (e) {
      errorMsg = String(e);
    } finally {
      loading = false;
    }
  }
</script>

<main>
  <h1>音频指纹识别</h1>
  <p class="sub">选择本地音频文件，识别曲目信息（基于 AcoustID + MusicBrainz）</p>

  <button on:click={pickAndIdentify} disabled={loading}>
    {loading ? "识别中…" : "选择音频文件"}
  </button>

  {#if errorMsg}
    <div class="card error">
      <strong>识别失败</strong>
      <p>{errorMsg}</p>
    </div>
  {:else if info}
    <div class="card">
      <div class="row"><span class="label">标题</span><span>{info.title}</span></div>
      <div class="row"><span class="label">艺术家</span><span>{info.artist}</span></div>
      {#if info.album}
        <div class="row">
          <span class="label">专辑</span>
          <span>{info.album}{info.album_date ? ` (${info.album_date})` : ""}</span>
        </div>
      {/if}
      <div class="row">
        <span class="label">置信度</span>
        <span>{info.confidence.toFixed(1)}%</span>
      </div>
    </div>
  {/if}
</main>

<style>
  main {
    max-width: 560px;
    margin: 0 auto;
    padding: 32px 24px;
  }
  h1 {
    margin: 0 0 4px;
    font-size: 24px;
  }
  .sub {
    margin: 0 0 24px;
    color: #6b7280;
    font-size: 14px;
  }
  button {
    background: #2563eb;
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 10px 18px;
    font-size: 15px;
    cursor: pointer;
  }
  button:disabled {
    background: #9ca3af;
    cursor: default;
  }
  .card {
    margin-top: 24px;
    background: #fff;
    border-radius: 10px;
    padding: 18px 20px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
  }
  .card.error {
    border: 1px solid #fca5a5;
    color: #b91c1c;
  }
  .row {
    display: flex;
    padding: 6px 0;
    border-bottom: 1px solid #f1f3f5;
  }
  .row:last-child {
    border-bottom: none;
  }
  .label {
    width: 80px;
    color: #6b7280;
    flex-shrink: 0;
  }
</style>
