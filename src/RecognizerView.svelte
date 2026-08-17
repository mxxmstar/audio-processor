<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import { open } from '@tauri-apps/plugin-dialog';

  interface SongInfo {
    title: string;
    artist: string;
    album: string | null;
    album_date: string | null;
    confidence: number;
  }

  let filePath = $state('');
  let result = $state<SongInfo | null>(null);
  let message = $state('');
  let identifying = $state(false);

  async function pickFile() {
    try {
      const sel = await open({
        multiple: false,
        title: '选择音频文件',
        filters: [
          { name: '音频', extensions: ['mp3', 'flac', 'm4a', 'wav', 'ogg', 'opus', 'aac'] },
        ],
      });
      if (typeof sel === 'string') {
        filePath = sel;
        result = null;
        message = '';
      }
    } catch (e) {
      message = '选择文件失败：' + String(e);
    }
  }

  async function doIdentify() {
    if (!filePath) {
      message = '请先选择音频文件';
      return;
    }
    identifying = true;
    message = '';
    result = null;
    try {
      const r = await invoke<SongInfo>('identify', { path: filePath });
      result = r;
    } catch (e) {
      message = '识别失败：' + String(e);
    } finally {
      identifying = false;
    }
  }
</script>

<div class="panel">
  <h2>音频识别</h2>
  <p class="desc">通过音频指纹（Chromaprint）匹配 AcoustID，识别歌曲的标题、艺术家与专辑信息。</p>

  <section class="input">
    <button onclick={pickFile}>选择音频文件</button>
    <span class="file" title={filePath}>{filePath || '未选择文件'}</span>
    <button onclick={doIdentify} disabled={identifying || !filePath}>
      {identifying ? '识别中…' : '开始识别'}
    </button>
  </section>

  {#if message}<p class="msg">{message}</p>{/if}

  {#if result}
    <div class="result">
      <div class="row">
        <span class="label">标题</span>
        <span class="value">{result.title}</span>
      </div>
      <div class="row">
        <span class="label">艺术家</span>
        <span class="value">{result.artist}</span>
      </div>
      <div class="row">
        <span class="label">专辑</span>
        <span class="value">{result.album ?? '—'}</span>
      </div>
      <div class="row">
        <span class="label">发行日期</span>
        <span class="value">{result.album_date ?? '—'}</span>
      </div>
      <div class="row">
        <span class="label">置信度</span>
        <span class="value">{result.confidence.toFixed(1)}%</span>
      </div>
    </div>
  {/if}
</div>

<style>
  .panel {
    padding: 1.5rem;
    height: 100%;
    overflow-y: auto;
  }
  h2 {
    margin-top: 0;
  }
  .desc {
    color: #6b7280;
    font-size: 0.9rem;
  }
  .input {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    align-items: center;
    margin: 1rem 0;
  }
  button {
    padding: 0.4rem 0.8rem;
    cursor: pointer;
    border: none;
    border-radius: 6px;
    background: #2563eb;
    color: #fff;
    font-weight: 600;
  }
  button:disabled {
    background: #9ca3af;
    cursor: not-allowed;
  }
  .file {
    max-width: 320px;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    color: #555;
    font-size: 0.85rem;
  }
  .msg {
    color: #b35900;
  }
  .result {
    margin-top: 1rem;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    background: #fff;
    padding: 1rem;
  }
  .row {
    display: flex;
    padding: 0.4rem 0;
    border-bottom: 1px solid #f1f5f9;
  }
  .row:last-child {
    border-bottom: none;
  }
  .label {
    width: 80px;
    color: #6b7280;
    font-size: 0.9rem;
  }
  .value {
    font-weight: 600;
  }
</style>
