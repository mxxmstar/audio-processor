<script lang="ts">
  import { invoke } from '@tauri-apps/api/core';
  import { listen } from '@tauri-apps/api/event';
  import { open } from '@tauri-apps/plugin-dialog';
  import { onMount } from 'svelte';

  type TaskStatus = 'Pending' | 'Downloading' | 'Completed' | 'Failed';

  interface Task {
    id: string;
    title: string;
    bvid: string;
    page: number;
    part: string;
    mode: string;
    out_path: string;
    status: TaskStatus;
    error: string | null;
  }

  interface ProgressEvent {
    task_id: string;
    title: string;
    status: string;
    percent: number;
    downloaded: number;
    total: number;
    speed: number;
    error: string | null;
  }

  interface LoginQr {
    qr_svg: string;
    qr_key: string;
  }

  interface LoginState {
    authed: boolean;
    message: string;
  }

  // ---- 状态 ----
  let loggedIn = $state(false);
  let checkingLogin = $state(true);

  let inputUrl = $state('');
  let preferFormat = $state('1080P');
  let mode = $state('audio'); // audio | video | merge
  let outputDir = $state('');

  let tasks = $state<Task[]>([]);
  let resolving = $state(false);
  let downloading = $state(false);
  let message = $state('');

  // 登录二维码弹窗
  let showQr = $state(false);
  let qr = $state<LoginQr | null>(null);
  let qrPolling = $state<number | null>(null);

  const progressMap = $state<Record<string, ProgressEvent>>({});

  function fmtBytes(n: number): string {
    if (!n) return '0 B';
    const u = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(n) / Math.log(1024));
    return (n / Math.pow(1024, i)).toFixed(2) + ' ' + u[i];
  }

  async function pickDir() {
    try {
      const sel = await open({
        directory: true,
        multiple: false,
        title: '选择下载目录',
      });
      if (typeof sel === 'string') {
        outputDir = sel;
      }
    } catch (e) {
      message = '选择目录失败：' + String(e);
    }
  }

  async function refreshLogin() {
    checkingLogin = true;
    try {
      loggedIn = await invoke<boolean>('bili_check_login');
    } catch {
      loggedIn = false;
    } finally {
      checkingLogin = false;
    }
  }

  async function doResolve() {
    if (!loggedIn) {
      message = '请先扫码登录';
      return;
    }
    if (!inputUrl.trim()) {
      message = '请输入 BV 号或链接';
      return;
    }
    resolving = true;
    message = '';
    try {
      tasks = await invoke<Task[]>('bili_resolve', {
        input: {
          input: inputUrl.trim(),
          mode,
          preferFormat,
          outputDir: outputDir || null,
        },
      });
    } catch (e) {
      message = String(e);
      tasks = [];
    } finally {
      resolving = false;
    }
  }

  async function doDownload() {
    if (tasks.length === 0) return;
    downloading = true;
    message = '开始下载…';
    try {
      await invoke<string[]>('bili_start_download', {
        input: { outputDir: outputDir || null, concurrency: 3 },
      });
    } catch (e) {
      message = String(e);
      downloading = false;
    }
  }

  async function doLogout() {
    await invoke('bili_logout');
    loggedIn = false;
    tasks = [];
  }

  async function openQr() {
    showQr = true;
    try {
      qr = await invoke<LoginQr>('bili_login_qr');
      startPoll();
    } catch (e) {
      message = '生成二维码失败：' + String(e);
    }
  }

  function startPoll() {
    stopPoll();
    qrPolling = setInterval(async () => {
      if (!qr) return;
      try {
        const r = await invoke<LoginState>('bili_login_poll', { qrKey: qr.qr_key });
        if (r.authed) {
          loggedIn = true;
          showQr = false;
          stopPoll();
          qr = null;
          message = '登录成功';
        }
      } catch {
        /* 继续轮询 */
      }
    }, 2000) as unknown as number;
  }

  function stopPoll() {
    if (qrPolling !== null) {
      clearInterval(qrPolling);
      qrPolling = null;
    }
  }

  onMount(() => {
    refreshLogin();
    const off1 = listen<ProgressEvent>('download-progress', (e) => {
      progressMap[e.payload.task_id] = e.payload;
    });
    const off2 = listen<{ ok: boolean; failed: number }>('download-finished', (e) => {
      downloading = false;
      message = e.payload.ok ? '全部下载完成' : `下载结束，${e.payload.failed} 个失败`;
      invoke<Task[]>('bili_list_tasks')
        .then((t) => (tasks = t))
        .catch(() => {});
    });
    return async () => {
      stopPoll();
      (await off1)();
      (await off2)();
    };
  });
</script>

<div class="panel">
  {#if checkingLogin}
    <p class="hint">检查登录态…</p>
  {:else if !loggedIn}
    <div class="login-box">
      <p>当前未登录，部分功能受限。</p>
      <button onclick={openQr}>扫码登录</button>
    </div>
  {:else}
    <div class="bar">
      <span class="ok">已登录</span>
      <button class="ghost" onclick={doLogout}>登出</button>
    </div>
  {/if}

  {#if showQr && qr}
    <div class="qr">
      <p>使用 B站 APP 扫码登录</p>
      <img src={qr.qr_svg} alt="qr" width="240" height="240" />
    </div>
  {/if}

  {#if loggedIn}
    <section class="input">
      <input
        placeholder="BV 号 / 链接 / av 号 / 合集(ss) / 番剧"
        bind:value={inputUrl}
      />
      <select bind:value={mode}>
        <option value="audio">仅音频</option>
        <option value="video">仅视频</option>
        <option value="merge">音视频合并</option>
      </select>
      <select bind:value={preferFormat}>
        <option>360P</option>
        <option>720P</option>
        <option>1080P</option>
        <option>4K</option>
        <option>8K</option>
      </select>
      <button onclick={pickDir}>选择目录</button>
      <span class="dir" title={outputDir}>
        {outputDir || '默认：应用配置目录'}
      </span>
      <button onclick={doResolve} disabled={resolving}>
        {resolving ? '解析中…' : '解析'}
      </button>
      <button onclick={doDownload} disabled={downloading || tasks.length === 0}>
        {downloading ? '下载中…' : '开始下载'}
      </button>
    </section>

    {#if message}<p class="msg">{message}</p>{/if}

    {#if tasks.length}
      <ul class="tasks">
        {#each tasks as t (t.id)}
          {@const pg = progressMap[t.id]}
          <li>
            <div class="t-title">{t.title}{t.part ? ' - ' + t.part : ''}</div>
            <div class="t-meta">
              {t.mode} · {t.status}
              {#if pg}
                · {fmtBytes(pg.downloaded)}{pg.total ? ' / ' + fmtBytes(pg.total) : ''}
                · {fmtBytes(pg.speed)}/s
              {/if}
              {#if t.error}<span class="err"> · {t.error}</span>{/if}
            </div>
            <div class="progress">
              <div
                class="fill"
                style="width: {pg ? Math.round(pg.percent * 100) : (t.status === 'Completed' ? 100 : 0)}%"
              ></div>
            </div>
          </li>
        {/each}
      </ul>
    {/if}
  {/if}
</div>

<style>
  .panel {
    padding: 1.5rem;
    height: 100%;
    overflow-y: auto;
  }
  .hint {
    color: #6b7280;
  }
  .login-box {
    border: 1px dashed #cbd5e1;
    border-radius: 10px;
    padding: 1rem;
    background: #fff;
  }
  .bar {
    display: flex;
    gap: 0.5rem;
    align-items: center;
  }
  .ok {
    color: #1a7f37;
    font-weight: 600;
  }
  .input {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin: 1rem 0;
  }
  .input input,
  .input select {
    padding: 0.4rem;
  }
  .input input:first-child {
    flex: 1 1 240px;
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
  button.ghost {
    background: transparent;
    color: #2563eb;
    border: 1px solid #2563eb;
  }
  .msg {
    color: #b35900;
  }
  .qr {
    margin: 1rem 0;
  }
  .tasks {
    list-style: none;
    padding: 0;
  }
  .tasks li {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 0.6rem;
    margin-bottom: 0.6rem;
    background: #fff;
  }
  .t-title {
    font-weight: 600;
  }
  .t-meta {
    font-size: 0.8rem;
    color: #555;
    margin: 0.3rem 0;
  }
  .err {
    color: #c00;
  }
  .progress {
    height: 8px;
    background: #eee;
    border-radius: 4px;
    overflow: hidden;
  }
  .fill {
    height: 100%;
    background: #1a7f37;
    transition: width 0.2s;
  }
</style>
