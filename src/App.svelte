<script lang="ts">
  import DownloadView from './DownloadView.svelte';
  import RecognizerView from './RecognizerView.svelte';

  // 当前选中的菜单项：download | recognize
  let active = $state<'download' | 'recognize'>('download');
</script>

<div class="layout">
  <aside class="sidebar">
    <div class="brand">
      <span class="logo">♪</span>
      <span class="brand-name">Audio Processor</span>
    </div>
    <nav>
      <button
        class="nav-item"
        class:active={active === 'download'}
        onclick={() => (active = 'download')}
      >
        <span class="nav-icon">⬇</span> B站下载
      </button>
      <button
        class="nav-item"
        class:active={active === 'recognize'}
        onclick={() => (active = 'recognize')}
      >
        <span class="nav-icon">🔍</span> 音频识别
      </button>
    </nav>
  </aside>

  <main class="content">
    {#if active === 'download'}
      <DownloadView />
    {:else}
      <RecognizerView />
    {/if}
  </main>
</div>

<style>
  .layout {
    display: flex;
    height: 100vh;
    width: 100vw;
  }
  .sidebar {
    width: 220px;
    flex-shrink: 0;
    background: #1f2937;
    color: #e5e7eb;
    display: flex;
    flex-direction: column;
    padding: 1rem 0;
  }
  .brand {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0 1.2rem 1rem;
    font-size: 1.05rem;
    font-weight: 700;
    border-bottom: 1px solid #374151;
    margin-bottom: 0.5rem;
  }
  .logo {
    color: #38bdf8;
    font-size: 1.3rem;
  }
  nav {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    padding: 0 0.6rem;
  }
  .nav-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    width: 100%;
    text-align: left;
    padding: 0.7rem 0.8rem;
    border: none;
    background: transparent;
    color: #cbd5e1;
    font-size: 0.95rem;
    border-radius: 8px;
    cursor: pointer;
  }
  .nav-item:hover {
    background: #374151;
    color: #fff;
  }
  .nav-item.active {
    background: #2563eb;
    color: #fff;
  }
  .nav-icon {
    font-size: 1rem;
  }
  .content {
    flex: 1;
    background: #f5f7fa;
    overflow: hidden;
  }
</style>
