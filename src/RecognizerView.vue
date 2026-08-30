<script setup lang="ts">
import { onActivated, ref } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { open } from "@tauri-apps/plugin-dialog";
import {
  AudioOutlined,
  SearchOutlined,
} from "@ant-design/icons-vue";

interface SongInfo {
  title: string;
  artist: string;
  album: string | null;
  album_date: string | null;
  confidence: number;
}

interface HistoryItem {
  id: number;
  payload: string;
}

const filePath = ref("");
const result = ref<SongInfo | null>(null);
const message = ref("");
const identifying = ref(false);

// 与后端同步：识别为同步阻塞调用（无后台任务队列），后端唯一权威数据即历史库。
// 重新可见时拉取最近一条识别记录回填；若用户已手动选择文件则保留本地结果不被覆盖。
async function syncFromHistory() {
  if (identifying.value || filePath.value) return;
  try {
    const rows = await invoke<HistoryItem[]>("get_history", {
      kind: "recognize",
      limit: 1,
    });
    result.value = rows.length > 0 ? (JSON.parse(rows[0].payload) as SongInfo) : null;
  } catch {
    // 同步失败不打断本地展示
  }
}

onActivated(syncFromHistory);

async function pickFile() {
  try {
    const sel = await open({
      multiple: false,
      title: "选择音频文件",
      filters: [
        {
          name: "音频",
          extensions: ["mp3", "flac", "m4a", "wav", "ogg", "opus", "aac"],
        },
      ],
    });
    if (typeof sel === "string") {
      filePath.value = sel;
      result.value = null;
      message.value = "";
    }
  } catch (e) {
    message.value = "选择文件失败：" + String(e);
  }
}

async function doIdentify() {
  if (!filePath.value) {
    message.value = "请先选择音频文件";
    return;
  }
  identifying.value = true;
  message.value = "";
  result.value = null;
  try {
    const r = await invoke<SongInfo>("identify", { path: filePath.value });
    result.value = r;
  } catch (e) {
    message.value = "识别失败：" + String(e);
  } finally {
    identifying.value = false;
  }
}
</script>

<template>
  <div class="panel">
    <a-card title="音频识别" :bordered="false" class="main-card">
      <p class="desc">
        通过音频指纹（Chromaprint）匹配 AcoustID，识别歌曲的标题、艺术家与专辑信息。
      </p>

      <a-space wrap class="input">
        <a-button @click="pickFile">
          <template #icon><AudioOutlined /></template>
          选择音频文件
        </a-button>
        <a-typography-text
          v-if="filePath"
          type="secondary"
          :ellipsis="{ tooltip: filePath }"
          style="max-width: 320px"
        >
          {{ filePath }}
        </a-typography-text>
        <a-typography-text v-else type="secondary">未选择文件</a-typography-text>
        <a-button
          type="primary"
          :loading="identifying"
          :disabled="!filePath"
          @click="doIdentify"
        >
          <template #icon><SearchOutlined /></template>
          {{ identifying ? "识别中…" : "开始识别" }}
        </a-button>
      </a-space>

      <a-alert
        v-if="message"
        class="msg"
        type="warning"
        show-icon
        :message="message"
      />

      <a-spin :spinning="identifying" tip="正在计算音频指纹并查询 AcoustID…">
        <a-card v-if="result" size="small" class="result-card" :bordered="false">
          <a-descriptions :column="1" bordered size="small">
            <a-descriptions-item label="标题">
              {{ result.title }}
            </a-descriptions-item>
            <a-descriptions-item label="艺术家">
              {{ result.artist }}
            </a-descriptions-item>
            <a-descriptions-item label="专辑">
              {{ result.album ?? "—" }}
            </a-descriptions-item>
            <a-descriptions-item label="发行日期">
              {{ result.album_date ?? "—" }}
            </a-descriptions-item>
            <a-descriptions-item label="置信度">
              {{ result.confidence.toFixed(1) }}%
            </a-descriptions-item>
          </a-descriptions>
        </a-card>
        <div v-else-if="!identifying" class="placeholder">
          选择音频文件并点击「开始识别」以查看结果
        </div>
      </a-spin>
    </a-card>
  </div>
</template>

<style scoped>
.panel {
  padding: 1.5rem;
  height: 100%;
  overflow-y: auto;
}
.main-card {
  background: #fff;
}
.desc {
  color: #6b7280;
  font-size: 0.9rem;
}
.input {
  margin: 1rem 0;
}
.msg {
  margin: 1rem 0;
}
.result-card {
  margin-top: 1rem;
  background: #fafafa;
}
.placeholder {
  margin-top: 1rem;
  padding: 2rem;
  text-align: center;
  color: #9ca3af;
  background: #fafafa;
  border: 1px dashed #e2e8f0;
  border-radius: 8px;
}
</style>
