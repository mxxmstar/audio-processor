<script setup lang="ts">
import { ref } from "vue";
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

const filePath = ref("");
const result = ref<SongInfo | null>(null);
const message = ref("");
const identifying = ref(false);

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
</style>
