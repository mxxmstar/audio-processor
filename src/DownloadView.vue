<script setup lang="ts">
import { onMounted, onUnmounted, reactive, ref } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import {
  FolderOpenOutlined,
  SearchOutlined,
  DownloadOutlined,
} from "@ant-design/icons-vue";
import type { MenuProps } from "ant-design-vue";

type TaskStatus = "Pending" | "Downloading" | "Completed" | "Failed";

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

// 从左侧栏接收登录态（登录态统一在 App.vue 管理）
const { loggedIn } = defineProps<{ loggedIn: boolean }>();

const inputUrl = ref("");
const preferFormat = ref("1080P");
const mode = ref<"audio" | "video" | "merge">("audio");
const outputDir = ref("");

const tasks = ref<Task[]>([]);
const resolving = ref(false);
const downloading = ref(false);
const message = ref("");

const progressMap: Record<string, ProgressEvent> = reactive({});

const modeOptions: MenuProps["items"] = [
  { label: "仅音频", value: "audio" },
  { label: "仅视频", value: "video" },
  { label: "音视频合并", value: "merge" },
];
const formatOptions = ["360P", "720P", "1080P", "4K", "8K"];

function fmtBytes(n: number): string {
  if (!n) return "0 B";
  const u = ["B", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(n) / Math.log(1024));
  return (n / Math.pow(1024, i)).toFixed(2) + " " + u[i];
}

function statusColor(s: TaskStatus): string {
  switch (s) {
    case "Completed":
      return "success";
    case "Failed":
      return "error";
    case "Downloading":
      return "processing";
    default:
      return "default";
  }
}

async function pickDir() {
  try {
    const sel = await open({
      directory: true,
      multiple: false,
      title: "选择下载目录",
    });
    if (typeof sel === "string") {
      outputDir.value = sel;
    }
  } catch (e) {
    message.value = "选择目录失败：" + String(e);
  }
}

async function doResolve() {
  if (!loggedIn) {
    message.value = "请先扫码登录";
    return;
  }
  if (!inputUrl.value.trim()) {
    message.value = "请输入 BV 号或链接";
    return;
  }
  resolving.value = true;
  message.value = "";
  try {
    tasks.value = await invoke<Task[]>("bili_resolve", {
      input: {
        input: inputUrl.value.trim(),
        mode: mode.value,
        preferFormat: preferFormat.value,
        outputDir: outputDir.value || null,
      },
    });
  } catch (e) {
    message.value = String(e);
    tasks.value = [];
  } finally {
    resolving.value = false;
  }
}

async function doDownload() {
  if (tasks.value.length === 0) return;
  downloading.value = true;
  message.value = "开始下载…";
  try {
    await invoke<string[]>("bili_start_download", {
      input: { outputDir: outputDir.value || null, concurrency: 3 },
    });
  } catch (e) {
    message.value = String(e);
    downloading.value = false;
  }
}

let off1: UnlistenFn | null = null;
let off2: UnlistenFn | null = null;

onMounted(async () => {
  off1 = await listen<ProgressEvent>("download-progress", (e) => {
    progressMap[e.payload.task_id] = e.payload;
  });
  off2 = await listen<{ ok: boolean; failed: number }>(
    "download-finished",
    (e) => {
      downloading.value = false;
      message.value = e.payload.ok
        ? "全部下载完成"
        : `下载结束，${e.payload.failed} 个失败`;
      invoke<Task[]>("bili_list_tasks")
        .then((t) => (tasks.value = t))
        .catch(() => {});
    }
  );
});

onUnmounted(() => {
  off1?.();
  off2?.();
});
</script>

<template>
  <div class="panel">
    <a-card v-if="loggedIn" title="B站下载" :bordered="false" class="main-card">
      <a-form layout="vertical">
        <a-form-item label="视频地址">
          <a-input
            v-model:value="inputUrl"
            placeholder="BV 号 / 链接 / av 号 / 合集(ss) / 番剧"
            allow-clear
          />
        </a-form-item>
        <a-row :gutter="12">
          <a-col :span="8">
            <a-form-item label="下载模式">
              <a-select v-model:value="mode" :options="modeOptions" />
            </a-form-item>
          </a-col>
          <a-col :span="8">
            <a-form-item label="首选清晰度">
              <a-select v-model:value="preferFormat" :options="formatOptions.map((f) => ({ label: f, value: f }))" />
            </a-form-item>
          </a-col>
          <a-col :span="8">
            <a-form-item label="输出目录">
              <a-space>
                <a-button @click="pickDir">
                  <template #icon><FolderOpenOutlined /></template>
                  选择目录
                </a-button>
                <a-typography-text type="secondary" :ellipsis="{ tooltip: outputDir }">
                  {{ outputDir || "默认：应用配置目录" }}
                </a-typography-text>
              </a-space>
            </a-form-item>
          </a-col>
        </a-row>
        <a-space>
          <a-button type="primary" :loading="resolving" @click="doResolve">
            <template #icon><SearchOutlined /></template>
            {{ resolving ? "解析中…" : "解析" }}
          </a-button>
          <a-button
            type="primary"
            :loading="downloading"
            :disabled="tasks.length === 0"
            @click="doDownload"
          >
            <template #icon><DownloadOutlined /></template>
            {{ downloading ? "下载中…" : "开始下载" }}
          </a-button>
        </a-space>
      </a-form>

      <a-alert
        v-if="message"
        class="msg"
        type="info"
        show-icon
        :message="message"
      />

      <a-list
        v-if="tasks.length"
        class="task-list"
        item-layout="horizontal"
        :data-source="tasks"
      >
        <template #renderItem="{ item }">
          <a-list-item>
            <a-card size="small" :bordered="false" class="task-card">
              <div class="t-title">
                {{ item.title
                }}{{ item.part ? " - " + item.part : "" }}
              </div>
              <div class="t-meta">
                <a-tag :color="statusColor(item.status)">{{ item.status }}</a-tag>
                <a-tag>{{ item.mode }}</a-tag>
                <template v-if="progressMap[item.id]">
                  {{ fmtBytes(progressMap[item.id].downloaded)
                  }}<template v-if="progressMap[item.id].total">
                    / {{ fmtBytes(progressMap[item.id].total) }}</template>
                  · {{ fmtBytes(progressMap[item.id].speed) }}/s
                </template>
                <span v-if="item.error" class="err"> · {{ item.error }}</span>
              </div>
              <a-progress
                :percent="
                  progressMap[item.id]
                    ? Math.round(progressMap[item.id].percent * 100)
                    : item.status === 'Completed'
                    ? 100
                    : 0
                "
                :status="item.status === 'Failed' ? 'exception' : undefined"
                size="small"
              />
            </a-card>
          </a-list-item>
        </template>
      </a-list>
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
.msg {
  margin: 1rem 0;
}
.task-list {
  margin-top: 1rem;
}
.task-card {
  width: 100%;
  background: #fafafa;
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
</style>
