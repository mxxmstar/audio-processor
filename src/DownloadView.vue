<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, reactive, ref } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import {
  FolderOpenOutlined,
  SearchOutlined,
  DownloadOutlined,
} from "@ant-design/icons-vue";
import type { MenuProps } from "ant-design-vue";

type TaskStatus = "Pending" | "Downloading" | "Completed" | "Failed" | "Paused" | "Cancelled";

interface TaskGroup {
  id: string;
  title: string;
}

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
  group: TaskGroup | null;
}

interface ProgressEvent {
  phase: string; // "resolve" | "download"
  task_id: string;
  title: string;
  status: string;
  percent: number;
  downloaded: number;
  total: number;
  speed: number;
  error: string | null;
}

interface ResolveFinished {
  ok: boolean;
  tasks: Task[];
  error: string | null;
  total: number;
  resolved: number;
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

// 解析阶段进度展示（区分「解析中 / 下载中」）
const resolveDone = ref(0);
const resolveTotal = ref(0);
const resolveCurrent = ref("");

// 任务按合集分组（折叠展示）
interface TaskGroupView {
  key: string;
  title: string;
  tasks: Task[];
}
const groupedTasks = computed<TaskGroupView[]>(() => {
  const map = new Map<string, TaskGroupView>();
  for (const t of tasks.value) {
    const g = t.group;
    const key = g ? `g:${g.id}` : "single";
    const title = g ? g.title : "单条视频";
    if (!map.has(key)) {
      map.set(key, { key, title, tasks: [] });
    }
    map.get(key)!.tasks.push(t);
  }
  // 合集组在前，单条视频组在后
  return Array.from(map.values()).sort((a, b) => {
    if (a.key === "single") return 1;
    if (b.key === "single") return -1;
    return 0;
  });
});
const activeGroupKeys = ref<string[]>(["single"]);

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
    case "Paused":
      return "warning";
    case "Cancelled":
      return "error";
    default:
      return "default";
  }
}

function statusText(s: TaskStatus): string {
  switch (s) {
    case "Pending":
      return "等待中";
    case "Downloading":
      return "下载中";
    case "Completed":
      return "已完成";
    case "Failed":
      return "失败";
    case "Paused":
      return "已暂停";
    case "Cancelled":
      return "已停止";
    default:
      return s;
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
  message.value = "解析中…";
  try {
    await invoke("bili_resolve_async", {
      input: {
        input: inputUrl.value.trim(),
        mode: mode.value,
        preferFormat: preferFormat.value,
        outputDir: outputDir.value || null,
      },
    });
    // 解析结果经 resolve-finished 事件异步填充；失败也在事件中处理
  } catch (e) {
    message.value = String(e);
    tasks.value = [];
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

async function doPause() {
  try {
    await invoke("bili_pause_download");
    message.value = "已发送暂停请求，下载完成后将暂停（可续传）";
  } catch (e) {
    message.value = String(e);
  }
}

async function doStop() {
  try {
    await invoke("bili_stop_download");
    message.value = "已发送停止请求，将删除已下载部分";
  } catch (e) {
    message.value = String(e);
  }
}

let off1: UnlistenFn | null = null;
let off2: UnlistenFn | null = null;
let off3: UnlistenFn | null = null;

onMounted(async () => {
  off1 = await listen<ProgressEvent>("download-progress", (e) => {
    const p = e.payload;
    if (p.phase === "resolve") {
      // 解析阶段进度：更新解析中提示
      const m = /^resolve:(\d+)\/(\d+)$/.exec(p.task_id);
      if (m) {
        resolveDone.value = Number(m[1]);
        resolveTotal.value = Number(m[2]);
      }
      resolveCurrent.value = p.title;
      message.value = `解析中 ${resolveDone.value}/${resolveTotal.value} · ${p.title}`;
    } else {
      // 下载阶段进度：无条件写入进度表。
      // 停止/暂停后的残留由 download-finished 事件统一清空（已取消任务仍会触发该事件）。
      progressMap[p.task_id] = p;
    }
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
      // 清理进度表：整轮下载已结束，移除所有残留进度展示
      for (const k of Object.keys(progressMap)) {
        delete progressMap[k];
      }
    }
  );
  off3 = await listen<ResolveFinished>("resolve-finished", (e) => {
    resolving.value = false;
    if (e.payload.ok) {
      tasks.value = e.payload.tasks;
      message.value =
        e.payload.resolved > 0
          ? `解析完成，共 ${e.payload.resolved} 个可下载项`
          : "解析完成";
      // 解析完成后默认展开所有分组（合集折叠面板）
      nextTick(() => {
        activeGroupKeys.value = groupedTasks.value.map((g) => g.key);
      });
    } else {
      tasks.value = [];
      message.value = e.payload.error || "解析失败";
    }
  });
});

onUnmounted(() => {
  off1?.();
  off2?.();
  off3?.();
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
            placeholder="BV 号 / 链接 / av 号 / 合集(含合集页URL) / 番剧"
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
          <a-button
            danger
            :disabled="!downloading"
            @click="doStop"
          >
            停止下载
          </a-button>
          <a-button
            :disabled="!downloading"
            @click="doPause"
          >
            暂停下载
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

      <a-card
        v-if="resolving && resolveTotal > 0"
        size="small"
        class="phase-card"
        :bordered="false"
      >
        <div class="phase-title">
          <a-tag color="processing">解析中</a-tag>
          已解析 {{ resolveDone }} / {{ resolveTotal }}
          <span class="phase-cur">· {{ resolveCurrent }}</span>
        </div>
        <a-progress
          :percent="Math.round((resolveDone / resolveTotal) * 100)"
          status="active"
          size="small"
        />
      </a-card>

      <a-collapse
        v-if="groupedTasks.length"
        v-model:activeKey="activeGroupKeys"
        class="task-collapse"
        :bordered="false"
      >
        <a-collapse-panel v-for="grp in groupedTasks" :key="grp.key">
          <template #header>
            <span class="grp-header">
              <a-tag :color="grp.key === 'single' ? 'default' : 'purple'">
                {{ grp.key === "single" ? "单条" : "合集" }}
              </a-tag>
              <span class="grp-title">{{ grp.title }}</span>
              <span class="grp-count">（{{ grp.tasks.length }} 项）</span>
            </span>
          </template>
          <a-list
            class="task-list"
            item-layout="horizontal"
            :data-source="grp.tasks"
          >
            <template #renderItem="{ item }">
              <a-list-item>
                <a-card size="small" :bordered="false" class="task-card">
                  <div class="t-title">
                    {{ item.title
                    }}{{ item.part ? " - " + item.part : "" }}
                  </div>
                  <div class="t-meta">
                    <a-tag :color="statusColor(item.status)">{{ statusText(item.status) }}</a-tag>
                    <a-tag>{{ item.mode }}</a-tag>
                    <a-tag v-if="progressMap[item.id]" color="blue">下载中</a-tag>
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
                    :status="
                      item.status === 'Failed'
                        ? 'exception'
                        : item.status === 'Paused'
                        ? 'normal'
                        : item.status === 'Cancelled'
                        ? 'exception'
                        : undefined
                    "
                    size="small"
                  />
                </a-card>
              </a-list-item>
            </template>
          </a-list>
        </a-collapse-panel>
      </a-collapse>
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
.phase-card {
  margin: 1rem 0;
  background: #f0f7ff;
}
.phase-title {
  font-weight: 600;
  margin-bottom: 0.4rem;
}
.phase-cur {
  color: #555;
  font-weight: 400;
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
.task-collapse {
  margin-top: 1rem;
}
.grp-header {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
}
.grp-title {
  font-weight: 600;
}
.grp-count {
  color: #8a94a6;
  font-size: 0.8rem;
}
</style>
