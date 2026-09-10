<script setup lang="ts">
import { computed, nextTick, onActivated, onMounted, onUnmounted, reactive, ref } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import {
  FolderOpenOutlined,
  SearchOutlined,
  DownloadOutlined,
  QrcodeOutlined,
  UserOutlined,
} from "@ant-design/icons-vue";
import type { MenuProps } from "ant-design-vue";

type TaskStatus = "Pending" | "Downloading" | "Completed" | "Failed" | "Paused" | "Cancelled";
type RecognitionStatus =
  | "Disabled"
  | "Pending"
  | "Recognizing"
  | "Renamed"
  | "NoMatch"
  | "BelowThreshold"
  | "Failed"
  | "RenameFailed";
type QualityStatus =
  | "Disabled"
  | "Pending"
  | "CheckingRuntime"
  | "LoadingModel"
  | "Enhancing"
  | "Completed"
  | "Failed";

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
  source_title?: string;
  recognition_status?: RecognitionStatus;
  recognition_result?: {
    title: string;
    artist: string;
    album: string | null;
    album_date: string | null;
    confidence: number;
  } | null;
  recognition_error?: string | null;
  quality_status?: QualityStatus;
  quality_model_id?: string | null;
  quality_output_path?: string | null;
  quality_error?: string | null;
  output_path?: string | null;
  group: TaskGroup | null;
  cover: string | null;
}

interface ProgressEvent {
  phase: string; // "resolve" | "download" | "recognize" | "rename" | "enhance"
  task_id: string;
  title: string;
  status: string;
  percent: number;
  downloaded: number;
  total: number;
  speed: number;
  error: string | null;
  source_title: string;
  output_path: string | null;
  confidence: number | null;
  quality_status?: QualityStatus;
  quality_model_id?: string | null;
  quality_output_path?: string | null;
  quality_error?: string | null;
}

interface ResolveFinished {
  ok: boolean;
  tasks: Task[];
  error: string | null;
  total: number;
  resolved: number;
}

// 从左侧栏接收登录态（登录态统一在 App.vue 管理）；未登录时请求拉起登录二维码
const props = defineProps<{ loggedIn: boolean }>();
const emit = defineEmits<{ (e: "requestLogin"): void }>();

const inputUrl = ref("");
const preferFormat = ref("1080P");
const mode = ref<"audio" | "video" | "merge">("audio");
const outputDir = ref("");
const autoRename = ref(true);
const confidenceThreshold = ref(70);
const pythonAiEnhancementEnabled = ref(false);
const pythonAiModelId = ref("audiosr-basic");

const tasks = ref<Task[]>([]);
const resolving = ref(false);
const downloading = ref(false);
const paused = ref(false);
const message = ref("");

// 勾选状态：以任务 id 为键的集合，仅勾选的任务会被下载
const selectedIds = ref<Set<string>>(new Set());

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

function recognitionText(s?: RecognitionStatus): string {
  switch (s) {
    case "Recognizing": return "识别中";
    case "Renamed": return "已重命名";
    case "NoMatch": return "未匹配";
    case "BelowThreshold": return "置信度不足";
    case "Failed": return "识别失败";
    case "RenameFailed": return "重命名失败";
    case "Disabled": return "未启用识别";
    default: return "等待识别";
  }
}

function recognitionColor(s?: RecognitionStatus): string {
  switch (s) {
    case "Renamed": return "success";
    case "Recognizing": return "processing";
    case "NoMatch":
    case "BelowThreshold": return "warning";
    case "Failed":
    case "RenameFailed": return "error";
    default: return "default";
  }
}

function qualityText(s?: QualityStatus): string {
  switch (s) {
    case "CheckingRuntime": return "检查 AI 运行时";
    case "LoadingModel": return "加载 AI 模型";
    case "Enhancing": return "AI 增强中";
    case "Completed": return "AI 增强完成";
    case "Failed": return "AI 增强失败";
    case "Disabled": return "未启用 AI";
    default: return "等待 AI 增强";
  }
}

function qualityColor(s?: QualityStatus): string {
  switch (s) {
    case "Completed": return "success";
    case "CheckingRuntime":
    case "LoadingModel":
    case "Enhancing": return "processing";
    case "Failed": return "error";
    default: return "default";
  }
}

function downloadInput() {
  return {
    outputDir: outputDir.value || null,
    concurrency: 3,
    taskIds: Array.from(selectedIds.value),
    autoRename: mode.value === "audio" ? autoRename.value : false,
    confidenceThreshold: confidenceThreshold.value,
    pythonAiEnhancementEnabled:
      mode.value === "audio" ? pythonAiEnhancementEnabled.value : false,
    pythonAiModelId: pythonAiModelId.value,
  };
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
  if (!props.loggedIn) {
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
  paused.value = false;
  message.value = "开始下载…";
  try {
    await invoke<string[]>("bili_start_download", {
      input: downloadInput(),
    });
  } catch (e) {
    message.value = String(e);
    downloading.value = false;
  }
}

// 继续下载：复用开始下载逻辑（基于已下载部分断点续传）
async function doResume() {
  if (tasks.value.length === 0) return;
  paused.value = false;
  downloading.value = true;
  message.value = "继续下载…";
  try {
    await invoke<string[]>("bili_start_download", {
      input: downloadInput(),
    });
  } catch (e) {
    message.value = String(e);
    downloading.value = false;
  }
}

async function doPause() {
  try {
    await invoke("bili_pause_download");
    paused.value = true;
    message.value = "已暂停下载（可继续 / 断点续传）";
  } catch (e) {
    message.value = String(e);
  }
}

async function doStop() {
  try {
    await invoke("bili_stop_download");
    // 立即复位下载按钮（后台仍在收尾取消其余任务，但本轮交互已结束）
    downloading.value = false;
    paused.value = false;
    message.value = "已停止下载，已删除已下载部分";
  } catch (e) {
    message.value = String(e);
  }
}

// 已勾选数量
const selectedCount = computed(() => selectedIds.value.size);

function isSelected(id: string): boolean {
  return selectedIds.value.has(id);
}

// 全部选择：勾选当前所有任务
function selectAll() {
  selectedIds.value = new Set(tasks.value.map((t) => t.id));
  rangeIds.value = new Set();
}

// 全部取消：清空勾选
function clearSelection() {
  selectedIds.value = new Set();
  rangeIds.value = new Set();
}

// 解析结果写入时，默认全选（保留已有勾选状态，新增任务默认选中）
function syncSelectionOnTasks() {
  const s = new Set(selectedIds.value);
  for (const t of tasks.value) s.add(t.id);
  selectedIds.value = s;
  rangeIds.value = new Set();
}

// 扁平化任务顺序（合集分组后的展示顺序），用于 shift 连续多选的索引定位
const orderedTaskIds = computed<string[]>(() =>
  groupedTasks.value.flatMap((g) => g.tasks.map((t) => t.id))
);
// 上一次点击的任务索引（shift 多选的锚点）
const lastTaskIndex = ref(-1);
// shift 框选的「蓝色临时选中」集合（尚未真正勾选，点击任意勾选框后批量应用）
const rangeIds = ref<Set<string>>(new Set());

function inRange(id: string): boolean {
  return rangeIds.value.has(id);
}

// 任务点击交互：
// - 普通点击 = 切换单个勾选；若当前存在蓝色框选区，则把点击的勾选状态批量应用到整个框选区后清除
// - shift+点击 = 以锚点为起点，框选 [锚点, 当前] 区间到 rangeIds（仅高亮，不立即勾选）
function onTaskClick(id: string, shift: boolean) {
  const flat = orderedTaskIds.value;
  const idx = flat.indexOf(id);
  if (shift) {
    if (lastTaskIndex.value < 0) lastTaskIndex.value = idx;
    const [a, b] = idx >= lastTaskIndex.value
      ? [lastTaskIndex.value, idx]
      : [idx, lastTaskIndex.value];
    const s = new Set<string>();
    for (let i = a; i <= b; i++) s.add(flat[i]);
    rangeIds.value = s;
    return;
  }
  // 普通点击：若存在蓝色框选区，则批量应用
  if (rangeIds.value.size > 0) {
    const target = !isSelected(id); // 以被点击项的「新状态」为准
    const s = new Set(selectedIds.value);
    for (const rid of rangeIds.value) {
      if (target) s.add(rid);
      else s.delete(rid);
    }
    selectedIds.value = s;
    rangeIds.value = new Set();
    lastTaskIndex.value = idx;
    return;
  }
  // 无任何选区：切换单个
  const s = new Set(selectedIds.value);
  if (s.has(id)) s.delete(id);
  else s.add(id);
  selectedIds.value = s;
  lastTaskIndex.value = idx;
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
      // 下载阶段进度。后端此时会携带任务的真实枚举状态（如
      // "Downloading" / "Cancelled" / "Paused" / "Completed"），
      // 据此实时更新对应任务的状态展示。
      progressMap[p.task_id] = p;
      const t = tasks.value.find((x) => x.id === p.task_id);
      if (t) {
        if (p.phase === "download") t.status = p.status as TaskStatus;
        if (p.phase === "recognize" || p.phase === "rename") {
          t.recognition_status = p.status as RecognitionStatus;
          t.recognition_error = p.error;
          t.output_path = p.output_path;
        }
        if (p.phase === "enhance") {
          t.quality_status = p.quality_status || (p.status as QualityStatus);
          t.quality_model_id = p.quality_model_id;
          t.quality_output_path = p.quality_output_path;
          t.quality_error = p.quality_error;
        }
      }
      if (p.phase === "recognize") message.value = `识别中 · ${p.title}`;
      if (p.phase === "rename") message.value = `正在生成 MP3 · ${p.title}`;
      if (p.phase === "enhance") message.value = `${qualityText(p.quality_status)} · ${p.title}`;
      // 仅「停止 / 失败」清除进度展示；「暂停 / 完成」保留
      // （暂停需展示断点进度，完成由 download-finished 统一清理）。
      if (p.status === "Cancelled" || (p.phase === "download" && p.status === "Failed")) {
        delete progressMap[p.task_id];
      }
    }
  });
  off2 = await listen<{
    ok: boolean;
    failed: number;
    renamed: number;
    recognize_failed: number;
    fallback: number;
    enhanced: number;
    enhance_failed: number;
  }>(
    "download-finished",
    (e) => {
      downloading.value = false;
      paused.value = false;
      message.value = e.payload.ok
        ? `下载完成，${e.payload.renamed} 个已重命名${e.payload.fallback ? `，${e.payload.fallback} 个使用原始标题` : ""}${e.payload.enhanced ? `，${e.payload.enhanced} 个已完成 AI 增强` : ""}${e.payload.enhance_failed ? `，${e.payload.enhance_failed} 个 AI 增强失败` : ""}`
        : `下载结束，${e.payload.failed} 个下载失败`;
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
    syncSelectionOnTasks();
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

// 由 keep-alive 缓存后，切换界面不会销毁组件；此处仅在重新可见时
// 与后端任务快照做一次轻量同步，确保展示与后端状态一致。
// 仅当本地已有任务时才同步，避免「解析失败清空后又被回填」。
onActivated(() => {
  if (tasks.value.length === 0) return;
  invoke<Task[]>("bili_list_tasks")
    .then((t) => {
      if (t.length > 0) tasks.value = t;
    })
    .catch(() => {});
});
</script>

<template>
  <div class="panel">
    <a-card v-if="props.loggedIn" title="B站下载" :bordered="false" class="main-card">
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
        <a-row v-if="mode === 'audio'" :gutter="12">
          <a-col :span="8">
            <a-form-item label="原始音频识别并重命名">
              <a-switch v-model:checked="autoRename" />
            </a-form-item>
          </a-col>
          <a-col :span="8" v-if="autoRename">
            <a-form-item label="最低识别置信度">
              <a-input-number v-model:value="confidenceThreshold" :min="0" :max="100" :precision="1" addon-after="%" />
            </a-form-item>
          </a-col>
          <a-col :span="8">
            <a-form-item label="Python AI 音频增强">
              <a-switch v-model:checked="pythonAiEnhancementEnabled" />
            </a-form-item>
          </a-col>
          <a-col :span="8" v-if="pythonAiEnhancementEnabled">
            <a-form-item label="AI 模型">
              <a-select v-model:value="pythonAiModelId" :options="[{ label: 'AudioSR 通用音乐恢复', value: 'audiosr-basic' }]" />
            </a-form-item>
          </a-col>
        </a-row>
        <a-space>
          <a-button type="primary" :loading="resolving" @click="doResolve">
            <template #icon><SearchOutlined /></template>
            {{ resolving ? "解析中…" : "解析" }}
          </a-button>
          <!-- 未开始：开始下载 -->
          <a-button
            v-if="!downloading"
            type="primary"
            :disabled="tasks.length === 0"
            @click="doDownload"
          >
            <template #icon><DownloadOutlined /></template>
            开始下载
          </a-button>
          <!-- 下载中：显示进行中 + 暂停 / 停止 -->
          <a-button
            v-else-if="downloading && !paused"
            type="primary"
            :loading="true"
            disabled
          >
            <template #icon><DownloadOutlined /></template>
            下载中…
          </a-button>
          <!-- 已暂停：显示继续下载 -->
          <a-button
            v-else
            type="primary"
            @click="doResume"
          >
            <template #icon><DownloadOutlined /></template>
            继续下载
          </a-button>
          <a-button
            v-if="downloading && !paused"
            :disabled="!downloading"
            @click="doPause"
          >
            暂停下载
          </a-button>
          <a-button
            v-if="downloading"
            danger
            :disabled="!downloading"
            @click="doStop"
          >
            停止下载
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

      <a-space
        v-if="tasks.length"
        class="select-bar"
        wrap
      >
        <a-checkbox
          :checked="selectedCount === tasks.length && tasks.length > 0"
          :indeterminate="selectedCount > 0 && selectedCount < tasks.length"
          @change="(e: any) => (e.target.checked ? selectAll() : clearSelection())"
        >
          全选
        </a-checkbox>
        <a-button size="small" @click="selectAll">全部选择</a-button>
        <a-button size="small" @click="clearSelection">全部取消</a-button>
        <span class="sel-count">已选 {{ selectedCount }} / {{ tasks.length }}</span>
        <span class="sel-tip">提示：按住 Shift 点击可批量选择连续项</span>
      </a-space>

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
                <div
                  class="task-row"
                  :class="{ 'is-selected': isSelected(item.id), 'in-range': inRange(item.id) }"
                  @click="(e: MouseEvent) => onTaskClick(item.id, e.shiftKey)"
                >
                  <a-checkbox
                    class="task-check"
                    :checked="isSelected(item.id)"
                    @click.stop="(e: MouseEvent) => onTaskClick(item.id, e.shiftKey)"
                  />
                  <img
                    v-if="item.cover"
                    :src="item.cover"
                    class="task-cover"
                    alt="cover"
                    referrerpolicy="no-referrer"
                  />
                  <a-card size="small" :bordered="false" class="task-card">
                    <div class="t-title">
                      {{ item.title
                      }}{{ item.part ? " - " + item.part : "" }}
                    </div>
                  <div class="t-meta">
                    <a-tag :color="statusColor(item.status)">{{ statusText(item.status) }}</a-tag>
                    <a-tag>{{ item.mode }}</a-tag>
                    <a-tag v-if="item.mode === 'AudioOnly' || item.mode === 'audio'" :color="recognitionColor(item.recognition_status)">
                      {{ recognitionText(item.recognition_status) }}
                    </a-tag>
                    <span v-if="item.recognition_result" class="recognition-meta">
                      {{ item.recognition_result.title }} · {{ item.recognition_result.artist }}
                      · {{ item.recognition_result.confidence.toFixed(1) }}%
                    </span>
                    <a-tag v-if="item.status === 'Downloading' && progressMap[item.id]" color="blue">下载中</a-tag>
                    <a-tag v-else-if="item.status === 'Paused' && progressMap[item.id]" color="gold">已暂停</a-tag>
                    <template v-if="progressMap[item.id] && (item.status === 'Downloading' || item.status === 'Paused')">
                      {{ fmtBytes(progressMap[item.id].downloaded)
                      }}<template v-if="progressMap[item.id].total">
                        / {{ fmtBytes(progressMap[item.id].total) }}</template>
                      <template v-if="item.status === 'Downloading'">
                        · {{ fmtBytes(progressMap[item.id].speed) }}/s</template>
                      <template v-else> · 已暂停</template>
                    </template>
                    <span v-if="item.error" class="err"> · {{ item.error }}</span>
                    <span v-if="item.recognition_error" class="err"> · {{ item.recognition_error }}</span>
                    <a-tag v-if="item.mode === 'AudioOnly' || item.mode === 'audio'" :color="qualityColor(item.quality_status)">
                      {{ qualityText(item.quality_status) }}
                    </a-tag>
                    <span v-if="item.quality_error" class="err"> · {{ item.quality_error }}</span>
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
                </div>
              </a-list-item>
            </template>
          </a-list>
        </a-collapse-panel>
      </a-collapse>
    </a-card>

    <!-- 未登录：给出明确提示与一键登录入口，避免内容区空白（白屏） -->
    <a-card v-else title="B站下载" :bordered="false" class="main-card login-tip">
      <a-result status="info" title="尚未登录 B站账号">
        <template #subTitle>
          <p>使用本功能前请先在左侧栏点击「扫码登录」，或点击下方按钮拉起二维码。</p>
          <p class="muted">登录状态仅影响「B站下载」，不影响音频识别、历史记录等其它模块。</p>
        </template>
        <template #extra>
          <a-button type="primary" @click="emit('requestLogin')">
            <template #icon><QrcodeOutlined /></template>
            扫码登录
          </a-button>
        </template>
      </a-result>
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
.login-tip {
  margin-top: 1rem;
}
.muted {
  color: #8a94a6;
  font-size: 0.85rem;
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
.select-bar {
  margin: 1rem 0 0.5rem;
}
.sel-count {
  font-size: 0.85rem;
  color: #8a94a6;
}
.sel-tip {
  font-size: 0.8rem;
  color: #b0b8c4;
}
.task-row {
  display: flex;
  align-items: stretch;
  gap: 0.8rem;
  width: 100%;
  cursor: pointer;
  border-radius: 6px;
  padding: 0.2rem 0.3rem;
  transition: background 0.15s ease;
  user-select: none;
}
.task-row:hover {
  background: #f3f6fb;
}
.task-row.is-selected {
  background: #e6e6e6;
}
.task-row.is-selected .task-card {
  background: #e6e6e6;
}
/* shift 框选的蓝色临时高亮（优先于灰色，表示待批量操作） */
.task-row.in-range {
  background: rgba(24, 144, 255, 0.18);
  box-shadow: inset 0 0 0 2px #1890ff;
}
.task-row.in-range .task-card {
  background: transparent;
}
.task-check {
  flex: 0 0 auto;
  display: flex;
  align-items: center;
}
.task-cover {
  width: 96px;
  height: 60px;
  object-fit: cover;
  border-radius: 6px;
  flex: 0 0 auto;
  align-self: center;
  background: #eee;
}
.task-card {
  flex: 1 1 auto;
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
.recognition-meta {
  color: #3d5a80;
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
