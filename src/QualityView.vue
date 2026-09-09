<script setup lang="ts">
import { computed, onActivated, onMounted, onUnmounted, ref, watch } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { message, Modal } from "ant-design-vue";
import {
  SoundOutlined,
  FileSearchOutlined,
  ReloadOutlined,
  FolderOpenOutlined,
  DownloadOutlined,
} from "@ant-design/icons-vue";

interface AiRuntimeCheck {
  available: boolean;
  productionReady: boolean;
  protocolVersion: number;
  workerKind: string;
  program: string | null;
  workerVersion: string | null;
  models: string[];
  modelErrors: string[];
  error: string | null;
}

interface QualityTask {
  id: string;
  inputPath: string;
  outputPath: string;
  /** queued | processing | completed | failed | cancelled */
  status: string;
  phase: string;
  percent: number;
  modelId: string;
  device: string;
  workerKind: string;
  message: string | null;
  error: string | null;
  outputSampleRate: number | null;
  outputChannels: number | null;
  outputDurationSeconds: number | null;
}

interface ModelInfo {
  id: string;
  backend: string;
}

const inputPath = ref("");
const modelId = ref("flashsr");
const device = ref("auto");
const sampleRate = ref(48000);
/** VoiceFixer 修复模式：0 原始（默认）/ 1 去高频预处理 / 2 训练模式 */
const vfMode = ref(0);

const tasks = ref<QualityTask[]>([]);
const models = ref<ModelInfo[]>([]);
const runtime = ref<AiRuntimeCheck | null>(null);
const checking = ref(false);
const starting = ref(false);
const installing = ref(false);
const activeTask = ref<QualityTask | null>(null);

const running = computed(() =>
  tasks.value.some((t) => t.status === "queued" || t.status === "processing")
);

// 由后端 `audio_quality_list_models` 驱动可选模型清单，并标注可用性。
const availableModelIds = computed(() => new Set(runtime.value?.models ?? []));

function backendOf(id: string): string {
  const found = models.value.find((m) => m.id === id);
  if (found) return found.backend;
  if (id.startsWith("flashsr")) return "flashsr";
  if (id.startsWith("audiosr")) return "audiosr";
  if (id.startsWith("voicefixer")) return "voicefixer";
  return "";
}

const modelOptions = computed(() => {
  const list = models.value.length
    ? models.value
    : [{ id: modelId.value, backend: backendOf(modelId.value) }];
  return list.map((m) => {
    const backendLabel =
      m.backend === "flashsr"
        ? "FlashSR"
        : m.backend === "audiosr"
          ? "AudioSR"
          : m.backend === "voicefixer"
            ? "语音修复（VoiceFixer）"
            : m.backend;
    const available = availableModelIds.value.has(m.id);
    const desc =
      m.backend === "flashsr"
        ? "标准（快）"
        : m.backend === "audiosr"
          ? "高保真（慢）"
          : m.backend === "voicefixer"
            ? "噪声 / 混响 / 低带宽 / 削波，面向人声"
            : "";
    return {
      label: `${backendLabel} · ${m.id}${available ? " · 可用" : " · 不可用"}`,
      value: m.id,
      title: desc,
    };
  });
});

const isVoiceFixer = computed(() => backendOf(modelId.value) === "voicefixer");

const modelHint = computed(() => {
  const backend = backendOf(modelId.value);
  if (backend === "flashsr") return "FlashSR = 标准（快）";
  if (backend === "audiosr") return "AudioSR = 高保真（慢）";
  if (backend === "voicefixer")
    return "面向人声/语音（播客、录音、视频人声、电话录音）；音乐请用 FlashSR / AudioSR";
  return "";
});

const modeOptions = [
  { label: "0 · 原始（推荐）", value: 0 },
  { label: "1 · 预处理去高频", value: 1 },
  { label: "2 · 训练模式（严重退化）", value: 2 },
];

const deviceOptions = [
  { label: "自动", value: "auto" },
  { label: "CPU", value: "cpu" },
  { label: "CUDA", value: "cuda" },
];

// FlashSR / AudioSR 固定 48 kHz；VoiceFixer 原生 **44.1 kHz**（实施计划 §3.5），
// 不能套用 48 kHz 校验，否则会把修复结果重采样。
const fixedSampleRate = computed(() => {
  if (isVoiceFixer.value) return 44100;
  const backend = backendOf(modelId.value);
  return backend === "flashsr" || backend === "audiosr" ? 48000 : null;
});

const sampleRateOptions = computed(() =>
  fixedSampleRate.value
    ? [{ label: `${fixedSampleRate.value} Hz（模型固定）`, value: fixedSampleRate.value }]
    : [
        { label: "48000 Hz", value: 48000 },
        { label: "44100 Hz", value: 44100 },
      ]
);

// 切换模型时把采样率锁定到该后端的原生速率。
watch(fixedSampleRate, (rate) => {
  if (rate) sampleRate.value = rate;
});

function statusColor(s: string): string {
  switch (s) {
    case "completed":
      return "success";
    case "processing":
      return "processing";
    case "failed":
      return "error";
    case "cancelled":
      return "warning";
    default:
      return "default";
  }
}

function statusText(s: string): string {
  switch (s) {
    case "queued":
      return "排队中";
    case "processing":
      return "处理中";
    case "completed":
      return "已完成";
    case "failed":
      return "失败";
    case "cancelled":
      return "已取消";
    default:
      return s;
  }
}

function fileName(p: string): string {
  const parts = p.split(/[\\/]/);
  return parts[parts.length - 1] || p;
}

async function checkRuntime() {
  checking.value = true;
  try {
    runtime.value = await invoke<AiRuntimeCheck>("audio_quality_check_ai_runtime", {
      modelId: modelId.value,
    });
    const available = runtime.value.models;
    // 默认优先 FlashSR；当前模型不可用时回退到首个可用模型。
    // 若连可用模型都没有（仅连上运行时但未安装权重），保留 FlashSR 作为
    // 选择项，以便用户点击「安装模型」下载权重。
    if (!available.includes(modelId.value) && available.length) {
      modelId.value = available[0];
    }
  } catch (e) {
    message.error("检测 AI 运行时失败：" + String(e));
  } finally {
    checking.value = false;
  }
}

async function loadModels() {
  try {
    models.value = await invoke<ModelInfo[]>("audio_quality_list_models");
  } catch {
    models.value = [];
  }
}

async function loadTasks() {
  try {
    tasks.value = await invoke<QualityTask[]>("audio_quality_list_tasks");
  } catch {
    // 列表加载失败不打断使用
  }
}

async function pickFile() {
  try {
    const sel = await open({
      multiple: false,
      title: "选择要提升音质的音频文件",
      filters: [
        {
          name: "音频",
          extensions: ["mp3", "flac", "m4a", "wav", "ogg", "opus", "aac"],
        },
      ],
    });
    if (typeof sel === "string") {
      inputPath.value = sel;
    }
  } catch (e) {
    message.error("选择文件失败：" + String(e));
  }
}

async function start() {
  if (!inputPath.value) {
    message.warning("请先选择音频文件");
    return;
  }
  starting.value = true;
  try {
    // 注意：AudioQualityStartInput 使用 serde rename_all = "camelCase"，
    // 因此字段名必须为驼峰（inputPath / modelId / outputSampleRate）
    const task = await invoke<QualityTask>("audio_quality_start", {
      input: {
        inputPath: inputPath.value,
        modelId: modelId.value,
        device: device.value,
        outputSampleRate: sampleRate.value,
        // 仅 VoiceFixer 使用；其它后端下发 undefined 即不传该字段
        mode: isVoiceFixer.value ? vfMode.value : undefined,
      },
    });
    activeTask.value = task;
    if (!tasks.value.some((t) => t.id === task.id)) {
      tasks.value = [task, ...tasks.value];
    }
    message.success("已开始音质提升");
  } catch (e) {
    message.error("启动失败：" + String(e));
  } finally {
    starting.value = false;
  }
}

async function cancel(taskId: string) {
  try {
    await invoke("audio_quality_cancel", { task_id: taskId });
    message.success("已请求取消");
  } catch (e) {
    message.error("取消失败：" + String(e));
  }
}

async function installModel() {
  installing.value = true;
  try {
    const r = await invoke<{ modelId: string; status: string; message: string }>(
      "audio_quality_download_model",
      { model_id: modelId.value, workers: 8 }
    );
    message.success(r.message);
    await checkRuntime();
  } catch (e) {
    message.error("模型安装失败：" + String(e));
  } finally {
    installing.value = false;
  }
}

async function openDir(path: string) {
  if (!path) return;
  try {
    await invoke("open_path", { path });
  } catch (e) {
    const msg = String(e);
    if (msg.includes("文件不存在")) {
      Modal.warning({
        title: "文件不存在",
        content: `该文件已被删除或移动：\n${path}`,
      });
    } else {
      message.error("打开目录失败：" + msg);
    }
  }
}

let off: UnlistenFn | null = null;

onMounted(async () => {
  await Promise.all([loadModels(), checkRuntime(), loadTasks()]);
  off = await listen<QualityTask>("audio-quality-progress", (e) => {
    const t = e.payload;
    const idx = tasks.value.findIndex((x) => x.id === t.id);
    if (idx >= 0) {
      tasks.value[idx] = t;
    } else {
      tasks.value = [t, ...tasks.value];
    }
    if (activeTask.value?.id === t.id) activeTask.value = t;
  });
});

// keep-alive 下重新可见时与后端任务列表同步
onActivated(loadTasks);

onUnmounted(() => {
  off?.();
});
</script>

<template>
  <div class="panel">
    <a-card title="音频品质提升" :bordered="false" class="main-card">
      <template #extra>
        <a-space>
          <a-button size="small" :loading="checking" @click="checkRuntime">
            <template #icon><ReloadOutlined /></template>
            重新检测
          </a-button>
        </a-space>
      </template>

      <a-alert
        v-if="runtime && !runtime.available"
        class="msg"
        type="warning"
        show-icon
        :message="`AI 运行时不可用：${runtime.error ?? '未检测到 Python AI Worker'}`"
      />
      <a-alert
        v-else-if="runtime && !runtime.productionReady"
        class="msg"
        type="info"
        show-icon
        message="AI 运行时已连接，但未检测到可用模型；请先安装模型。"
      />

      <a-descriptions
        v-if="runtime"
        :column="2"
        bordered
        size="small"
        class="runtime"
      >
        <a-descriptions-item label="运行时">
          <a-tag :color="runtime.available ? 'green' : 'red'">
            {{ runtime.available ? "可用" : "不可用" }}
          </a-tag>
          <span class="dim"> · {{ runtime.workerKind }}</span>
        </a-descriptions-item>
        <a-descriptions-item label="协议版本">
          {{ runtime.protocolVersion }}
        </a-descriptions-item>
        <a-descriptions-item label="Worker 版本">
          {{ runtime.workerVersion ?? "—" }}
        </a-descriptions-item>
        <a-descriptions-item label="可用模型">
          <span v-if="runtime.models.length">{{ runtime.models.join("、") }}</span>
          <span v-else class="dim">—</span>
        </a-descriptions-item>
      </a-descriptions>

      <a-form layout="vertical" class="form">
        <a-form-item label="输入音频">
          <a-space wrap>
            <a-button @click="pickFile">
              <template #icon><FileSearchOutlined /></template>
              选择音频文件
            </a-button>
            <a-typography-text
              v-if="inputPath"
              type="secondary"
              :ellipsis="{ tooltip: inputPath }"
              style="max-width: 420px"
            >
              {{ inputPath }}
            </a-typography-text>
            <a-typography-text v-else type="secondary">未选择文件</a-typography-text>
          </a-space>
        </a-form-item>

        <a-row :gutter="12">
          <a-col :span="6">
            <a-form-item label="模型">
              <a-space direction="vertical" :size="4">
                <a-space>
                  <a-select
                    v-model:value="modelId"
                    :options="modelOptions"
                    style="min-width: 200px"
                  />
                  <a-button
                    size="small"
                    :loading="installing"
                    :disabled="installing"
                    @click="installModel"
                  >
                    <template #icon><DownloadOutlined /></template>
                    安装模型
                  </a-button>
                </a-space>
                <span v-if="modelHint" style="color: rgba(0, 0, 0, 0.45); font-size: 12px">
                  {{ modelHint }}
                  <template v-if="fixedSampleRate">
                    （该后端输出固定 {{ fixedSampleRate }} Hz）
                  </template>
                  <template v-else>（FlashSR 仅支持 48000 Hz 输出）</template>
                </span>
              </a-space>
            </a-form-item>
          </a-col>
          <a-col :span="6">
            <a-form-item label="计算设备">
              <a-select
                v-model:value="device"
                :options="deviceOptions"
                style="min-width: 120px"
              />
            </a-form-item>
          </a-col>
          <a-col :span="6">
            <a-form-item label="输出采样率">
              <a-select
                v-model:value="sampleRate"
                :options="sampleRateOptions"
                :disabled="!!fixedSampleRate"
                style="min-width: 120px"
              />
            </a-form-item>
          </a-col>
          <a-col :span="6">
            <a-form-item label="修复模式（仅 VoiceFixer）">
              <a-select
                v-model:value="vfMode"
                :options="modeOptions"
                :disabled="!isVoiceFixer"
                style="min-width: 160px"
              />
            </a-form-item>
          </a-col>
        </a-row>

        <a-space>
          <a-button
            type="primary"
            :loading="starting"
            :disabled="!inputPath || running"
            @click="start"
          >
            <template #icon><SoundOutlined /></template>
            开始提升
          </a-button>
          <a-button
            v-if="running"
            danger
            @click="cancel((tasks.find((t) => t.status === 'processing' || t.status === 'queued') as QualityTask).id)"
          >
            取消任务
          </a-button>
        </a-space>
        <div class="hint">输出文件将自动生成在源文件旁（扩展名 .ai.flac），不会覆盖原文件。</div>
      </a-form>

      <a-empty
        v-if="!tasks.length"
        description="暂无音质提升任务"
        class="empty"
      />

      <a-list
        v-else
        :data-source="tasks"
        item-layout="vertical"
        class="task-list"
      >
        <template #renderItem="{ item }">
          <a-list-item>
            <a-card size="small" :bordered="false" class="task-card">
              <div class="t-head">
                <a-tag :color="statusColor(item.status)">
                  {{ statusText(item.status) }}
                </a-tag>
                <a-typography-text strong>{{ fileName(item.inputPath) }}</a-typography-text>
                <span class="dim">· {{ item.modelId }} · {{ item.device }}</span>
              </div>

              <div class="t-meta">
                <span v-if="item.message">{{ item.message }}</span>
                <span v-if="item.error" class="err"> · {{ item.error }}</span>
              </div>

              <a-progress
                :percent="Math.round(item.percent)"
                :status="
                  item.status === 'failed'
                    ? 'exception'
                    : item.status === 'completed'
                    ? 'success'
                    : undefined
                "
                size="small"
              />

              <div v-if="item.status === 'completed'" class="t-out">
                <a-typography-text type="secondary" :ellipsis="{ tooltip: item.outputPath }">
                  输出：{{ item.outputPath }}
                </a-typography-text>
                <a-space>
                  <span v-if="item.outputSampleRate" class="dim">
                    {{ item.outputSampleRate }} Hz
                    <template v-if="item.outputChannels">· {{ item.outputChannels }} ch</template>
                    <template v-if="item.outputDurationSeconds">
                      · {{ item.outputDurationSeconds.toFixed(1) }} s
                    </template>
                  </span>
                  <a-button type="link" size="small" @click="openDir(item.outputPath)">
                    <template #icon><FolderOpenOutlined /></template>
                    打开目录
                  </a-button>
                </a-space>
              </div>
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
  margin-bottom: 1rem;
}
.runtime {
  margin-bottom: 1rem;
}
.form {
  margin-top: 0.5rem;
}
.hint {
  margin-top: 0.5rem;
  font-size: 0.8rem;
  color: #b0b8c4;
}
.empty {
  margin: 1.5rem 0;
}
.task-list {
  margin-top: 1rem;
}
.task-card {
  background: #fafafa;
}
.t-head {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  flex-wrap: wrap;
}
.t-meta {
  font-size: 0.8rem;
  color: #555;
  margin: 0.4rem 0;
}
.err {
  color: #c00;
}
.t-out {
  margin-top: 0.4rem;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.8rem;
  flex-wrap: wrap;
}
.dim {
  color: #8a94a6;
  font-size: 0.8rem;
}
</style>
