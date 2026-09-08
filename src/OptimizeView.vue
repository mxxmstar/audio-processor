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

// 「音质优化」：HiFi-GAN 后端提供两档可选。hifigan-48k 为快速/轻量档（原 22.05k
// 重建，机械音偏重）；bigvgan-48k 为高质量/增强档（BigVGAN-v2 44.1k，机械音消除）。
const HIFIGAN_MODEL_ID = "hifigan-48k";

// 两档模型的档位说明，用于下拉标注与提示。
const MODEL_TIERS: Record<string, { tier: string; tag: string; desc: string }> = {
  "hifigan-48k": {
    tier: "快速 / 轻量",
    tag: "fast",
    desc: "原 HiFi-GAN（22.05k 权重，重建后重采样到 48k）：推理快、权重仅约 53MB，但带宽受限、机械音偏重，不提升已丢失的高频。",
  },
  "bigvgan-48k": {
    tier: "高质量 / 增强",
    tag: "quality",
    desc: "BigVGAN-v2 44.1kHz（重采样到 48k）：原生带宽约 22kHz，机械音大幅消除、更自然；但权重 451MB、推理更慢。",
  },
};

const inputPath = ref("");
const modelId = ref(HIFIGAN_MODEL_ID);
const device = ref("auto");
const sampleRate = ref(48000);

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
  if (id.startsWith("hifigan")) return "hifigan";
  return "";
}

// 仅暴露 HiFi-GAN 后端模型（本视图固定声码器档位）。
const modelOptions = computed(() => {
  const list = models.value.length
    ? models.value
    : [{ id: HIFIGAN_MODEL_ID, backend: "hifigan" }];
  return list
    .filter((m) => m.backend === "hifigan")
    .map((m) => {
      const available = availableModelIds.value.has(m.id);
      const tier = MODEL_TIERS[m.id];
      const label = tier
        ? `${m.id} · ${tier.tier}${available ? " · 可用" : " · 不可用"}`
        : `${m.id}${available ? " · 可用" : " · 不可用"}`;
      return {
        label,
        value: m.id,
        title: tier ? `${tier.tier}：${tier.desc}` : "",
      };
    });
});

const modelHint = computed(() => {
  const tier = MODEL_TIERS[modelId.value];
  if (tier) return `${tier.tier}：${tier.desc}`;
  return "";
});

const deviceOptions = [
  { label: "自动", value: "auto" },
  { label: "CPU", value: "cpu" },
  { label: "CUDA", value: "cuda" },
];

// HiFi-GAN 固定输出 48 kHz，锁定采样率并提示。
const isFixed48k = computed(() => backendOf(modelId.value) === "hifigan");

const sampleRateOptions = computed(() =>
  isFixed48k.value
    ? [{ label: "48000 Hz（模型固定 · 22.05k 重采样）", value: 48000 }]
    : [
        { label: "48000 Hz", value: 48000 },
        { label: "44100 Hz", value: 44100 },
      ]
);

watch(isFixed48k, (locked) => {
  if (locked) sampleRate.value = 48000;
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
    // 本视图固定 hifigan-48k：即使该模型尚未安装也不回退到其它后端，
    // 保留 hifigan-48k 作为选择项以便用户点击「安装模型」下载权重。
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
      title: "选择要优化音质的音频文件",
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
    const task = await invoke<QualityTask>("audio_quality_start", {
      input: {
        inputPath: inputPath.value,
        modelId: modelId.value,
        device: device.value,
        outputSampleRate: sampleRate.value,
      },
    });
    activeTask.value = task;
    if (!tasks.value.some((t) => t.id === task.id)) {
      tasks.value = [task, ...tasks.value];
    }
    message.success("已开始音质优化");
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
    <a-card title="音质优化" :bordered="false" class="main-card">
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

      <a-alert
        v-if="runtime"
        class="msg"
        type="info"
        show-icon
        message="音质优化提供两档：快速/轻量（hifigan-48k，原 22.05k 重建，机械音偏重、推理快）与 高质量/增强（bigvgan-48k，BigVGAN-v2 44.1k，机械音大幅消除、更自然，但权重 451MB、推理更慢）。按需求选择。"
      />

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
          <a-col :span="8">
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
                </span>
              </a-space>
            </a-form-item>
          </a-col>
          <a-col :span="8">
            <a-form-item label="计算设备">
              <a-select
                v-model:value="device"
                :options="deviceOptions"
                style="min-width: 120px"
              />
            </a-form-item>
          </a-col>
          <a-col :span="8">
            <a-form-item label="输出采样率">
              <a-select
                v-model:value="sampleRate"
                :options="sampleRateOptions"
                :disabled="isFixed48k"
                style="min-width: 120px"
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
            开始优化
          </a-button>
          <a-button
            v-if="running"
            danger
            @click="cancel((tasks.find((t) => t.status === 'processing' || t.status === 'queued') as QualityTask).id)"
          >
            取消任务
          </a-button>
        </a-space>
        <div class="hint">输出文件将自动生成在源文件旁（扩展名 .ai.flac），不会覆盖原文件。两档均为神经声码器，重建波形而非提升已丢失的高频：快速档权重原生 22050 Hz、重建后重采样至 48000 Hz（机械音偏重）；高质量档 BigVGAN-v2 原生 44.1kHz、重采样至 48000 Hz（更自然）。</div>
      </a-form>

      <a-empty
        v-if="!tasks.length"
        description="暂无音质优化任务"
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
