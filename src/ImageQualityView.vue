<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { message, Modal } from "ant-design-vue";
import {
  PictureOutlined,
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

interface ImageQualityTask {
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
  width: number | null;
  height: number | null;
  scale: number | null;
  format: string | null;
}

interface ModelInfo {
  id: string;
  backend: string;
  scale?: number;
}

const inputPath = ref("");
const modelId = ref("realesrgan-x4plus");
const device = ref("cpu");
const format = ref<"png" | "jpg">("png");
const outscale = ref<string>("");
const jpgQuality = ref(95);

const tasks = ref<ImageQualityTask[]>([]);
const models = ref<ModelInfo[]>([]);
const runtime = ref<AiRuntimeCheck | null>(null);
const checking = ref(false);
const starting = ref(false);
const installing = ref(false);
const activeTask = ref<ImageQualityTask | null>(null);

const running = computed(() =>
  tasks.value.some((t) => t.status === "queued" || t.status === "processing")
);

const availableModelIds = computed(() => new Set(runtime.value?.models ?? []));

// 模型清单由后端 list_models 驱动（Real-ESRGAN + SwinIR 等超分后端）。
const modelTitle: Record<string, string> = {
  "realesrgan-x4plus": "Real-ESRGAN 通用 4× 超分（最常用）",
  "realesrgan-x2plus": "Real-ESRGAN 通用 2× 超分",
  "realesrgan-x4plus-anime": "Real-ESRGAN 动漫 4× 超分",
  "swinir-classical-x2": "SwinIR 经典 2× 超分（DIV2K 训练）",
  "swinir-classical-x3": "SwinIR 经典 3× 超分（唯一原生 3×）",
  "swinir-classical-x4": "SwinIR 经典 4× 超分（DIV2K 训练）",
  "swinir-real-x4": "SwinIR 真实世界 4× 超分（BSRGAN 退化）",
  "gfpgan-v1.4": "GFPGAN 人脸修复 2× 修复（通用）",
  "realesr-general-x4v3": "Real-ESRGAN 通用盲超分 4× 背景修复（x4v3，真实退化）",
};
const modelOptions = computed(() => {
  const list = models.value.length
    ? models.value
    : [{ id: modelId.value, backend: "realesrgan", scale: 4 }];
  return list.map((m) => {
    const available = availableModelIds.value.has(m.id);
    const backendName = m.backend === "swinir" ? "SwinIR" : "Real-ESRGAN";
    const scaleText = m.scale ? `${m.scale}×` : "";
    return {
      label: `${m.id}${available ? " · 可用" : " · 不可用"}`,
      value: m.id,
      title: modelTitle[m.id] ?? `${backendName} ${scaleText} 超分`,
    };
  });
});

const deviceOptions = [
  { label: "自动", value: "auto" },
  { label: "CPU", value: "cpu" },
  { label: "CUDA", value: "cuda" },
];

const formatOptions = [
  { label: "PNG（无损）", value: "png" },
  { label: "JPG（有损，可调质量）", value: "jpg" },
];

// 放大倍数：留空 = 模型原生倍数；具体倍数 = 以该倍数输出（模型先超分再缩放）。
// “模型默认”标签随当前模型原生倍数动态显示，便于直观了解默认结果；
// 切换模型时重置为“模型默认”，避免沿用上次倍数导致意外输出。
const selectedModelScale = computed(() => {
  const m = models.value.find((x) => x.id === modelId.value);
  return m?.scale ?? 0;
});
const scaleOptions = computed(() => {
  const def = selectedModelScale.value
    ? `模型默认 (${selectedModelScale.value}×)`
    : "模型默认";
  return [
    { label: def, value: "" },
    { label: "2×", value: "2" },
    { label: "3×", value: "3" },
    { label: "4×", value: "4" },
    { label: "8×", value: "8" },
  ];
});

// 切换模型时重置放大倍数为“模型默认”，避免沿用上一次选择导致意外结果。
watch(modelId, () => {
  outscale.value = "";
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

// 依据格式推导建议输出扩展名（Rust 端按 outputPath 后缀决定 png/jpg）。
function deriveOutputPath(src: string, ext: string): string {
  const base = src.replace(/\.[^./\\]+$/, "");
  return `${base}.${ext}`;
}

async function checkRuntime() {
  checking.value = true;
  try {
    runtime.value = await invoke<AiRuntimeCheck>("enhance_image_check_ai_runtime", {
      modelId: modelId.value,
    });
    const available = runtime.value.models;
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
    models.value = await invoke<ModelInfo[]>("enhance_image_list_models");
  } catch {
    models.value = [];
  }
}

async function pickFile() {
  try {
    const sel = await open({
      multiple: false,
      title: "选择要提升画质的图片",
      filters: [
        {
          name: "图片",
          extensions: ["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
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
    message.warning("请先选择图片文件");
    return;
  }
  starting.value = true;
  try {
    // EnhanceImageInput 使用 serde rename_all = "camelCase"（inputPath / modelId / jpgQuality）
    const payload: Record<string, unknown> = {
      inputPath: inputPath.value,
      modelId: modelId.value,
      device: device.value,
      outputPath: deriveOutputPath(inputPath.value, format.value),
    };
    if (format.value === "jpg") {
      payload.jpgQuality = jpgQuality.value;
    }
    if (outscale.value) {
      payload.outscale = Number(outscale.value);
    }
    const task = await invoke<ImageQualityTask>("enhance_image", { input: payload });
    activeTask.value = task;
    // 只保留当前记录：用新任务替换历史列表，不累积过往任务
    tasks.value = [task];
    message.success("已开始画质提升");
  } catch (e) {
    message.error("启动失败：" + String(e));
  } finally {
    starting.value = false;
  }
}

async function cancel(taskId: string) {
  try {
    await invoke("enhance_image_cancel", { task_id: taskId });
    message.success("已请求取消");
  } catch (e) {
    message.error("取消失败：" + String(e));
  }
}

async function installModel() {
  installing.value = true;
  try {
    // 注意：该命令声明为 `#[tauri::command(rename_all = "snake_case")]`，
    // 参数名必须是 snake_case（model_id），与 `enhance_image_cancel` 的
    // `task_id` 一致；写成 modelId 会报 missing required key model_id。
    const r = await invoke<{ modelId: string; status: string; message: string }>(
      "enhance_image_download_model",
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
  await Promise.all([loadModels(), checkRuntime()]);
  off = await listen<ImageQualityTask>("image-quality-progress", (e) => {
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

onUnmounted(() => {
  off?.();
});
</script>

<template>
  <div class="panel">
    <a-card title="图像画质提升（AI 超分）" :bordered="false" class="main-card">
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
        <a-form-item label="输入图片">
          <a-space wrap>
            <a-button @click="pickFile">
              <template #icon><FileSearchOutlined /></template>
              选择图片文件
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
            <a-form-item label="输出格式">
              <a-select
                v-model:value="format"
                :options="formatOptions"
                style="min-width: 160px"
              />
            </a-form-item>
          </a-col>
        </a-row>

        <a-form-item label="放大倍数">
          <a-select
            v-model:value="outscale"
            :options="scaleOptions"
            style="min-width: 160px"
          />
          <span class="dim" style="margin-left: 0.5rem">留空 = 模型原生倍数输出（Real-ESRGAN 4× / SwinIR 标注倍数）；选具体倍数 = 以该倍数输出（先超分再缩放）。</span>
        </a-form-item>

        <a-form-item v-if="format === 'jpg'" label="JPG 质量">
          <a-slider v-model:value="jpgQuality" :min="40" :max="100" :step="1" style="max-width: 320px" />
          <span class="dim" style="margin-left: 0.5rem">{{ jpgQuality }}</span>
        </a-form-item>

        <a-space>
          <a-button
            type="primary"
            :loading="starting"
            :disabled="!inputPath || running"
            @click="start"
          >
            <template #icon><PictureOutlined /></template>
            开始超分
          </a-button>
          <a-button
            v-if="running"
            danger
            @click="cancel((tasks.find((t) => t.status === 'processing' || t.status === 'queued') as ImageQualityTask).id)"
          >
            取消任务
          </a-button>
        </a-space>
        <div class="hint">
          输出文件自动生成在源文件旁（默认 .png / 选 JPG 时 .jpg，带尺寸去重），不覆盖原图；放大倍数留空时按模型原生倍数（Real-ESRGAN 4× / SwinIR 2×·3×·4×），也可手动选择 2× / 3× / 4× / 8× 直接以该倍数输出。
        </div>
      </a-form>

      <a-empty
        v-if="!tasks.length"
        description="暂无画质提升任务"
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
                  <span v-if="item.width" class="dim">
                    {{ item.width }} × {{ item.height }}
                    <template v-if="item.scale"> · {{ item.scale }}×</template>
                    <template v-if="item.format"> · {{ item.format.toUpperCase() }}</template>
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
