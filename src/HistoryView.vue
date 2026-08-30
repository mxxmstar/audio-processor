<script setup lang="ts">
import { computed, onMounted, ref } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { message, Modal } from "ant-design-vue";
import { HistoryOutlined, DeleteOutlined, FolderOpenOutlined } from "@ant-design/icons-vue";

interface HistoryItem {
  id: number;
  kind: string;
  title: string;
  subtitle: string;
  /** 业务详情的 JSON 字符串（识别为 SongInfo、下载为 DownloadTask） */
  payload: string;
  /** 关联文件本地绝对路径（下载类记录有值，识别类为空） */
  file_path: string;
  created_at: string;
}

interface TaskGroup {
  id: string;
  title: string;
}

interface DownloadPayload {
  title: string;
  group: TaskGroup | null;
  [k: string]: unknown;
}

interface GroupView {
  key: string;
  title: string;
  items: HistoryItem[];
}

type KindFilter = "recognize" | "download" | "aria2" | "enhance";

const props = defineProps<{ kind: KindFilter }>();

const kindLabels: Record<string, string> = {
  recognize: "音频识别",
  download: "B站下载",
  aria2: "aria2下载",
  enhance: "音质提升",
};

const records = ref<HistoryItem[]>([]);
const loading = ref(false);
const detail = ref<HistoryItem | null>(null);
const detailOpen = ref(false);
const activeGroupKeys = ref<string[]>(["single"]);

// 勾选状态：以历史记录 id 为键的集合
const selectedIds = ref<Set<number>>(new Set());

// 扁平化记录顺序（合集分组后的展示顺序），用于 shift 连续多选的索引定位
const orderedRecordIds = computed<number[]>(() =>
  groupedRecords.value.flatMap((g) => g.items.map((r) => r.id))
);
// 上一次点击的记录索引（shift 多选的锚点）
const lastRecordIndex = ref(-1);
// shift 框选的「蓝色临时选中」集合（尚未真正勾选，点击任意勾选框后批量应用）
const rangeIds = ref<Set<number>>(new Set());

const selectedCount = computed(() => selectedIds.value.size);
const totalCount = computed(() => records.value.length);

function isSelected(id: number): boolean {
  return selectedIds.value.has(id);
}

function inRange(id: number): boolean {
  return rangeIds.value.has(id);
}

// 记录点击交互（同下载列表）：
// - 普通点击 = 切换单个；若存在蓝色框选区，则把点击状态批量应用到整个框选区后清除
// - shift+点击 = 以锚点为起点框选 [锚点, 当前] 区间到 rangeIds（仅高亮，不立即勾选）
function onRecordClick(id: number, shift: boolean) {
  const flat = orderedRecordIds.value;
  const idx = flat.indexOf(id);
  if (shift) {
    if (lastRecordIndex.value < 0) lastRecordIndex.value = idx;
    const [a, b] = idx >= lastRecordIndex.value
      ? [lastRecordIndex.value, idx]
      : [idx, lastRecordIndex.value];
    const s = new Set<number>();
    for (let i = a; i <= b; i++) s.add(flat[i]);
    rangeIds.value = s;
    return;
  }
  if (rangeIds.value.size > 0) {
    const target = !isSelected(id);
    const s = new Set(selectedIds.value);
    for (const rid of rangeIds.value) {
      if (target) s.add(rid);
      else s.delete(rid);
    }
    selectedIds.value = s;
    rangeIds.value = new Set();
    lastRecordIndex.value = idx;
    return;
  }
  const s = new Set(selectedIds.value);
  if (s.has(id)) s.delete(id);
  else s.add(id);
  selectedIds.value = s;
  lastRecordIndex.value = idx;
}

function selectAll() {
  selectedIds.value = new Set(records.value.map((r) => r.id));
  rangeIds.value = new Set();
}

function clearSelection() {
  selectedIds.value = new Set();
  rangeIds.value = new Set();
}

// 历史记录按合集分组（仅 download 类型带 group 信息）
const groupedRecords = computed<GroupView[]>(() => {
  const map = new Map<string, GroupView>();
  for (const rec of records.value) {
    let key = "single";
    let title = "单条记录";
    if (props.kind === "download") {
      try {
        const p = JSON.parse(rec.payload) as DownloadPayload;
        if (p.group && p.group.id) {
          key = `g:${p.group.id}`;
          title = p.group.title || `合集 ${p.group.id}`;
        }
      } catch {
        // 解析失败则归入单条
      }
    }
    if (!map.has(key)) {
      map.set(key, { key, title, items: [] });
    }
    map.get(key)!.items.push(rec);
  }
  return Array.from(map.values()).sort((a, b) => {
    if (a.key === "single") return 1;
    if (b.key === "single") return -1;
    return 0;
  });
});

async function load() {
  loading.value = true;
  try {
    records.value = await invoke<HistoryItem[]>("get_history", {
      kind: props.kind,
      limit: 200,
    });
    activeGroupKeys.value = groupedRecords.value.map((g) => g.key);
    rangeIds.value = new Set();
  } catch (e) {
    message.error("加载历史失败：" + String(e));
  } finally {
    loading.value = false;
  }
}

async function remove(id: number) {
  try {
    await invoke("delete_history", { id });
    message.success("已删除");
    await load();
  } catch (e) {
    message.error("删除失败：" + String(e));
  }
}

// 批量删除：删除当前所有勾选的记录
async function removeSelected() {
  if (selectedIds.value.size === 0) return;
  const ids = Array.from(selectedIds.value);
  loading.value = true;
  try {
    for (const id of ids) {
      await invoke("delete_history", { id });
    }
    message.success(`已删除 ${ids.length} 条记录`);
    clearSelection();
    await load();
  } catch (e) {
    message.error("批量删除失败：" + String(e));
  } finally {
    loading.value = false;
  }
}

function view(rec: HistoryItem) {
  detail.value = rec;
  detailOpen.value = true;
}

async function openDir(rec: HistoryItem) {
  if (!rec.file_path) return;
  try {
    await invoke("open_path", { path: rec.file_path });
  } catch (e) {
    const msg = String(e);
    // 文件已被删除时弹出明确提醒
    if (msg.includes("文件不存在")) {
      Modal.warning({
        title: "文件不存在",
        content: `该记录对应的文件已被删除或移动：\n${rec.file_path}`,
      });
    } else {
      message.error("打开目录失败：" + msg);
    }
  }
}

function payloadText(json: string): string {
  try {
    return JSON.stringify(JSON.parse(json), null, 2);
  } catch {
    return json;
  }
}

function kindColor(kind: string): string {
  switch (kind) {
    case "download":
      return "blue";
    case "aria2":
      return "cyan";
    case "enhance":
      return "purple";
    default:
      return "green";
  }
}

onMounted(load);
</script>

<template>
  <div class="panel">
    <a-card :title="`${kindLabels[props.kind]}历史`" :bordered="false" class="main-card">
      <template #extra>
        <a-space>
          <a-button size="small" :loading="loading" @click="load">刷新</a-button>
        </a-space>
      </template>

      <a-space
        v-if="records.length"
        class="select-bar"
        wrap
      >
        <a-checkbox
          :checked="selectedCount === totalCount && totalCount > 0"
          :indeterminate="selectedCount > 0 && selectedCount < totalCount"
          @click="(e: MouseEvent) => (e.shiftKey ? undefined : (selectedCount === totalCount ? clearSelection() : selectAll()))"
        >
          全选
        </a-checkbox>
        <a-button size="small" @click="selectAll">全部选择</a-button>
        <a-button size="small" @click="clearSelection">全部取消</a-button>
        <a-button
          size="small"
          danger
          :disabled="selectedCount === 0"
          @click="removeSelected"
        >
          批量删除（{{ selectedCount }}）
        </a-button>
        <span class="sel-count">已选 {{ selectedCount }} / {{ totalCount }}</span>
        <span class="sel-tip">提示：按住 Shift 点击可批量选择连续项</span>
      </a-space>

      <a-spin :spinning="loading">
        <a-empty v-if="!records.length && !loading" description="暂无历史记录" />

        <a-collapse
          v-else
          v-model:activeKey="activeGroupKeys"
          class="hist-collapse"
          :bordered="false"
        >
          <a-collapse-panel
            v-for="grp in groupedRecords"
            :key="grp.key"
          >
            <template #header>
              <span class="grp-header">
                <a-tag :color="grp.key === 'single' ? 'default' : 'purple'">
                  {{ grp.key === "single" ? "单条" : "合集" }}
                </a-tag>
                <span class="grp-title">{{ grp.title }}</span>
                <span class="grp-count">（{{ grp.items.length }} 条）</span>
              </span>
            </template>
            <a-list :data-source="grp.items" item-layout="horizontal" class="hist-list">
              <template #renderItem="{ item }">
                <a-list-item
                  :class="{ 'is-selected': isSelected(item.id), 'in-range': inRange(item.id) }"
                  @click="(e: MouseEvent) => onRecordClick(item.id, e.shiftKey)"
                >
                  <a-checkbox
                    class="rec-check"
                    :checked="isSelected(item.id)"
                    @click.stop="(e: MouseEvent) => onRecordClick(item.id, e.shiftKey)"
                  />
                  <a-list-item-meta>
                    <template #title>
                      <a-tag :color="kindColor(item.kind)">
                        {{ kindLabels[item.kind] ?? item.kind }}
                      </a-tag>
                      <a-typography-text strong>{{ item.title }}</a-typography-text>
                    </template>
                    <template #description>
                      <span class="meta">
                        {{ item.subtitle || "—" }} · {{ item.created_at }}
                      </span>
                    </template>
                    <template #avatar>
                      <a-avatar><HistoryOutlined /></a-avatar>
                    </template>
                  </a-list-item-meta>
                  <template #actions>
                    <a-button type="link" size="small" @click="view(item)">查看</a-button>
                    <a-button
                      v-if="item.file_path"
                      type="link"
                      size="small"
                      @click="openDir(item)"
                    >
                      <template #icon><FolderOpenOutlined /></template>
                      打开目录
                    </a-button>
                    <a-popconfirm title="确认删除这条记录？" @confirm="remove(item.id)">
                      <a-button type="link" size="small" danger>
                        <template #icon><DeleteOutlined /></template>
                        删除
                      </a-button>
                    </a-popconfirm>
                  </template>
                </a-list-item>
              </template>
            </a-list>
          </a-collapse-panel>
        </a-collapse>
      </a-spin>
    </a-card>

    <a-drawer
      v-model:open="detailOpen"
      title="记录详情"
      width="460"
      :footer="null"
    >
      <a-descriptions v-if="detail" :column="1" bordered size="small">
        <a-descriptions-item label="类型">
          {{ kindLabels[detail.kind] ?? detail.kind }}
        </a-descriptions-item>
        <a-descriptions-item label="标题">
          {{ detail.title }}
        </a-descriptions-item>
        <a-descriptions-item label="副标题">
          {{ detail.subtitle || "—" }}
        </a-descriptions-item>
        <a-descriptions-item label="记录时间">
          {{ detail.created_at }}
        </a-descriptions-item>
        <a-descriptions-item label="详情">
          <pre class="payload">{{ payloadText(detail.payload) }}</pre>
        </a-descriptions-item>
      </a-descriptions>
    </a-drawer>
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
.hist-list {
  margin-top: 0.5rem;
}
.hist-collapse {
  margin-top: 0.5rem;
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
.meta {
  font-size: 0.8rem;
  color: #8a94a6;
}
.sel-count {
  font-size: 0.8rem;
  color: #8a94a6;
}
.sel-tip {
  font-size: 0.8rem;
  color: #b0b8c4;
}
.select-bar {
  margin-bottom: 0.5rem;
}
.rec-check {
  flex: 0 0 auto;
  margin-right: 0.6rem;
  align-self: flex-start;
  margin-top: 0.2rem;
}
.hist-list :deep(.ant-list-item) {
  user-select: none;
}
.hist-list :deep(.ant-list-item.is-selected) {
  background: #e6e6e6;
  border-radius: 6px;
}
.hist-list :deep(.ant-list-item.in-range) {
  background: rgba(24, 144, 255, 0.18);
  box-shadow: inset 0 0 0 2px #1890ff;
  border-radius: 6px;
}
.payload {
  max-height: 50vh;
  overflow: auto;
  white-space: pre-wrap;
  word-break: break-all;
  font-size: 0.75rem;
  background: #f5f5f5;
  padding: 0.5rem;
  border-radius: 4px;
  margin: 0;
}
</style>
