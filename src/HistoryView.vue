<script setup lang="ts">
import { onMounted, ref } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { message } from "ant-design-vue";
import { HistoryOutlined, DeleteOutlined } from "@ant-design/icons-vue";

interface HistoryItem {
  id: number;
  kind: string;
  title: string;
  subtitle: string;
  /** 业务详情的 JSON 字符串（识别为 SongInfo、下载为 DownloadTask） */
  payload: string;
  created_at: string;
}

type KindFilter = "all" | "recognize" | "download";

const kindLabels: Record<string, string> = {
  recognize: "音频识别",
  download: "B站下载",
};

const activeKind = ref<KindFilter>("all");
const records = ref<HistoryItem[]>([]);
const loading = ref(false);
const detail = ref<HistoryItem | null>(null);
const detailOpen = ref(false);

async function load() {
  loading.value = true;
  try {
    const kind = activeKind.value === "all" ? null : activeKind.value;
    records.value = await invoke<HistoryItem[]>("get_history", {
      kind,
      limit: 200,
    });
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

function view(rec: HistoryItem) {
  detail.value = rec;
  detailOpen.value = true;
}

function payloadText(json: string): string {
  try {
    return JSON.stringify(JSON.parse(json), null, 2);
  } catch {
    return json;
  }
}

onMounted(load);
</script>

<template>
  <div class="panel">
    <a-card title="历史记录" :bordered="false" class="main-card">
      <template #extra>
        <a-space>
          <a-segmented
            v-model:value="activeKind"
            :options="[
              { label: '全部', value: 'all' },
              { label: '音频识别', value: 'recognize' },
              { label: 'B站下载', value: 'download' },
            ]"
            @change="load"
          />
          <a-button size="small" :loading="loading" @click="load">刷新</a-button>
        </a-space>
      </template>

      <a-spin :spinning="loading">
        <a-empty v-if="!records.length && !loading" description="暂无历史记录" />

        <a-list v-else :data-source="records" item-layout="horizontal" class="hist-list">
          <template #renderItem="{ item }">
            <a-list-item>
              <a-list-item-meta>
                <template #title>
                  <a-tag :color="item.kind === 'download' ? 'blue' : 'green'">
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
.meta {
  font-size: 0.8rem;
  color: #8a94a6;
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
