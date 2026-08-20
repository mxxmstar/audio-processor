<script setup lang="ts">
import { onMounted, ref } from "vue";
import { invoke } from "@tauri-apps/api/core";
import { message } from "ant-design-vue";
import { HistoryOutlined, DeleteOutlined } from "@ant-design/icons-vue";

interface HistoryRecord {
  id: number;
  title: string;
  artist: string;
  album: string | null;
  album_date: string | null;
  confidence: number;
  file_path: string;
  created_at: string;
}

const records = ref<HistoryRecord[]>([]);
const loading = ref(false);
const detail = ref<HistoryRecord | null>(null);
const detailOpen = ref(false);

async function load() {
  loading.value = true;
  try {
    records.value = await invoke<HistoryRecord[]>("get_recognize_history", {
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
    await invoke("delete_recognize_record", { id });
    message.success("已删除");
    await load();
  } catch (e) {
    message.error("删除失败：" + String(e));
  }
}

function view(rec: HistoryRecord) {
  detail.value = rec;
  detailOpen.value = true;
}

onMounted(load);
</script>

<template>
  <div class="panel">
    <a-card title="识别历史" :bordered="false" class="main-card">
      <template #extra>
        <a-button size="small" :loading="loading" @click="load">
          刷新
        </a-button>
      </template>

      <a-spin :spinning="loading">
        <a-empty v-if="!records.length && !loading" description="暂无识别记录" />

        <a-list v-else :data-source="records" item-layout="horizontal" class="hist-list">
          <template #renderItem="{ item }">
            <a-list-item>
              <a-list-item-meta>
                <template #title>
                  <a-typography-text strong>{{ item.title }}</a-typography-text>
                  <span class="artist"> · {{ item.artist }}</span>
                </template>
                <template #description>
                  <span class="meta">
                    {{ item.album ?? "未知专辑" }} · 置信度
                    {{ item.confidence.toFixed(1) }}% · {{ item.created_at }}
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
      title="识别详情"
      width="420"
      :footer="null"
    >
      <a-descriptions v-if="detail" :column="1" bordered size="small">
        <a-descriptions-item label="标题">{{ detail.title }}</a-descriptions-item>
        <a-descriptions-item label="艺术家">{{ detail.artist }}</a-descriptions-item>
        <a-descriptions-item label="专辑">
          {{ detail.album ?? "—" }}
        </a-descriptions-item>
        <a-descriptions-item label="发行日期">
          {{ detail.album_date ?? "—" }}
        </a-descriptions-item>
        <a-descriptions-item label="置信度">
          {{ detail.confidence.toFixed(1) }}%
        </a-descriptions-item>
        <a-descriptions-item label="识别时间">
          {{ detail.created_at }}
        </a-descriptions-item>
        <a-descriptions-item label="文件路径">
          <a-typography-text :ellipsis="{ tooltip: detail.file_path }">
            {{ detail.file_path }}
          </a-typography-text>
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
.artist {
  color: #6b7280;
}
.meta {
  font-size: 0.8rem;
  color: #8a94a6;
}
</style>
