<script setup lang="ts">
import { h, ref } from "vue";
import DownloadView from "./DownloadView.vue";
import RecognizerView from "./RecognizerView.vue";
import HistoryView from "./HistoryView.vue";
import {
  DownloadOutlined,
  AudioOutlined,
  HistoryOutlined,
} from "@ant-design/icons-vue";

type ViewKey = "download" | "recognize" | "history" | "download-history";
const active = ref<ViewKey>("download");
// 子菜单展开状态（受控）
const openKeys = ref<string[]>([]);

// antd Menu 的 items：音频识别与 B站下载均作为父项，各自下挂「操作」「历史记录」子项
const items = [
  {
    key: "download-group",
    icon: h(DownloadOutlined),
    label: "B站下载",
    children: [
      { key: "download", icon: h(DownloadOutlined), label: "下载" },
      { key: "download-history", icon: h(HistoryOutlined), label: "历史记录" },
    ],
  },
  {
    key: "recognize-group",
    icon: h(AudioOutlined),
    label: "音频识别",
    children: [
      { key: "recognize", icon: h(AudioOutlined), label: "识别" },
      { key: "history", icon: h(HistoryOutlined), label: "历史记录" },
    ],
  },
];

function onMenuClick({ key }: { key: string }) {
  // 仅子项（无 children）才切换主视图
  if (key === "recognize" || key === "history" || key === "download" || key === "download-history") {
    active.value = key as ViewKey;
  }
}

function onOpenChange(keys: string[]) {
  openKeys.value = keys;
}
</script>

<template>
  <a-layout class="layout">
    <a-layout-sider :width="220" theme="dark" class="sider">
      <div class="brand">
        <span class="logo">♪</span>
        <span class="brand-name">Audio Processor</span>
      </div>
      <a-menu
        :selectedKeys="[active]"
        :openKeys="openKeys"
        theme="dark"
        mode="inline"
        :items="items"
        @click="onMenuClick"
        @openChange="onOpenChange"
      />
    </a-layout-sider>

    <a-layout-content class="content">
      <download-view v-if="active === 'download'" />
      <recognizer-view v-else-if="active === 'recognize'" />
      <history-view v-else-if="active === 'history'" kind="recognize" />
      <history-view v-else-if="active === 'download-history'" kind="download" />
    </a-layout-content>
  </a-layout>
</template>

<style scoped>
.layout {
  height: 100vh;
  width: 100vw;
}
.sider {
  overflow: auto;
}
.brand {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  height: 64px;
  padding: 0 1.2rem;
  font-size: 1.05rem;
  font-weight: 700;
  color: #fff;
  border-bottom: 1px solid #303030;
}
.logo {
  color: #38bdf8;
  font-size: 1.3rem;
}
.content {
  background: #f5f7fa;
  overflow: hidden;
}
</style>
