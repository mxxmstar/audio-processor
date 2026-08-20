<script setup lang="ts">
import { h, ref } from "vue";
import DownloadView from "./DownloadView.vue";
import RecognizerView from "./RecognizerView.vue";
import {
  DownloadOutlined,
  AudioOutlined,
} from "@ant-design/icons-vue";

const active = ref<"download" | "recognize">("download");
const items = [
  { key: "download", icon: DownloadOutlined, label: "B站下载" },
  { key: "recognize", icon: AudioOutlined, label: "音频识别" },
];
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
        theme="dark"
        mode="inline"
        :items="items.map((i) => ({ key: i.key, icon: h(i.icon), label: i.label }))"
        @click="({ key }) => (active = key as 'download' | 'recognize')"
      />
    </a-layout-sider>

    <a-layout-content class="content">
      <download-view v-if="active === 'download'" />
      <recognizer-view v-else />
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
