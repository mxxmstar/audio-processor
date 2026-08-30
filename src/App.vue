<script setup lang="ts">
import { h, onMounted, onUnmounted, ref } from "vue";
import { invoke } from "@tauri-apps/api/core";
import DownloadView from "./DownloadView.vue";
import RecognizerView from "./RecognizerView.vue";
import HistoryView from "./HistoryView.vue";
import Aria2View from "./Aria2View.vue";
import QualityView from "./QualityView.vue";
import PortCheckView from "./PortCheckView.vue";
import {
  DownloadOutlined,
  AudioOutlined,
  HistoryOutlined,
  QrcodeOutlined,
  LogoutOutlined,
  UserOutlined,
  CloudDownloadOutlined,
  SoundOutlined,
  ApiOutlined,
} from "@ant-design/icons-vue";

type ViewKey =
  | "download"
  | "recognize"
  | "history"
  | "download-history"
  | "aria2-download"
  | "aria2-history"
  | "quality"
  | "quality-history"
  | "port-check";
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
    key: "aria2-group",
    icon: h(CloudDownloadOutlined),
    label: "aria2 下载",
    children: [
      { key: "aria2-download", icon: h(CloudDownloadOutlined), label: "下载" },
      { key: "aria2-history", icon: h(HistoryOutlined), label: "历史记录" },
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
  {
    key: "quality-group",
    icon: h(SoundOutlined),
    label: "音频品质提升",
    children: [
      { key: "quality", icon: h(SoundOutlined), label: "音质提升" },
      { key: "quality-history", icon: h(HistoryOutlined), label: "历史记录" },
    ],
  },
  {
    key: "port-check",
    icon: h(ApiOutlined),
    label: "端口占用",
  },
];

// 可切换主视图的子项（父分组项不参与切换）
const leafKeys: string[] = [
  "recognize",
  "history",
  "download",
  "download-history",
  "aria2-download",
  "aria2-history",
  "quality",
  "quality-history",
  "port-check",
];

function onMenuClick({ key }: { key: string }) {
  // 仅子项（无 children）才切换主视图
  if (leafKeys.includes(key)) {
    active.value = key as ViewKey;
  }
}

function onOpenChange(keys: string[]) {
  openKeys.value = keys;
}

// ---- 登录态（提升到左侧栏统一管理）----
const loggedIn = ref(false);
const checkingLogin = ref(true);
const showQr = ref(false);
const qr = ref<{ qr_svg: string; qr_key: string } | null>(null);
const userInfo = ref<{ name: string; face: string } | null>(null);
const imgError = ref(false);
let qrPolling: number | null = null;

async function loadUserInfo() {
  try {
    userInfo.value = await invoke<{ name: string; face: string } | null>(
      "bili_user_info"
    );
  } catch {
    userInfo.value = null;
  }
}

async function refreshLogin() {
  checkingLogin.value = true;
  try {
    loggedIn.value = await invoke<boolean>("bili_check_login");
    if (loggedIn.value) {
      await loadUserInfo();
    }
  } catch {
    loggedIn.value = false;
  } finally {
    checkingLogin.value = false;
  }
}

async function openQr() {
  showQr.value = true;
  try {
    qr.value = await invoke<{ qr_svg: string; qr_key: string }>("bili_login_qr");
    startPoll();
  } catch (e) {
    console.error("生成二维码失败：" + String(e));
  }
}

function startPoll() {
  stopPoll();
  qrPolling = setInterval(async () => {
    if (!qr.value) return;
    try {
      const r = await invoke<{ authed: boolean; message: string }>(
        "bili_login_poll",
        { qrKey: qr.value.qr_key }
      );
      if (r.authed) {
        loggedIn.value = true;
        showQr.value = false;
        stopPoll();
        qr.value = null;
        await loadUserInfo();
      }
    } catch {
      /* 继续轮询 */
    }
  }, 2000) as unknown as number;
}

function stopPoll() {
  if (qrPolling !== null) {
    clearInterval(qrPolling);
    qrPolling = null;
  }
}

async function doLogout() {
  await invoke("bili_logout");
  loggedIn.value = false;
  userInfo.value = null;
}

onMounted(refreshLogin);
onUnmounted(stopPoll);
</script>

<template>
  <a-layout class="layout">
    <a-layout-sider :width="220" theme="dark" class="sider">
      <!-- 登录信息区（置于侧边栏顶部） -->
      <div class="login-box">
        <a-skeleton v-if="checkingLogin" active :paragraph="false" :title="false" />
        <template v-else-if="!loggedIn">
          <a-button
            type="primary"
            block
            size="small"
            @click="openQr"
          >
            <template #icon><QrcodeOutlined /></template>
            扫码登录
          </a-button>
        </template>
        <a-space v-else direction="vertical" :size="8" class="login-info" style="width: 100%">
          <a-space align="center" :size="8">
            <img
              v-if="userInfo?.face && !imgError"
              :src="userInfo.face"
              referrerpolicy="no-referrer"
              class="avatar-img"
              width="36"
              height="36"
              alt="avatar"
              @error="imgError = true"
            />
            <a-avatar v-else :size="36">
              <template #icon><UserOutlined /></template>
            </a-avatar>
            <span class="login-text">{{ userInfo?.name || "已登录" }}</span>
          </a-space>
          <a-button
            class="logout-btn"
            type="text"
            size="small"
            block
            @click="doLogout"
          >
            <template #icon><LogoutOutlined /></template>
            退出登录
          </a-button>
        </a-space>
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
      <!-- keep-alive：缓存各视图组件实例，切换界面时不销毁，
           保留下载任务列表/进度、识别结果等状态与事件监听 -->
      <keep-alive>
        <download-view
          v-if="active === 'download'"
          key="view-download"
          :logged-in="loggedIn"
        />
        <recognizer-view v-else-if="active === 'recognize'" key="view-recognize" />
        <history-view
          v-else-if="active === 'history'"
          key="view-history-recognize"
          kind="recognize"
        />
        <history-view
          v-else-if="active === 'download-history'"
          key="view-history-download"
          kind="download"
        />
        <aria2-view
          v-else-if="active === 'aria2-download'"
          key="view-aria2-download"
        />
        <history-view
          v-else-if="active === 'aria2-history'"
          key="view-history-aria2"
          kind="aria2"
        />
        <quality-view v-else-if="active === 'quality'" key="view-quality" />
        <history-view
          v-else-if="active === 'quality-history'"
          key="view-history-quality"
          kind="enhance"
        />
        <port-check-view v-else-if="active === 'port-check'" key="view-port-check" />
      </keep-alive>
    </a-layout-content>
  </a-layout>

  <a-modal
    v-model:open="showQr"
    title="扫码登录"
    :footer="null"
    centered
  >
    <a-spin :spinning="!qr" tip="正在生成登录二维码…">
      <div class="qr-wrap" v-if="qr">
        <p>使用 B站 APP 扫码登录</p>
        <img :src="qr.qr_svg" alt="qr" width="240" height="240" />
      </div>
    </a-spin>
  </a-modal>
</template>

<style scoped>
.layout {
  height: 100vh;
  width: 100vw;
}
.sider {
  overflow: auto;
}
.login-box {
  display: flex;
  align-items: center;
  min-height: 64px;
  padding: 0.8rem 1.2rem;
  border-bottom: 1px solid #303030;
}
.login-text {
  color: #e5e7eb;
  font-size: 0.9rem;
}
.avatar-img {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  object-fit: cover;
  box-shadow: 0 0 0 2px rgba(56, 189, 248, 0.35);
}
.logout-btn {
  color: #c9d1d9;
  border: 1px solid #3a3f47;
  border-radius: 6px;
  background: rgba(255, 255, 255, 0.04);
  transition: all 0.2s ease;
  justify-content: flex-start;
}
.logout-btn:hover {
  color: #ff7875;
  border-color: #ff7875;
  background: rgba(255, 77, 79, 0.1);
}
.content {
  background: #f5f7fa;
  overflow: hidden;
}
.qr-wrap {
  text-align: center;
}
</style>
