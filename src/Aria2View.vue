<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { invoke } from '@tauri-apps/api/core'
import { listen, type UnlistenFn } from '@tauri-apps/api/event'
import { open } from '@tauri-apps/plugin-dialog'
import {
  CloudDownloadOutlined,
  PauseCircleOutlined,
  PlayCircleOutlined,
  DeleteOutlined,
  FolderOpenOutlined,
  PlusOutlined,
  ReloadOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  ClockCircleOutlined,
  MinusCircleOutlined
} from '@ant-design/icons-vue'
import { message } from 'ant-design-vue'

interface Aria2Task {
  gid: string
  name: string
  url: string
  status: 'Active' | 'Waiting' | 'Paused' | 'Complete' | 'Error' | 'Removed'
  total_length: number
  completed_length: number
  download_speed: number
  upload_speed: number
  progress: number
  error_message: string | null
  created_at: number
  output_dir: string
  file_path: string | null
}

const tasks = ref<Aria2Task[]>([])
const loading = ref(false)
const serviceRunning = ref(false)
const inputUrls = ref('')
const outputDir = ref('')
let unlistenTasks: UnlistenFn | null = null
let refreshInterval: number | null = null

// 格式化文件大小
function formatSize(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return (bytes / Math.pow(k, i)).toFixed(2) + ' ' + sizes[i]
}

// 格式化速度
function formatSpeed(bytesPerSec: number): string {
  return formatSize(bytesPerSec) + '/s'
}

// 获取状态颜色
function getStatusColor(status: string): string {
  switch (status) {
    case 'Active':
      return 'blue'
    case 'Waiting':
      return 'orange'
    case 'Paused':
      return 'gold'
    case 'Complete':
      return 'green'
    case 'Error':
      return 'red'
    default:
      return 'default'
  }
}

// 获取状态图标
function getStatusIcon(status: string) {
  switch (status) {
    case 'Active':
      return CloudDownloadOutlined
    case 'Waiting':
      return ClockCircleOutlined
    case 'Paused':
      return MinusCircleOutlined
    case 'Complete':
      return CheckCircleOutlined
    case 'Error':
      return CloseCircleOutlined
    default:
      return ClockCircleOutlined
  }
}

// 获取状态文本
function getStatusText(status: string): string {
  switch (status) {
    case 'Active':
      return '下载中'
    case 'Waiting':
      return '等待中'
    case 'Paused':
      return '已暂停'
    case 'Complete':
      return '已完成'
    case 'Error':
      return '错误'
    default:
      return status
  }
}

// 启动 aria2 服务
async function startService() {
  try {
    await invoke('aria2_start_service')
    serviceRunning.value = true
    message.success('aria2 服务已启动')
    await refreshTasks()
  } catch (error) {
    message.error(`启动服务失败: ${error}`)
  }
}

// 停止 aria2 服务
async function stopService() {
  try {
    await invoke('aria2_stop_service')
    serviceRunning.value = false
    message.success('aria2 服务已停止')
    tasks.value = []
  } catch (error) {
    message.error(`停止服务失败: ${error}`)
  }
}

// 检查服务状态
async function checkServiceStatus() {
  try {
    const running = await invoke<boolean>('aria2_is_running')
    serviceRunning.value = running
  } catch (error) {
    console.error('检查服务状态失败:', error)
  }
}

// 刷新任务列表
async function refreshTasks() {
  if (!serviceRunning.value) return
  
  try {
    loading.value = true
    await invoke('aria2_refresh_tasks')
    const taskList = await invoke<Aria2Task[]>('aria2_list_tasks')
    tasks.value = taskList
  } catch (error) {
    console.error('刷新任务失败:', error)
  } finally {
    loading.value = false
  }
}

// 添加下载任务
async function addTask() {
  if (!inputUrls.value.trim()) {
    message.warning('请输入下载链接')
    return
  }

  const urls = inputUrls.value
    .split('\n')
    .map(url => url.trim())
    .filter(url => url.length > 0)

  if (urls.length === 0) {
    message.warning('请输入有效的下载链接')
    return
  }

  try {
    loading.value = true
    await invoke('aria2_add_task', {
      request: {
        urls,
        output_dir: outputDir.value || null,
        file_name: null,
        options: null
      }
    })
    message.success(`已添加 ${urls.length} 个下载任务`)
    inputUrls.value = ''
    await refreshTasks()
  } catch (error) {
    message.error(`添加任务失败: ${error}`)
  } finally {
    loading.value = false
  }
}

// 暂停任务
async function pauseTask(gid: string) {
  try {
    await invoke('aria2_pause_task', { gid })
    message.success('任务已暂停')
    await refreshTasks()
  } catch (error) {
    message.error(`暂停任务失败: ${error}`)
  }
}

// 恢复任务
async function resumeTask(gid: string) {
  try {
    await invoke('aria2_resume_task', { gid })
    message.success('任务已恢复')
    await refreshTasks()
  } catch (error) {
    message.error(`恢复任务失败: ${error}`)
  }
}

// 删除任务
async function removeTask(gid: string) {
  try {
    await invoke('aria2_remove_task', { gid })
    message.success('任务已删除')
    await refreshTasks()
  } catch (error) {
    message.error(`删除任务失败: ${error}`)
  }
}

// 选择输出目录
async function selectOutputDir() {
  try {
    const selected = await open({
      directory: true,
      multiple: false,
      title: '选择输出目录'
    })
    if (selected && typeof selected === 'string') {
      outputDir.value = selected
    }
  } catch (error) {
    message.error(`选择目录失败: ${error}`)
  }
}

// 打开文件所在目录
async function openFilePath(filePath: string) {
  try {
    await invoke('open_path', { path: filePath })
  } catch (error) {
    message.error(`打开文件失败: ${error}`)
  }
}

// 监听任务更新事件
async function setupTaskListener() {
  unlistenTasks = await listen<Aria2Task[]>('aria2-tasks-update', (event) => {
    tasks.value = event.payload
  })
}

onMounted(async () => {
  await checkServiceStatus()
  await setupTaskListener()
  
  // 如果服务未运行，自动启动
  if (!serviceRunning.value) {
    await startService()
  } else {
    await refreshTasks()
  }
  
  // 定时刷新任务列表（每 2 秒）
  refreshInterval = window.setInterval(() => {
    if (serviceRunning.value) {
      refreshTasks()
    }
  }, 2000)
})

onUnmounted(() => {
  if (unlistenTasks) {
    unlistenTasks()
  }
  if (refreshInterval) {
    clearInterval(refreshInterval)
  }
})
</script>

<template>
  <div class="aria2-view">
    <div class="header">
      <h2>aria2 下载器</h2>
      <div class="service-controls">
        <a-tag :color="serviceRunning ? 'green' : 'red'">
          {{ serviceRunning ? '服务运行中' : '服务已停止' }}
        </a-tag>
        <a-button
          v-if="!serviceRunning"
          type="primary"
          size="small"
          @click="startService"
        >
          启动服务
        </a-button>
        <a-button
          v-else
          danger
          size="small"
          @click="stopService"
        >
          停止服务
        </a-button>
      </div>
    </div>

    <div class="add-task-section">
      <a-card title="添加下载任务" size="small">
        <a-form layout="vertical">
          <a-form-item label="下载链接（支持多个链接，每行一个）">
            <a-textarea
              v-model:value="inputUrls"
              placeholder="输入 HTTP/HTTPS/FTP/BT/磁力链接，每行一个"
              :rows="4"
              :disabled="!serviceRunning"
            />
          </a-form-item>
          <a-form-item label="输出目录">
            <a-space>
              <a-input
                v-model:value="outputDir"
                placeholder="留空使用默认目录"
                :disabled="!serviceRunning"
                style="width: 400px"
              />
              <a-button @click="selectOutputDir" :disabled="!serviceRunning">
                <template #icon>
                  <FolderOpenOutlined />
                </template>
                选择目录
              </a-button>
            </a-space>
          </a-form-item>
          <a-button
            type="primary"
            @click="addTask"
            :loading="loading"
            :disabled="!serviceRunning"
          >
            <template #icon>
              <PlusOutlined />
            </template>
            添加下载
          </a-button>
        </a-form>
      </a-card>
    </div>

    <div class="task-list-section">
      <div class="section-header">
        <h3>下载任务</h3>
        <a-button
          size="small"
          @click="refreshTasks"
          :loading="loading"
          :disabled="!serviceRunning"
        >
          <template #icon>
            <ReloadOutlined />
          </template>
          刷新
        </a-button>
      </div>

      <a-spin :spinning="loading">
        <div v-if="tasks.length === 0 && serviceRunning" class="empty-state">
          <a-empty description="暂无下载任务" />
        </div>

        <div v-else class="task-list">
          <a-card
            v-for="task in tasks"
            :key="task.gid"
            size="small"
            class="task-card"
          >
            <div class="task-header">
              <div class="task-info">
                <div class="task-name">{{ task.name }}</div>
                <div class="task-meta">
                  <a-tag :color="getStatusColor(task.status)">
                    <template #icon>
                      <component :is="getStatusIcon(task.status)" />
                    </template>
                    {{ getStatusText(task.status) }}
                  </a-tag>
                  <span class="size-info">
                    {{ formatSize(task.completed_length) }} / {{ formatSize(task.total_length) }}
                  </span>
                  <span v-if="task.status === 'Active'" class="speed-info">
                    {{ formatSpeed(task.download_speed) }}
                  </span>
                </div>
              </div>
              <div class="task-actions">
                <a-button
                  v-if="task.status === 'Active'"
                  size="small"
                  @click="pauseTask(task.gid)"
                >
                  <template #icon>
                    <PauseCircleOutlined />
                  </template>
                  暂停
                </a-button>
                <a-button
                  v-if="task.status === 'Paused'"
                  size="small"
                  type="primary"
                  @click="resumeTask(task.gid)"
                >
                  <template #icon>
                    <PlayCircleOutlined />
                  </template>
                  继续
                </a-button>
                <a-button
                  size="small"
                  danger
                  @click="removeTask(task.gid)"
                >
                  <template #icon>
                    <DeleteOutlined />
                  </template>
                  删除
                </a-button>
                <a-button
                  v-if="task.file_path && task.status === 'Complete'"
                  size="small"
                  @click="openFilePath(task.file_path!)"
                >
                  <template #icon>
                    <FolderOpenOutlined />
                  </template>
                  打开
                </a-button>
              </div>
            </div>
            <a-progress
              :percent="Math.round(task.progress)"
              :status="task.status === 'Error' ? 'exception' : task.status === 'Complete' ? 'success' : 'active'"
              size="small"
            />
            <div v-if="task.error_message" class="error-message">
              错误: {{ task.error_message }}
            </div>
          </a-card>
        </div>
      </a-spin>
    </div>
  </div>
</template>

<style scoped>
.aria2-view {
  padding: 20px;
  height: 100%;
  overflow-y: auto;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.header h2 {
  margin: 0;
  font-size: 20px;
}

.service-controls {
  display: flex;
  align-items: center;
  gap: 10px;
}

.add-task-section {
  margin-bottom: 20px;
}

.task-list-section {
  margin-top: 20px;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 15px;
}

.section-header h3 {
  margin: 0;
  font-size: 16px;
}

.empty-state {
  padding: 40px;
  text-align: center;
}

.task-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.task-card {
  transition: box-shadow 0.3s;
}

.task-card:hover {
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.15);
}

.task-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  margin-bottom: 10px;
}

.task-info {
  flex: 1;
}

.task-name {
  font-weight: 500;
  font-size: 14px;
  margin-bottom: 6px;
  word-break: break-all;
}

.task-meta {
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 12px;
  color: #666;
}

.size-info {
  color: #999;
}

.speed-info {
  color: #1890ff;
  font-weight: 500;
}

.task-actions {
  display: flex;
  gap: 8px;
}

.error-message {
  margin-top: 8px;
  padding: 6px 10px;
  background: #fff2f0;
  border: 1px solid #ffccc7;
  border-radius: 4px;
  color: #ff4d4f;
  font-size: 12px;
}
</style>
