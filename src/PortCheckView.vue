<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { invoke } from '@tauri-apps/api/core'
import {
  ReloadOutlined,
  SearchOutlined,
  CloseCircleOutlined,
  LoadingOutlined,
  CheckCircleOutlined,
  ExclamationCircleOutlined,
  MinusCircleOutlined,
} from '@ant-design/icons-vue'
import { message, Modal } from 'ant-design-vue'

interface PortInfo {
  protocol: string
  local_ip: string
  local_port: number
  remote_ip: string
  remote_port: number
  state: string
  pid: number
  process_name: string
}

interface PortQuery {
  port?: number | null
  protocol?: string | null
  state?: string | null
  pid?: number | null
}

const ports = ref<PortInfo[]>([])
const loading = ref(false)
const filterPort = ref<number | null>(null)
const filterProtocol = ref<string | null>(null)
const filterState = ref<string | null>(null)
const quickPort = ref<number | null>(null)
const checkingPort = ref(false)
const checkResult = ref<boolean | null>(null)
const killing = ref<number | null>(null)
let refreshInterval: number | null = null

// 格式化地址
function formatAddress(ip: string, port: number): string {
  if (ip === '*') return '*:*'
  return `${ip}:${port}`
}

// 获取状态颜色
function getStateColor(state: string): string {
  switch (state.toUpperCase()) {
    case 'LISTENING':
      return 'green'
    case 'ESTABLISHED':
      return 'blue'
    case 'CLOSE_WAIT':
    case 'TIME_WAIT':
      return 'orange'
    default:
      return 'default'
  }
}

// 获取协议颜色
function getProtocolColor(protocol: string): string {
  return protocol.toUpperCase() === 'TCP' ? 'blue' : 'purple'
}

// 查询端口
async function queryPorts() {
  loading.value = true
  try {
    const filter: PortQuery = {
      port: filterPort.value || null,
      protocol: filterProtocol.value || null,
      state: filterState.value || null,
      pid: null,
    }
    ports.value = await invoke<PortInfo[]>('port_query', { filter })
  } catch (error) {
    message.error(`查询端口失败: ${error}`)
  } finally {
    loading.value = false
  }
}

// 快速检查端口
async function checkPort() {
  if (!quickPort.value) {
    message.warning('请输入端口号')
    return
  }
  checkingPort.value = true
  checkResult.value = null
  try {
    checkResult.value = await invoke<boolean>('port_check', { port: quickPort.value })
  } catch (error) {
    message.error(`检查端口失败: ${error}`)
  } finally {
    checkingPort.value = false
  }
}

// 终止进程
async function killProcess(pid: number, port: number) {
  Modal.confirm({
    title: '确认终止进程',
    content: `确定要终止占用端口 ${port} 的进程 (PID=${pid}) 吗？`,
    okText: '确认终止',
    cancelText: '取消',
    okButtonProps: { danger: true },
    onOk: async () => {
      killing.value = pid
      try {
        const name = await invoke<string>('port_kill_process', { pid })
        message.success(`已终止进程 ${name || pid}`)
        await queryPorts()
      } catch (error) {
        message.error(`终止进程失败: ${error}`)
      } finally {
        killing.value = null
      }
    },
  })
}

// 按端口终止
async function killByPort(port: number) {
  Modal.confirm({
    title: '确认释放端口',
    content: `确定要终止占用端口 ${port} 的进程吗？`,
    okText: '确认释放',
    cancelText: '取消',
    okButtonProps: { danger: true },
    onOk: async () => {
      killing.value = -1
      try {
        const info = await invoke<PortInfo>('port_kill_by_port', { port })
        message.success(`已释放端口 ${port}（${info.process_name || `PID=${info.pid}`}）`)
        await queryPorts()
      } catch (error) {
        message.error(`释放端口失败: ${error}`)
      } finally {
        killing.value = null
      }
    },
  })
}

// 清空过滤条件
function clearFilters() {
  filterPort.value = null
  filterProtocol.value = null
  filterState.value = null
}

// 表格列定义
const columns = [
  { title: '协议', dataIndex: 'protocol', key: 'protocol', width: 80 },
  { title: '本地地址', key: 'local', width: 200 },
  { title: '远程地址', key: 'remote', width: 200 },
  { title: '状态', dataIndex: 'state', key: 'state', width: 130 },
  { title: 'PID', dataIndex: 'pid', key: 'pid', width: 80 },
  { title: '进程名', dataIndex: 'process_name', key: 'process_name', width: 160 },
  { title: '操作', key: 'action', width: 100, fixed: 'right' as const },
]

onMounted(async () => {
  await queryPorts()
  // 每 5 秒自动刷新
  refreshInterval = window.setInterval(() => {
    queryPorts()
  }, 5000)
})

onUnmounted(() => {
  if (refreshInterval !== null) {
    clearInterval(refreshInterval)
  }
})
</script>

<template>
  <div class="port-check-view">
    <!-- 头部 -->
    <div class="header">
      <h2>端口占用查询</h2>
      <a-button type="primary" @click="queryPorts" :loading="loading">
        <template #icon><ReloadOutlined /></template>
        刷新
      </a-button>
    </div>

    <!-- 快速检查 -->
    <a-card class="quick-check" size="small">
      <a-space>
        <span class="section-label">快速检查：</span>
        <a-input-number
          v-model:value="quickPort"
          :min="1"
          :max="65535"
          placeholder="端口号"
          style="width: 140px"
        />
        <a-button @click="checkPort" :loading="checkingPort">
          <template #icon><SearchOutlined /></template>
          检查
        </a-button>
        <template v-if="checkResult !== null">
          <a-tag v-if="checkResult" color="red">
            <template #icon><CloseCircleOutlined /></template>
            端口 {{ quickPort }} 已被占用
          </a-tag>
          <a-tag v-else color="green">
            <template #icon><CheckCircleOutlined /></template>
            端口 {{ quickPort }} 可用
          </a-tag>
          <a-button
            v-if="checkResult"
            size="small"
            danger
            :loading="killing === -1"
            @click="killByPort(quickPort!)"
          >
            <template #icon><CloseCircleOutlined /></template>
            释放端口
          </a-button>
        </template>
      </a-space>
    </a-card>

    <!-- 过滤条件 -->
    <a-card class="filter-section" size="small">
      <a-space wrap>
        <span class="section-label">过滤条件：</span>
        <a-input-number
          v-model:value="filterPort"
          :min="1"
          :max="65535"
          placeholder="端口号"
          style="width: 120px"
        />
        <a-select
          v-model:value="filterProtocol"
          placeholder="协议"
          allow-clear
          style="width: 100px"
        >
          <a-select-option value="TCP">TCP</a-select-option>
          <a-select-option value="UDP">UDP</a-select-option>
        </a-select>
        <a-select
          v-model:value="filterState"
          placeholder="状态"
          allow-clear
          style="width: 130px"
        >
          <a-select-option value="LISTENING">LISTENING</a-select-option>
          <a-select-option value="ESTABLISHED">ESTABLISHED</a-select-option>
          <a-select-option value="TIME_WAIT">TIME_WAIT</a-select-option>
          <a-select-option value="CLOSE_WAIT">CLOSE_WAIT</a-select-option>
        </a-select>
        <a-button type="primary" @click="queryPorts" :loading="loading">
          <template #icon><SearchOutlined /></template>
          查询
        </a-button>
        <a-button @click="clearFilters">清空</a-button>
      </a-space>
    </a-card>

    <!-- 数据表格 -->
    <a-card class="table-section" size="small">
      <template #title>
        <span>端口列表</span>
        <a-tag style="margin-left: 8px">{{ ports.length }} 条</a-tag>
      </template>
      <a-table
        :dataSource="ports"
        :columns="columns"
        :rowKey="(r: PortInfo) => `${r.protocol}-${r.local_port}-${r.pid}-${Math.random()}`"
        :pagination="{ pageSize: 50, showSizeChanger: true, showTotal: (t: number) => `共 ${t} 条` }"
        :scroll="{ y: 'calc(100vh - 380px)' }"
        size="small"
        :loading="loading"
      >
        <!-- 协议 -->
        <template #bodyCell="{ column, record }: { column: any; record: PortInfo }">
          <template v-if="column.key === 'protocol'">
            <a-tag :color="getProtocolColor(record.protocol)">{{ record.protocol }}</a-tag>
          </template>
          <!-- 本地地址 -->
          <template v-else-if="column.key === 'local'">
            <code>{{ formatAddress(record.local_ip, record.local_port) }}</code>
          </template>
          <!-- 远程地址 -->
          <template v-else-if="column.key === 'remote'">
            <code>{{ formatAddress(record.remote_ip, record.remote_port) }}</code>
          </template>
          <!-- 状态 -->
          <template v-else-if="column.key === 'state'">
            <a-tag v-if="record.state" :color="getStateColor(record.state)">{{ record.state }}</a-tag>
            <span v-else class="state-na">-</span>
          </template>
          <!-- 进程名 -->
          <template v-else-if="column.key === 'process_name'">
            {{ record.process_name || `PID=${record.pid}` }}
          </template>
          <!-- 操作 -->
          <template v-else-if="column.key === 'action'">
            <a-button
              v-if="record.pid > 0 && record.pid !== 4"
              size="small"
              danger
              :loading="killing === record.pid"
              @click="killProcess(record.pid, record.local_port)"
            >
              <template #icon><CloseCircleOutlined /></template>
              终止
            </a-button>
            <a-tag v-else color="red">系统进程</a-tag>
          </template>
        </template>
      </a-table>
    </a-card>
  </div>
</template>

<style scoped>
.port-check-view {
  padding: 20px;
  height: 100%;
  overflow-y: auto;
}

.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.header h2 {
  margin: 0;
  font-size: 20px;
}

.section-label {
  font-weight: 500;
  color: #666;
  white-space: nowrap;
}

.quick-check {
  margin-bottom: 12px;
}

.filter-section {
  margin-bottom: 12px;
}

.table-section {
  flex: 1;
}

.state-na {
  color: #999;
}

code {
  font-size: 12px;
  background: #f5f5f5;
  padding: 1px 6px;
  border-radius: 3px;
}
</style>