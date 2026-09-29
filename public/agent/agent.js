// ============================================================
// 智能体工作流 - 主脚本
// ============================================================

// 录制数据（真实的 Qwen3 agent 运行记录）
let TASKS = {};

// 从 JSON 文件加载任务数据
async function loadTaskData() {
  try {
    const response = await fetch('./data/manifest.json');
    const manifest = await response.json();

    // 转换数据格式
    for (const task of manifest.tasks) {
      TASKS[task.task_id] = task;
    }

    console.log('✓ 已加载任务数据:', Object.keys(TASKS));
  } catch (error) {
    console.error('加载任务数据失败，使用内置数据:', error);
    // 回退到内置数据
    TASKS = {
      'fix-bug': {
        title: '修复 Bug',
        description: '找到并修复一个数组越界错误，添加单元测试',
        steps: [
      {
        type: 'think',
        content: '我需要先了解项目结构，看看哪里有数组越界的问题。',
        status: '分析任务...'
      },
      {
        type: 'tool',
        tool: 'bash',
        command: 'ls -la',
        output: 'total 24\ndrwxr-xr-x  5 user  staff  160 Jan 15 10:30 .\ndrwxr-xr-x  8 user  staff  256 Jan 15 10:29 ..\n-rw-r--r--  1 user  staff  234 Jan 15 10:30 main.py\n-rw-r--r--  1 user  staff  156 Jan 15 10:30 utils.py\ndrwxr-xr-x  3 user  staff   96 Jan 15 10:30 tests'
      },
      {
        type: 'think',
        content: '好的，项目结构很简单。让我先读取 main.py 看看代码。',
        status: '检查代码...'
      },
      {
        type: 'tool',
        tool: 'read_file',
        file: 'main.py',
        content: `def process_data(items):
    """处理数据列表"""
    results = []
    for i in range(len(items) + 1):  # BUG: 这里会越界
        results.append(items[i] * 2)
    return results

if __name__ == "__main__":
    data = [1, 2, 3, 4, 5]
    print(process_data(data))`
      },
      {
        type: 'think',
        content: '找到问题了！在第 4 行，`range(len(items) + 1)` 会导致索引超出范围。应该是 `range(len(items))`。让我修复它。',
        status: '发现问题...'
      },
      {
        type: 'tool',
        tool: 'edit_file',
        file: 'main.py',
        changes: [
          { line: 4, old: '    for i in range(len(items) + 1):  # BUG: 这里会越界', new: '    for i in range(len(items)):' }
        ]
      },
      {
        type: 'think',
        content: '现在让我运行代码看看是否修复成功。',
        status: '测试修复...'
      },
      {
        type: 'tool',
        tool: 'bash',
        command: 'python main.py',
        output: '[2, 4, 6, 8, 10]'
      },
      {
        type: 'think',
        content: '很好！代码现在正常运行了。接下来我需要添加单元测试来防止这个问题再次出现。',
        status: '编写测试...'
      },
      {
        type: 'tool',
        tool: 'write_file',
        file: 'tests/test_main.py',
        content: `import unittest
import sys
sys.path.append('..')
from main import process_data

class TestProcessData(unittest.TestCase):
    def test_normal_case(self):
        """测试正常情况"""
        result = process_data([1, 2, 3])
        self.assertEqual(result, [2, 4, 6])

    def test_empty_list(self):
        """测试空列表"""
        result = process_data([])
        self.assertEqual(result, [])

    def test_single_item(self):
        """测试单个元素"""
        result = process_data([5])
        self.assertEqual(result, [10])

if __name__ == '__main__':
    unittest.main()`
      },
      {
        type: 'tool',
        tool: 'bash',
        command: 'cd tests && python -m pytest test_main.py -v',
        output: '============================= test session starts ==============================\ntest_main.py::TestProcessData::test_empty_list PASSED              [ 33%]\ntest_main.py::TestProcessData::test_normal_case PASSED             [ 66%]\ntest_main.py::TestProcessData::test_single_item PASSED             [100%]\n\n============================== 3 passed in 0.12s ==============================='
      },
      {
        type: 'complete',
        summary: '任务完成！我修复了数组越界错误，并添加了 3 个单元测试来确保代码的正确性。',
        files_changed: ['main.py', 'tests/test_main.py'],
        tests_passed: 3
      }
    ],
    tokenHistory: [512, 687, 923, 1156, 1489, 1723, 1956, 2234, 2567, 2801],
    generation: {
      text: 'for i in range(len(items)):',
      chars: [
        { char: 'f', prob: 0.94 },
        { char: 'o', prob: 0.98 },
        { char: 'r', prob: 0.99 },
        { char: ' ', prob: 0.97 },
        { char: 'i', prob: 0.92 },
        { char: ' ', prob: 0.96 },
        { char: 'i', prob: 0.89 },
        { char: 'n', prob: 0.95 },
        { char: ' ', prob: 0.98 },
        { char: 'r', prob: 0.88 },
        { char: 'a', prob: 0.94 },
        { char: 'n', prob: 0.97 },
        { char: 'g', prob: 0.99 },
        { char: 'e', prob: 0.98 },
        { char: '(', prob: 0.96 },
        { char: 'l', prob: 0.91 },
        { char: 'e', prob: 0.95 },
        { char: 'n', prob: 0.98 },
        { char: '(', prob: 0.97 },
        { char: 'i', prob: 0.93 },
        { char: 't', prob: 0.96 },
        { char: 'e', prob: 0.98 },
        { char: 'm', prob: 0.97 },
        { char: 's', prob: 0.99 },
        { char: ')', prob: 0.98 },
        { char: ')', prob: 0.96 },
        { char: ':', prob: 0.95 }
      ]
    }
  };
    }
  }
}

// 状态管理
let currentTask = null;
let currentStep = 0;
let isPlaying = false;
let playInterval = null;

// DOM 元素
const els = {
  taskPanel: document.getElementById('taskPanel'),
  workspace: document.getElementById('workspace'),
  agentCtl: document.getElementById('agentCtl'),
  depthPanel: document.getElementById('depthPanel'),

  termBody: document.getElementById('termBody'),
  edTabs: document.getElementById('edTabs'),
  edBody: document.getElementById('edBody'),
  edGutter: document.getElementById('edGutter'),
  edCode: document.getElementById('edCode'),

  thinkStatus: document.getElementById('thinkStatus'),
  thinkContent: document.getElementById('thinkContent'),
  toolCalls: document.getElementById('toolCalls'),
  testPanel: document.getElementById('testPanel'),
  testBody: document.getElementById('testBody'),

  ctlSteps: document.getElementById('ctlSteps'),
  ctlStep: document.getElementById('ctlStep'),
  ctlTime: document.getElementById('ctlTime'),

  btnPlay: document.getElementById('btnPlay'),
  btnPrev: document.getElementById('btnPrev'),
  btnNext: document.getElementById('btnNext'),
  btnReset: document.getElementById('btnReset'),
  btnDepth: document.getElementById('btnDepth'),
  btnDepthClose: document.getElementById('btnDepthClose'),

  loopViz: document.getElementById('loopViz'),
  tokenChart: document.getElementById('tokenChart'),
  genViz: document.getElementById('genViz')
};

// 初始化
async function init() {
  // 先加载任务数据
  await loadTaskData();

  // 绑定任务卡片
  document.querySelectorAll('.task-card').forEach(card => {
    card.addEventListener('click', () => {
      const taskId = card.dataset.task;
      startTask(taskId);
    });
  });

  // 绑定控制按钮
  els.btnPlay.addEventListener('click', togglePlay);
  els.btnPrev.addEventListener('click', prevStep);
  els.btnNext.addEventListener('click', nextStep);
  els.btnReset.addEventListener('click', resetTask);
  els.btnDepth.addEventListener('click', () => {
    els.depthPanel.hidden = false;
    renderDepthPanel();
  });
  els.btnDepthClose.addEventListener('click', () => {
    els.depthPanel.hidden = true;
  });

  // 键盘快捷键
  document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;

    switch(e.key) {
      case ' ':
        e.preventDefault();
        togglePlay();
        break;
      case 'ArrowLeft':
        e.preventDefault();
        prevStep();
        break;
      case 'ArrowRight':
        e.preventDefault();
        nextStep();
        break;
      case 'd':
      case 'D':
        if (!els.workspace.hidden) {
          els.depthPanel.hidden = !els.depthPanel.hidden;
          if (!els.depthPanel.hidden) renderDepthPanel();
        }
        break;
    }
  });

  // 初始化背景
  initBackground();
}

// 开始任务
function startTask(taskId) {
  currentTask = TASKS[taskId];
  currentStep = 0;

  els.taskPanel.hidden = true;
  els.workspace.hidden = false;
  els.agentCtl.hidden = false;

  renderStepProgress();
  renderStep();
}

// 重置任务
function resetTask() {
  if (isPlaying) togglePlay();

  currentTask = null;
  currentStep = 0;

  els.taskPanel.hidden = false;
  els.workspace.hidden = true;
  els.agentCtl.hidden = true;
  els.depthPanel.hidden = true;

  clearWorkspace();
}

// 渲染步骤进度
function renderStepProgress() {
  els.ctlSteps.innerHTML = '';
  currentTask.steps.forEach((step, i) => {
    const item = document.createElement('div');
    item.className = 'ctl-step-item';
    if (i === currentStep) item.classList.add('active');
    if (i < currentStep) item.classList.add('done');

    let label = '';
    switch(step.type) {
      case 'think': label = '💭 思考'; break;
      case 'tool': label = `🔧 ${step.tool}`; break;
      case 'complete': label = '✓ 完成'; break;
    }
    item.textContent = label;
    item.addEventListener('click', () => {
      currentStep = i;
      renderStep();
      renderStepProgress();
    });
    els.ctlSteps.appendChild(item);
  });

  els.ctlStep.textContent = `${currentStep + 1} / ${currentTask.steps.length}`;
}

// 渲染当前步骤
function renderStep() {
  const step = currentTask.steps[currentStep];

  switch(step.type) {
    case 'think':
      renderThinkStep(step);
      break;
    case 'tool':
      renderToolStep(step);
      break;
    case 'complete':
      renderCompleteStep(step);
      break;
  }

  renderStepProgress();
}

// 渲染思考步骤
function renderThinkStep(step) {
  els.thinkStatus.textContent = step.status || '思考中...';
  els.thinkContent.innerHTML = `<p>${step.content}</p>`;
  els.toolCalls.innerHTML = '';
}

// 渲染工具调用步骤
function renderToolStep(step) {
  els.thinkStatus.textContent = '调用工具...';

  const toolCall = document.createElement('div');
  toolCall.className = 'tool-call';

  toolCall.innerHTML = `
    <div class="tool-name">${step.tool}</div>
    <div class="tool-args">${step.command || step.file || ''}</div>
  `;

  if (step.output || step.content) {
    const result = document.createElement('div');
    result.className = 'tool-result success';
    result.textContent = step.output || step.content || '';
    toolCall.appendChild(result);
  }

  els.toolCalls.innerHTML = '';
  els.toolCalls.appendChild(toolCall);

  // 更新终端或编辑器
  if (step.tool === 'bash') {
    addTerminalLine(`$ ${step.command}`, 'command');
    if (step.output) {
      step.output.split('\n').forEach(line => {
        addTerminalLine(line, 'output');
      });
    }
  } else if (step.tool === 'read_file' || step.tool === 'write_file' || step.tool === 'edit_file') {
    showFileInEditor(step);
  }
}

// 渲染完成步骤
function renderCompleteStep(step) {
  els.thinkStatus.textContent = '任务完成 ✓';
  els.thinkContent.innerHTML = `<p>${step.summary}</p>`;
  els.toolCalls.innerHTML = '';

  els.testPanel.hidden = false;
  els.testBody.innerHTML = `
    <div>✓ 修改了 ${step.files_changed.length} 个文件</div>
    <div>✓ 通过了 ${step.tests_passed} 个测试</div>
    <div style="margin-top: 8px; color: var(--dim);">
      ${step.files_changed.map(f => `  ${f}`).join('<br>')}
    </div>
  `;
}

// 添加终端行
function addTerminalLine(text, type) {
  const line = document.createElement('div');
  line.className = `term-line term-${type}`;

  if (type === 'command') {
    line.innerHTML = `<span class="term-prompt">$</span> ${text.substring(2)}`;
  } else {
    line.textContent = text;
  }

  els.termBody.appendChild(line);
  els.termBody.scrollTop = els.termBody.scrollHeight;
}

// 在编辑器中显示文件
function showFileInEditor(step) {
  const filename = step.file;
  let content = step.content || '';

  // 更新标签
  els.edTabs.innerHTML = '';
  const tab = document.createElement('button');
  tab.className = 'ed-tab active';
  tab.textContent = filename;
  els.edTabs.appendChild(tab);

  // 更新内容
  if (step.tool === 'edit_file' && step.changes) {
    // 显示文件改动
    const lines = content.split('\n');
    step.changes.forEach(change => {
      if (change.old) {
        lines[change.line - 1] = `<span class="ed-line-remove">${escapeHtml(change.old)}</span>`;
      }
      if (change.new) {
        lines.splice(change.line, 0, `<span class="ed-line-add">${escapeHtml(change.new)}</span>`);
      }
    });
    content = lines.join('\n');
    els.edCode.innerHTML = content;
  } else {
    els.edCode.textContent = content;
  }

  // 更新行号
  const lineCount = content.split('\n').length;
  els.edGutter.innerHTML = Array.from({ length: lineCount }, (_, i) => i + 1).join('\n');
}

// 清空工作区
function clearWorkspace() {
  els.termBody.innerHTML = '';
  els.edTabs.innerHTML = '';
  els.edCode.textContent = '';
  els.edGutter.innerHTML = '';
  els.thinkContent.innerHTML = '';
  els.toolCalls.innerHTML = '';
  els.testPanel.hidden = true;
}

// 播放控制
function togglePlay() {
  isPlaying = !isPlaying;
  els.btnPlay.textContent = isPlaying ? '⏸' : '▶';

  if (isPlaying) {
    playInterval = setInterval(() => {
      if (currentStep < currentTask.steps.length - 1) {
        nextStep();
      } else {
        togglePlay();
      }
    }, 2000);
  } else {
    clearInterval(playInterval);
  }
}

function prevStep() {
  if (currentStep > 0) {
    currentStep--;
    renderStep();
  }
}

function nextStep() {
  if (currentStep < currentTask.steps.length - 1) {
    currentStep++;
    renderStep();
  }
}

// 渲染深度面板
function renderDepthPanel() {
  if (!currentTask) return;

  // 渲染循环可视化
  els.loopViz.innerHTML = `
    <div class="loop-step">
      <div class="loop-num">1</div>
      <div class="loop-desc">模型接收任务和上下文</div>
    </div>
    <div class="loop-arrow">↓</div>
    <div class="loop-step">
      <div class="loop-num">2</div>
      <div class="loop-desc">生成思考和工具调用</div>
    </div>
    <div class="loop-arrow">↓</div>
    <div class="loop-step">
      <div class="loop-num">3</div>
      <div class="loop-desc">执行工具（bash / 文件操作）</div>
    </div>
    <div class="loop-arrow">↓</div>
    <div class="loop-step">
      <div class="loop-num">4</div>
      <div class="loop-desc">将结果添加到上下文</div>
    </div>
    <div class="loop-arrow">↓</div>
    <div class="loop-step">
      <div class="loop-num">5</div>
      <div class="loop-desc">循环直到任务完成</div>
    </div>
  `;

  // 渲染 Token 图表
  renderTokenChart();

  // 渲染字符生成
  els.genViz.innerHTML = '';
  currentTask.generation.chars.forEach(({ char, prob }) => {
    const span = document.createElement('span');
    span.className = 'gen-char';
    span.innerHTML = `${escapeHtml(char)}<span class="gen-char-prob">${(prob * 100).toFixed(0)}%</span>`;
    span.title = `字符: "${char}" | 概率: ${(prob * 100).toFixed(1)}%`;
    els.genViz.appendChild(span);
  });
}

// 渲染 Token 图表
function renderTokenChart() {
  const canvas = els.tokenChart;
  const ctx = canvas.getContext('2d');
  const width = canvas.width;
  const height = canvas.height;

  ctx.clearRect(0, 0, width, height);

  const data = currentTask.tokenHistory;
  const max = Math.max(...data);
  const stepWidth = width / (data.length - 1);

  // 绘制网格
  ctx.strokeStyle = 'rgba(150, 180, 230, 0.1)';
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = height - (i * height / 4);
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  // 绘制线条
  ctx.strokeStyle = '#5ef0d4';
  ctx.lineWidth = 2;
  ctx.beginPath();
  data.forEach((value, i) => {
    const x = i * stepWidth;
    const y = height - (value / max * height * 0.9) - height * 0.05;
    if (i === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();

  // 绘制点
  ctx.fillStyle = '#5ef0d4';
  data.forEach((value, i) => {
    const x = i * stepWidth;
    const y = height - (value / max * height * 0.9) - height * 0.05;
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  });

  // 绘制标签
  ctx.fillStyle = '#7a859e';
  ctx.font = '11px "JetBrains Mono", monospace';
  ctx.fillText('0', 5, height - 5);
  ctx.fillText(`${max}`, 5, 15);
  ctx.fillText('Context Tokens', width - 90, height - 5);
}

// 初始化背景
function initBackground() {
  const canvas = document.getElementById('bg');
  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  resize();
  window.addEventListener('resize', resize);

  const particles = [];
  const particleCount = 50;

  for (let i = 0; i < particleCount; i++) {
    particles.push({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.5,
      vy: (Math.random() - 0.5) * 0.5,
      radius: Math.random() * 1.5 + 0.5
    });
  }

  function animate() {
    ctx.fillStyle = 'rgba(6, 13, 26, 0.05)';
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    ctx.fillStyle = 'rgba(94, 240, 212, 0.4)';
    particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;

      if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fill();
    });

    requestAnimationFrame(animate);
  }

  animate();
}

// 工具函数
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// 启动
init();
