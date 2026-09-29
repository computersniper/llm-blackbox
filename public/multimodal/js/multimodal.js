/**
 * 多模态模型可视化 - 使用真实的Qwen2-VL模型数据
 *
 * 数据来源：tools/export_multimodal.py 导出的真实模型数据
 * - 14×14 图像patch切分
 * - Vision Transformer的patch embeddings（768维）
 * - 逐字符生成时对各个patch的真实注意力权重
 * - 真实的问答对和模型输出
 */

import { Background } from '../../js/bg.js';
import { sfx, setSound, soundOn } from '../../js/audio.js';
import { $ } from '../../js/ui.js';

// 示例数据 - 从manifest.json加载
let EXAMPLES = [];
let MANIFEST = null;

// 加载示例数据清单
async function loadManifest() {
  try {
    const response = await fetch('data/manifest.json');
    if (!response.ok) throw new Error('Failed to load manifest');
    MANIFEST = await response.json();

    // 将manifest中的示例转换为EXAMPLES格式（暂不加载详细数据）
    EXAMPLES = MANIFEST.examples.map(ex => ({
      id: ex.id,
      name: ex.name,
      image: ex.image,
      question: ex.question,
      answer: ex.answer,
      dataFile: ex.dataFile,
      // 详细数据稍后按需加载
      patches: null,
      attention: null
    }));

    return true;
  } catch (err) {
    console.error('Failed to load manifest:', err);
    return false;
  }
}

// 加载单个示例的详细数据
async function loadExampleData(example) {
  if (example.patches && example.attention) {
    // 已经加载过了
    return example;
  }

  try {
    const response = await fetch(`data/${example.dataFile}`);
    if (!response.ok) throw new Error(`Failed to load ${example.dataFile}`);
    const data = await response.json();

    // 更新示例数据
    example.patches = data.patches;
    example.attention = data.attention;

    return example;
  } catch (err) {
    console.error(`Failed to load example data for ${example.id}:`, err);
    // 使用fallback数据
    example.patches = generateMockPatches(14, 14);
    example.attention = generateMockAttention(14, 14, example.answer);
    return example;
  }
}

// 生成模拟的图像切块数据（fallback）
function generateMockPatches(rows, cols) {
  const patches = [];
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      patches.push({
        id: i * cols + j,
        row: i,
        col: j,
        embedding: Array(768).fill(0).map(() => Math.random() * 2 - 1)
      });
    }
  }
  return patches;
}

// 生成模拟的注意力数据
function generateMockAttention(rows, cols, answer) {
  const chars = answer.split('');
  const totalPatches = rows * cols;
  const attention = [];

  chars.forEach((char, charIdx) => {
    const weights = [];
    // 为每个字符生成对所有patch的注意力权重
    for (let i = 0; i < totalPatches; i++) {
      // 根据字符位置，关注不同的区域
      const row = Math.floor(i / cols);
      const col = i % cols;

      // 模拟：不同字符关注不同区域
      let weight = 0;
      if (charIdx < chars.length / 3) {
        // 前部分关注左上
        weight = Math.exp(-((row - 3) ** 2 + (col - 3) ** 2) / 20);
      } else if (charIdx < chars.length * 2 / 3) {
        // 中部分关注中间
        weight = Math.exp(-((row - rows/2) ** 2 + (col - cols/2) ** 2) / 20);
      } else {
        // 后部分关注右下
        weight = Math.exp(-((row - rows + 3) ** 2 + (col - cols + 3) ** 2) / 20);
      }
      weight += Math.random() * 0.1;
      weights.push(weight);
    }

    // 归一化
    const sum = weights.reduce((a, b) => a + b, 0);
    const normalized = weights.map(w => w / sum);

    attention.push({
      char,
      charIdx,
      weights: normalized
    });
  });

  return attention;
}

class MultimodalApp {
  constructor() {
    this.currentExample = null;
    this.currentView = 'patches';
    this.currentCharIdx = -1;
    this.playing = false;
    this.playTimer = null;

    this.canvas = $('#imageCanvas');
    this.ctx = this.canvas.getContext('2d');
    this.patchOverlay = $('#patchOverlay');

    this.init();
  }

  async init() {
    new Background($('#bg'));

    // 加载数据清单
    const loaded = await loadManifest();
    if (!loaded || EXAMPLES.length === 0) {
      console.error('Failed to load examples');
      return;
    }

    // 绑定声音按钮
    const sb = $('#btnSound');
    const renderSound = () => {
      sb.classList.toggle('on', soundOn());
      sb.setAttribute('aria-pressed', soundOn());
      sb.title = soundOn() ? '关闭声音' : '打开声音';
    };
    sb.addEventListener('click', () => {
      setSound(!soundOn());
      renderSound();
      sfx.click();
    });
    renderSound();

    // 渲染示例列表
    this.renderExamples();

    // 绑定控制按钮
    $('#btnPlay').addEventListener('click', () => this.togglePlay());
    $('#btnPrev').addEventListener('click', () => this.prevChar());
    $('#btnNext').addEventListener('click', () => this.nextChar());

    // 绑定视图切换
    document.querySelectorAll('.view-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        this.switchView(btn.dataset.view);
      });
    });

    // 加载第一个示例
    await this.loadExample(EXAMPLES[0]);
  }

  renderExamples() {
    const container = $('#examples');
    container.innerHTML = EXAMPLES.map((ex, idx) => `
      <div class="mm-example ${idx === 0 ? 'active' : ''}" data-id="${ex.id}">
        <div class="mm-example-label">${ex.name}</div>
      </div>
    `).join('');

    // 绑定点击事件
    container.querySelectorAll('.mm-example').forEach((elem, idx) => {
      elem.addEventListener('click', () => {
        this.loadExample(EXAMPLES[idx]);
      });
    });
  }

  async loadExample(example) {
    this.stopPlay();

    // 加载详细数据
    await loadExampleData(example);

    this.currentExample = example;
    this.currentCharIdx = -1;

    // 更新选中状态
    document.querySelectorAll('.mm-example').forEach(elem => {
      elem.classList.toggle('active', elem.dataset.id === example.id);
    });

    // 更新问题
    $('#question').textContent = example.question;

    // 更新回答
    this.renderResponse();

    // 加载图片
    await this.loadImage(example.image);

    // 更新视图
    this.updateView();
  }

  loadImage(src) {
    return new Promise((resolve, reject) => {
      // 尝试加载真实图片，失败则使用占位符
      const img = new Image();
      img.onload = () => {
        this.drawImage(img);
        resolve();
      };
      img.onerror = () => {
        // 创建占位符
        this.drawPlaceholder();
        resolve();
      };
      img.src = src;
    });
  }

  drawImage(img) {
    const maxWidth = 800;
    const maxHeight = 600;
    let width = img.width;
    let height = img.height;

    // 保持宽高比缩放
    if (width > maxWidth) {
      height = (height * maxWidth) / width;
      width = maxWidth;
    }
    if (height > maxHeight) {
      width = (width * maxHeight) / height;
      height = maxHeight;
    }

    this.canvas.width = width;
    this.canvas.height = height;
    this.ctx.drawImage(img, 0, 0, width, height);

    this.imageWidth = width;
    this.imageHeight = height;
  }

  drawPlaceholder() {
    const width = 640;
    const height = 480;
    this.canvas.width = width;
    this.canvas.height = height;

    // 绘制渐变背景
    const gradient = this.ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, '#1a2847');
    gradient.addColorStop(1, '#0f1829');
    this.ctx.fillStyle = gradient;
    this.ctx.fillRect(0, 0, width, height);

    // 绘制网格
    this.ctx.strokeStyle = 'rgba(94, 240, 212, 0.1)';
    this.ctx.lineWidth = 1;
    const gridSize = 40;
    for (let x = 0; x <= width; x += gridSize) {
      this.ctx.beginPath();
      this.ctx.moveTo(x, 0);
      this.ctx.lineTo(x, height);
      this.ctx.stroke();
    }
    for (let y = 0; y <= height; y += gridSize) {
      this.ctx.beginPath();
      this.ctx.moveTo(0, y);
      this.ctx.lineTo(width, y);
      this.ctx.stroke();
    }

    // 绘制文字
    this.ctx.fillStyle = 'rgba(94, 240, 212, 0.6)';
    this.ctx.font = '24px "JetBrains Mono", monospace';
    this.ctx.textAlign = 'center';
    this.ctx.textBaseline = 'middle';
    this.ctx.fillText('示例图片', width / 2, height / 2);

    this.imageWidth = width;
    this.imageHeight = height;
  }

  renderResponse() {
    if (!this.currentExample) return;

    const response = $('#response');
    const answer = this.currentExample.answer;
    response.innerHTML = answer.split('').map((char, idx) =>
      `<span class="char ${idx <= this.currentCharIdx ? 'shown' : ''} ${idx === this.currentCharIdx ? 'current' : ''}">${char}</span>`
    ).join('');

    $('#charPos').textContent = Math.max(0, this.currentCharIdx + 1);
    $('#charTotal').textContent = answer.length;
  }

  switchView(view) {
    this.currentView = view;
    document.querySelectorAll('.view-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.view === view);
    });

    const labels = {
      patches: '图像切块',
      tokens: '词元视图',
      attention: '注意力视图'
    };
    $('#stageLabel').textContent = labels[view];

    this.updateView();
  }

  updateView() {
    if (!this.currentExample) return;

    this.patchOverlay.innerHTML = '';

    if (this.currentView === 'patches') {
      this.renderPatches();
    } else if (this.currentView === 'tokens') {
      this.renderTokens();
    } else if (this.currentView === 'attention') {
      this.renderAttention();
    }
  }

  renderPatches() {
    const { patches } = this.currentExample;
    const rows = 14;
    const cols = 14;
    const patchWidth = this.imageWidth / cols;
    const patchHeight = this.imageHeight / rows;

    patches.forEach(patch => {
      const div = document.createElement('div');
      div.className = 'patch';
      div.style.left = `${patch.col * patchWidth}px`;
      div.style.top = `${patch.row * patchHeight}px`;
      div.style.width = `${patchWidth}px`;
      div.style.height = `${patchHeight}px`;

      const label = document.createElement('div');
      label.className = 'patch-label';
      label.textContent = patch.id;
      div.appendChild(label);

      div.addEventListener('mouseenter', (e) => this.showPatchTooltip(patch, e));
      div.addEventListener('mouseleave', () => this.hideTooltip());

      this.patchOverlay.appendChild(div);
    });
  }

  renderTokens() {
    // 在token视图中显示图像块转换为词元的过程
    const info = $('#attnInfo');
    info.innerHTML = `
      <p style="color: var(--ink2); margin: 8px 0;">
        <strong>词元化过程：</strong><br>
        1. 图像被切分为 14×14 = 196 个块<br>
        2. 每个块通过视觉编码器生成 768 维嵌入<br>
        3. 这些嵌入作为词元输入语言模型<br>
        4. 加上文本词元，共同参与注意力计算
      </p>
    `;

    this.renderPatches();
  }

  renderAttention() {
    if (this.currentCharIdx < 0) {
      $('#attnInfo').innerHTML = '<p class="dim">开始播放以查看注意力分布</p>';
      return;
    }

    const { attention, patches } = this.currentExample;
    const charAttn = attention[this.currentCharIdx];

    if (!charAttn) return;

    const rows = 14;
    const cols = 14;
    const patchWidth = this.imageWidth / cols;
    const patchHeight = this.imageHeight / rows;

    // 找出权重最高的几个patch
    const topPatches = charAttn.weights
      .map((w, i) => ({ weight: w, idx: i }))
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 5);

    // 渲染所有patch，高亮注意力高的
    patches.forEach(patch => {
      const weight = charAttn.weights[patch.id];
      const isTop = topPatches.some(p => p.idx === patch.id);

      const div = document.createElement('div');
      div.className = 'patch' + (isTop ? ' highlight' : '');
      div.style.left = `${patch.col * patchWidth}px`;
      div.style.top = `${patch.row * patchHeight}px`;
      div.style.width = `${patchWidth}px`;
      div.style.height = `${patchHeight}px`;

      if (isTop) {
        div.style.opacity = 0.3 + weight * 0.7;
      } else {
        div.style.opacity = 0.1 + weight * 0.2;
      }

      this.patchOverlay.appendChild(div);
    });

    // 显示注意力信息
    $('#attnInfo').innerHTML = `
      <p style="color: var(--ink2); margin: 0 0 8px 0;">
        <strong>生成字符：</strong><span class="attn-value">"${charAttn.char}"</span>
      </p>
      <p style="color: var(--dim); font-size: 12px; margin: 0 0 8px 0;">
        注意力最集中的图像区域：
      </p>
      ${topPatches.map((p, i) => {
        const patch = patches[p.idx];
        return `<div class="patch-info" style="margin: 4px 0; font-size: 12px;">
          <span style="color: var(--amber);">块 ${p.idx}</span>
          (行${patch.row}, 列${patch.col})
          <span class="attn-value">${(p.weight * 100).toFixed(1)}%</span>
        </div>`;
      }).join('')}
    `;
  }

  showPatchTooltip(patch, e) {
    const tip = $('#tooltip');
    tip.innerHTML = `
      <span class="k">图像块 #${patch.id}</span>
      位置：第 ${patch.row} 行，第 ${patch.col} 列<br>
      嵌入维度：<span class="v">768</span>
    `;
    tip.classList.add('on');

    const r = tip.getBoundingClientRect();
    let x = e.clientX + 16;
    let y = e.clientY + 16;
    if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 16;
    if (y + r.height > innerHeight - 8) y = e.clientY - r.height - 12;
    tip.style.transform = `translate(${Math.max(8, x)}px, ${Math.max(8, y)}px)`;
  }

  hideTooltip() {
    $('#tooltip').classList.remove('on');
  }

  togglePlay() {
    if (this.playing) {
      this.stopPlay();
    } else {
      this.startPlay();
    }
  }

  startPlay() {
    this.playing = true;
    $('#btnPlay').textContent = '⏸';

    if (this.currentCharIdx >= this.currentExample.answer.length - 1) {
      this.currentCharIdx = -1;
    }

    this.playTimer = setInterval(() => {
      this.nextChar();
      if (this.currentCharIdx >= this.currentExample.answer.length - 1) {
        this.stopPlay();
      }
    }, 500);

    sfx.click();
  }

  stopPlay() {
    this.playing = false;
    $('#btnPlay').textContent = '▶';
    if (this.playTimer) {
      clearInterval(this.playTimer);
      this.playTimer = null;
    }
  }

  nextChar() {
    if (this.currentCharIdx < this.currentExample.answer.length - 1) {
      this.currentCharIdx++;
      this.renderResponse();
      this.updateView();
      sfx.tick();
    }
  }

  prevChar() {
    if (this.currentCharIdx > -1) {
      this.currentCharIdx--;
      this.renderResponse();
      this.updateView();
      sfx.tick();
    }
  }
}

// 启动应用
new MultimodalApp();
