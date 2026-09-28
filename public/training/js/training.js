import { Background } from '../../js/bg.js';
import { sfx, setSound, soundOn } from '../../js/audio.js';
import { $, sleep } from '../../js/ui.js';

// 从零训练场景的模拟数据
const FROM_SCRATCH_DATA = {
  totalSteps: 3000,
  checkpoints: [
    { step: 0, loss: 8.524, perplexity: 5041, sample: '��$#@春天�%&*花朵��^!@#天气', quality: 'gibberish' },
    { step: 100, loss: 7.234, perplexity: 1384, sample: '春天春天春天春天春天春天春天春天', quality: 'gibberish' },
    { step: 200, loss: 6.012, perplexity: 408, sample: '春天是一个季节，有很多花', quality: 'improving' },
    { step: 300, loss: 5.123, perplexity: 168, sample: '春天来了，花儿开了，鸟儿叫了', quality: 'improving' },
    { step: 500, loss: 4.456, perplexity: 86, sample: '春风拂面花满园，柳绿莺歌燕舞欢', quality: 'improving' },
    { step: 800, loss: 3.678, perplexity: 40, sample: '春回大地万物苏，桃红柳绿映江湖', quality: 'good' },
    { step: 1200, loss: 2.934, perplexity: 19, sample: '东风送暖入屠苏，春色满园关不住', quality: 'good' },
    { step: 1800, loss: 2.456, perplexity: 12, sample: '春江潮水连海平，海上明月共潮生\n滟滟随波千万里，何处春江无月明', quality: 'good' },
    { step: 2500, loss: 2.187, perplexity: 9, sample: '春眠不觉晓，处处闻啼鸟\n夜来风雨声，花落知多少', quality: 'good' },
    { step: 3000, loss: 2.089, perplexity: 8, sample: '碧玉妆成一树高，万条垂下绿丝绦\n不知细叶谁裁出，二月春风似剪刀', quality: 'good' }
  ]
};

// 一步训练场景的模拟数据
const ONE_STEP_DATA = {
  phases: [
    { name: 'forward', label: '前向传播', duration: 2000 },
    { name: 'backward', label: '反向传播', duration: 3000 },
    { name: 'update', label: '权重更新', duration: 1500 }
  ],
  layers: [
    'Embedding',
    'Layer 0', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4',
    'Layer 5', 'Layer 6', 'Layer 7', 'Layer 8', 'Layer 9',
    'Layer 10', 'Layer 11', 'Layer 12', 'Layer 13', 'Layer 14',
    'Layer 15', 'Layer 16', 'Layer 17', 'Layer 18', 'Layer 19',
    'Layer 20', 'Layer 21', 'Layer 22', 'Layer 23', 'Layer 24',
    'Layer 25', 'Layer 26', 'Layer 27', 'Output'
  ]
};

let currentScenario = null;
let scratchState = { step: 0, playing: false, speed: 1 };
let oneStepState = { phase: 0, layer: 0, playing: false };
let scratchEngine = null, oneStepEngine = null;

/* ---------------------------------------------------------------- 启动 */

async function boot() {
  new Background($('#bg'));

  // 声音按钮
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

  // 场景卡片点击
  document.querySelectorAll('.scenario-card').forEach((card) => {
    const scenario = card.dataset.scenario;
    card.addEventListener('click', () => enterScenario(scenario));
    card.querySelector('.scenario-btn').addEventListener('click', (e) => {
      e.stopPropagation();
      enterScenario(scenario);
    });
  });

  // 从零训练场景控制
  $('#btnBackScratch').addEventListener('click', () => exitScenario());
  $('#btnPlayScratch').addEventListener('click', () => playScratch());
  $('#btnPauseScratch').addEventListener('click', () => pauseScratch());
  $('#btnStepScratch').addEventListener('click', () => stepScratch());
  $('#btnResetScratch').addEventListener('click', () => resetScratch());

  document.querySelectorAll('#stageFromScratch .speed-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      const speed = Number(btn.dataset.speed);
      scratchState.speed = speed;
      document.querySelectorAll('#stageFromScratch .speed-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });

  // 一步训练场景控制
  $('#btnBackOneStep').addEventListener('click', () => exitScenario());
  $('#btnPlayOneStep').addEventListener('click', () => playOneStep());
  $('#btnPauseOneStep').addEventListener('click', () => pauseOneStep());
  $('#btnStepOneStep').addEventListener('click', () => stepOneStep());
  $('#btnResetOneStep').addEventListener('click', () => resetOneStep());
}

/* ---------------------------------------------------------------- 场景切换 */

function enterScenario(scenario) {
  currentScenario = scenario;
  document.body.classList.remove('mode-overview');
  document.body.classList.add('mode-stage');

  $('#overview').hidden = true;

  if (scenario === 'from-scratch') {
    $('#stageFromScratch').hidden = false;
    initScratchEngine();
    sfx.dive();
  } else if (scenario === 'one-step') {
    $('#stageOneStep').hidden = false;
    initOneStepEngine();
    sfx.dive();
  }
}

function exitScenario() {
  document.body.classList.remove('mode-stage');
  document.body.classList.add('mode-overview');

  $('#stageFromScratch').hidden = true;
  $('#stageOneStep').hidden = true;
  $('#overview').hidden = false;

  if (scratchState.playing) pauseScratch();
  if (oneStepState.playing) pauseOneStep();

  currentScenario = null;
  sfx.rise();
}

/* ---------------------------------------------------------------- 从零训练场景 */

async function initScratchEngine() {
  if (scratchEngine) return;

  const { Engine } = await import('../../js/stage/engine.js');
  const container = $('#glScratch');
  scratchEngine = new Engine(container, {
    onFrame: (dt, t) => updateScratchViz(dt, t),
    onHover: () => {},
    onPick: () => {},
    onFreeChange: () => {}
  });
  scratchEngine.active = true;

  // 初始化 Three.js 场景
  await setupScratchScene();
}

async function setupScratchScene() {
  if (!scratchEngine) return;

  const THREE = scratchEngine.THREE;
  const scene = scratchEngine.scene;

  // 添加训练可视化元素（占位，后续可扩展）
  const geometry = new THREE.BoxGeometry(2, 2, 2);
  const material = new THREE.MeshBasicMaterial({
    color: 0x5ef0d4,
    wireframe: true,
    transparent: true,
    opacity: 0.6
  });
  const cube = new THREE.Mesh(geometry, material);
  scene.add(cube);

  scratchEngine.setView(
    new THREE.Vector3(5, 3, 5),
    new THREE.Vector3(0, 0, 0),
    { snap: true }
  );
}

function updateScratchViz(dt, t) {
  // 动画更新逻辑
  if (scratchEngine && scratchEngine.scene) {
    const cube = scratchEngine.scene.children.find(c => c.type === 'Mesh');
    if (cube) {
      cube.rotation.y += dt * 0.5;
      cube.rotation.x += dt * 0.3;
    }
  }
}

function playScratch() {
  scratchState.playing = true;
  $('#btnPlayScratch').hidden = true;
  $('#btnPauseScratch').hidden = false;

  runScratchLoop();
}

function pauseScratch() {
  scratchState.playing = false;
  $('#btnPlayScratch').hidden = false;
  $('#btnPauseScratch').hidden = true;
}

async function runScratchLoop() {
  while (scratchState.playing && scratchState.step < FROM_SCRATCH_DATA.totalSteps) {
    await stepScratch();
    await sleep(100 / scratchState.speed);
  }

  if (scratchState.step >= FROM_SCRATCH_DATA.totalSteps) {
    pauseScratch();
  }
}

async function stepScratch() {
  scratchState.step += 10;
  if (scratchState.step > FROM_SCRATCH_DATA.totalSteps) {
    scratchState.step = FROM_SCRATCH_DATA.totalSteps;
  }

  updateScratchUI();

  // 检查是否到达检查点
  const checkpoint = FROM_SCRATCH_DATA.checkpoints.find(cp => cp.step === scratchState.step);
  if (checkpoint) {
    sfx.discover();
    updateSample(checkpoint);
  }
}

function updateScratchUI() {
  const progress = scratchState.step / FROM_SCRATCH_DATA.totalSteps;
  $('#currentStep').textContent = scratchState.step;
  $('#totalSteps').textContent = FROM_SCRATCH_DATA.totalSteps;
  $('#progressFill').style.width = `${progress * 100}%`;

  // 插值计算当前指标
  const cp = FROM_SCRATCH_DATA.checkpoints;
  let loss, perplexity;

  for (let i = 0; i < cp.length - 1; i++) {
    if (scratchState.step >= cp[i].step && scratchState.step <= cp[i + 1].step) {
      const t = (scratchState.step - cp[i].step) / (cp[i + 1].step - cp[i].step);
      loss = cp[i].loss + (cp[i + 1].loss - cp[i].loss) * t;
      perplexity = Math.round(cp[i].perplexity + (cp[i + 1].perplexity - cp[i].perplexity) * t);
      break;
    }
  }

  if (loss !== undefined) {
    $('#lossValue').textContent = loss.toFixed(3);
    $('#perplexity').textContent = perplexity;
  }

  // 学习率衰减
  const lr = 3e-4 * Math.max(0.1, 1 - progress * 0.7);
  $('#lrValue').textContent = lr.toExponential(1);

  // 用时
  const seconds = Math.floor(scratchState.step * 0.5);
  $('#elapsed').textContent = seconds < 60 ? `${seconds}s` : `${Math.floor(seconds / 60)}m${seconds % 60}s`;
}

function updateSample(checkpoint) {
  const sampleText = $('#sampleText');
  const qualityClass = checkpoint.quality;

  sampleText.innerHTML = `<span class="${qualityClass}">${checkpoint.sample.replace(/\n/g, '<br>')}</span>`;
}

function resetScratch() {
  pauseScratch();
  scratchState.step = 0;
  updateScratchUI();
  $('#sampleText').innerHTML = '<span class="placeholder">训练开始后，每 100 步会展示模型的输出…</span>';
  sfx.click();
}

/* ---------------------------------------------------------------- 一步训练场景 */

async function initOneStepEngine() {
  if (oneStepEngine) return;

  const { Engine } = await import('../../js/stage/engine.js');
  const container = $('#glOneStep');
  oneStepEngine = new Engine(container, {
    onFrame: (dt, t) => updateOneStepViz(dt, t),
    onHover: () => {},
    onPick: () => {},
    onFreeChange: () => {}
  });
  oneStepEngine.active = true;

  await setupOneStepScene();
  updateOneStepUI();
}

async function setupOneStepScene() {
  if (!oneStepEngine) return;

  const THREE = oneStepEngine.THREE;
  const scene = oneStepEngine.scene;

  // 创建神经网络层可视化
  const layerGeometry = new THREE.PlaneGeometry(3, 0.2);
  const layerMaterial = new THREE.MeshBasicMaterial({
    color: 0x6b9bff,
    transparent: true,
    opacity: 0.7,
    side: THREE.DoubleSide
  });

  for (let i = 0; i < 5; i++) {
    const layer = new THREE.Mesh(layerGeometry, layerMaterial.clone());
    layer.position.z = i * 0.5 - 1;
    layer.userData.layerIndex = i;
    scene.add(layer);
  }

  oneStepEngine.setView(
    new THREE.Vector3(4, 2, 4),
    new THREE.Vector3(0, 0, 0),
    { snap: true }
  );

  // 初始化梯度画布
  initGradientCanvas();
}

function initGradientCanvas() {
  const canvas = $('#gradCanvas');
  const ctx = canvas.getContext('2d');

  // 绘制初始梯度分布
  ctx.fillStyle = '#0a111f';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  // 绘制梯度柱状图
  const bars = 50;
  const barWidth = canvas.width / bars;

  for (let i = 0; i < bars; i++) {
    const height = Math.random() * canvas.height * 0.8;
    const grad = ctx.createLinearGradient(0, canvas.height, 0, canvas.height - height);
    grad.addColorStop(0, '#5ef0d4');
    grad.addColorStop(1, '#6b9bff');

    ctx.fillStyle = grad;
    ctx.fillRect(i * barWidth, canvas.height - height, barWidth - 1, height);
  }
}

function updateOneStepViz(dt, t) {
  if (!oneStepEngine || !oneStepEngine.scene) return;

  const phase = ONE_STEP_DATA.phases[oneStepState.phase];

  // 更新层的高亮
  oneStepEngine.scene.children.forEach((obj, idx) => {
    if (obj.type === 'Mesh' && obj.geometry.type === 'PlaneGeometry') {
      const isActive = phase && phase.name === 'backward' ?
        (obj.userData.layerIndex <= oneStepState.layer / 6) :
        (obj.userData.layerIndex >= 5 - oneStepState.layer / 6);

      obj.material.opacity = isActive ? 0.9 : 0.3;
      obj.material.color.setHex(isActive ? 0x5ef0d4 : 0x6b9bff);
    }
  });
}

function playOneStep() {
  oneStepState.playing = true;
  $('#btnPlayOneStep').hidden = true;
  $('#btnPauseOneStep').hidden = false;

  runOneStepLoop();
}

function pauseOneStep() {
  oneStepState.playing = false;
  $('#btnPlayOneStep').hidden = false;
  $('#btnPauseOneStep').hidden = true;
}

async function runOneStepLoop() {
  while (oneStepState.playing) {
    const hasNext = await stepOneStep();
    if (!hasNext) {
      pauseOneStep();
      break;
    }
    await sleep(150);
  }
}

async function stepOneStep() {
  const phases = ONE_STEP_DATA.phases;
  const currentPhase = phases[oneStepState.phase];

  if (currentPhase.name === 'forward') {
    oneStepState.layer++;
    if (oneStepState.layer >= ONE_STEP_DATA.layers.length) {
      oneStepState.layer = 0;
      oneStepState.phase = 1; // 进入反向传播
      sfx.dive();
    }
  } else if (currentPhase.name === 'backward') {
    oneStepState.layer++;
    if (oneStepState.layer >= ONE_STEP_DATA.layers.length) {
      oneStepState.layer = 0;
      oneStepState.phase = 2; // 进入权重更新
      sfx.click();
    }
  } else if (currentPhase.name === 'update') {
    oneStepState.layer++;
    if (oneStepState.layer >= ONE_STEP_DATA.layers.length) {
      return false; // 完成
    }
  }

  updateOneStepUI();
  return true;
}

function updateOneStepUI() {
  const phase = ONE_STEP_DATA.phases[oneStepState.phase];
  const layerName = ONE_STEP_DATA.layers[oneStepState.layer] || 'Output';

  // 更新阶段指示器
  $('#phaseLabel').textContent = phase.label;
  document.querySelectorAll('.phase-dot').forEach((dot, idx) => {
    dot.classList.toggle('active', idx === oneStepState.phase);
  });

  // 更新当前层
  $('#currentLayer').textContent = layerName;

  // 更新梯度可视化
  updateGradientViz();

  // 更新权重细节（模拟数据）
  if (phase.name === 'update') {
    const current = 0.0234 + (Math.random() - 0.5) * 0.01;
    const grad = -0.0012 + (Math.random() - 0.5) * 0.0005;
    const momentum = grad * 0.9;
    const variance = grad * grad;
    const newVal = current - 0.0001 * momentum / Math.sqrt(variance + 1e-8);

    $('#weightCurrent').textContent = current.toFixed(4);
    $('#weightGrad').textContent = grad.toFixed(4);
    $('#weightMomentum').textContent = momentum.toFixed(4);
    $('#weightVariance').textContent = variance.toExponential(1);
    $('#weightNew').textContent = newVal.toFixed(4);
  }
}

function updateGradientViz() {
  const canvas = $('#gradCanvas');
  const ctx = canvas.getContext('2d');

  ctx.fillStyle = '#0a111f';
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const bars = 50;
  const barWidth = canvas.width / bars;
  const scale = Math.max(0.3, 1 - oneStepState.layer / ONE_STEP_DATA.layers.length);

  for (let i = 0; i < bars; i++) {
    const height = (Math.random() * canvas.height * 0.8) * scale;
    const grad = ctx.createLinearGradient(0, canvas.height, 0, canvas.height - height);

    if (oneStepState.phase === 1) {
      grad.addColorStop(0, '#ffb65c');
      grad.addColorStop(1, '#ff6b93');
    } else {
      grad.addColorStop(0, '#5ef0d4');
      grad.addColorStop(1, '#6b9bff');
    }

    ctx.fillStyle = grad;
    ctx.fillRect(i * barWidth, canvas.height - height, barWidth - 1, height);
  }
}

function resetOneStep() {
  pauseOneStep();
  oneStepState.phase = 0;
  oneStepState.layer = 0;
  updateOneStepUI();
  sfx.click();
}

/* ---------------------------------------------------------------- 启动 */

boot();
