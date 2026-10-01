// Three.js 舞台：渲染器、泛光、CSS2D 标签、跟随镜头 + 自由视角（拖动 / 滚轮 / WASD / 双击聚焦）、拾取。
import * as THREE from '../vendor/three/three.module.js';
import { EffectComposer } from '../vendor/three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from '../vendor/three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from '../vendor/three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from '../vendor/three/addons/postprocessing/OutputPass.js';
import { CSS2DRenderer, CSS2DObject } from '../vendor/three/addons/renderers/CSS2DRenderer.js';
import { RoomEnvironment } from '../vendor/three/addons/environments/RoomEnvironment.js';

export { THREE, CSS2DObject };

const BG = 0x050b17;
const UP = new THREE.Vector3(0, 1, 0);

export class Engine {
  constructor(host, { onFrame, onHover, onPick, onFreeChange } = {}) {
    this.host = host;
    this.onFrame = onFrame;
    this.onHover = onHover;
    this.onPick = onPick;
    this.onFreeChange = onFreeChange;
    this.active = false;
    this.pickables = [];

    const small = matchMedia('(max-width: 900px)').matches;
    const r = (this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' }));
    r.setPixelRatio(Math.min(devicePixelRatio || 1, small ? 1.5 : 1.75));
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 0.88;
    host.append(r.domElement);

    const scene = (this.scene = new THREE.Scene());
    scene.background = new THREE.Color(BG);
    scene.fog = new THREE.Fog(BG, 70, 240);
    this.camera = new THREE.PerspectiveCamera(34, 1, 0.004, 400);

    const pmrem = new THREE.PMREMGenerator(r);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environmentIntensity = 0.3;
    scene.add(new THREE.HemisphereLight(0x9fb8ff, 0x0a0f1c, 0.5));
    const key = new THREE.DirectionalLight(0xffffff, 1.0);
    key.position.set(8, 14, 12);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x5ef0d4, 0.6);
    rim.position.set(-10, 6, -12);
    scene.add(rim);

    this.composer = new EffectComposer(r);
    this.composer.addPass(new RenderPass(scene, this.camera));
    // 阈值调高：只有真正亮的东西才会泛光
    this.bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), small ? 0.36 : 0.42, 0.45, 0.52);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());

    this.css = new CSS2DRenderer();
    this.css.domElement.className = 'css2d';
    host.append(this.css.domElement);

    // 跟随镜头：目标位姿 + 阻尼
    this.pos = new THREE.Vector3(60, 30, 90);
    this.look = new THREE.Vector3(0, 4, 0);
    this.tPos = this.pos.clone();
    this.tLook = this.look.clone();
    this.camSpeed = 3.2;
    // 自由视角：围绕 fTarget 的球坐标；键盘移动的是 fTarget 本身
    this.free = false;
    this.fTarget = new THREE.Vector3();
    this.fSph = new THREE.Spherical(10, 1, 0);
    this.fGoal = null;
    this.keys = new Set();

    this.raycaster = new THREE.Raycaster();
    this.ndc = new THREE.Vector2();
    this.bindInput();

    new ResizeObserver(() => this.resize()).observe(host);
    this.resize();
    this.last = performance.now();
    this.clock = 0;
    requestAnimationFrame((t) => this.loop(t));
  }

  resize() {
    const w = Math.max(1, this.host.clientWidth), h = Math.max(1, this.host.clientHeight);
    this.w = w; this.h = h;
    this.renderer.setSize(w, h, false);
    this.composer.setSize(w, h);
    this.bloom.resolution.set(w / 2, h / 2);
    this.css.setSize(w, h);
    this.camera.aspect = w / h;
    this.applyInsets();
  }

  // 右侧调试面板、底部控制条会挡住一部分画面：把投影中心挪到空出来的区域
  setInsets(right, bottom) { this.insetR = right; this.insetB = bottom; this.applyInsets(); }
  // 算式板之类的浮层再额外挡住上 / 下一块（像素）。边距在每帧里慢慢过渡，画面不会跳
  setOverlay(top = 0, bottom = 0) { this.ovGoal = [top, bottom]; }
  get padT() { return this.ov?.[0] || 0; }
  get padB() { return Math.max(this.insetB || 0, this.ov?.[1] || 0); }
  applyInsets() {
    const w = this.w, h = this.h;
    if (!w) return;
    this.camera.setViewOffset(w, h, (this.insetR || 0) / 2, (this.padB - this.padT) / 2, w, h);
    this.camera.updateProjectionMatrix();
  }

  // 让宽 width、高 height 的区域刚好装进画面所需的距离（只按没被面板挡住的那块区域来算）
  fitDistance(width, height, margin = 1.15) {
    const vw = Math.max(160, this.w - (this.insetR || 0)), vh = Math.max(140, this.h - this.padB - this.padT);
    const vf = (this.camera.fov * Math.PI) / 180;
    const tv = Math.tan(vf / 2) * (vh / this.h);
    const th = Math.tan(vf / 2) * (vw / this.h);
    return Math.max((height * margin) / 2 / tv, (width * margin) / 2 / th);
  }

  setView(pos, look, { snap = false, speed = 3.2 } = {}) {
    this.tPos.copy(pos);
    this.tLook.copy(look);
    this.camSpeed = speed;
    if (snap) { this.pos.copy(pos); this.look.copy(look); }
  }

  /* ---------------- 自由视角 ---------------- */

  enterFree() {
    if (this.free) return;
    this.free = true;
    this.fTarget.copy(this.look);
    this.fSph.setFromVector3(this.camera.position.clone().sub(this.look));
    this.fSph.radius = Math.max(0.12, this.fSph.radius);
    this.onFreeChange?.(true);
  }

  exitFree() {
    if (!this.free) return;
    this.free = false;
    this.fGoal = null;
    this.pos.copy(this.camera.position);
    this.look.copy(this.fTarget);
    this.onFreeChange?.(false);
  }

  orbit(dx, dy) {
    this.fSph.theta -= dx * 0.006;
    this.fSph.phi = Math.max(0.04, Math.min(Math.PI - 0.04, this.fSph.phi - dy * 0.006));
    this.fGoal = null;
  }

  pan(dx, dy) {
    const k = (2 * this.fSph.radius * Math.tan((this.camera.fov * Math.PI) / 360)) / this.h;
    const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 0);
    const up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 1);
    this.fTarget.addScaledVector(right, -dx * k).addScaledVector(up, dy * k);
    this.fGoal = null;
  }

  // f < 1 靠近，f > 1 远离；推到最近还继续推，就连目标点一起往前走，可以穿过物体
  dolly(f) {
    const MIN = 0.12;
    const r = this.fSph.radius * f;
    if (r < MIN && f < 1) {
      const fw = this.fTarget.clone().sub(this.camera.position).normalize();
      this.fTarget.addScaledVector(fw, Math.max(0.03, this.fSph.radius * (1 - f)));
      this.fSph.radius = MIN;
    } else this.fSph.radius = Math.max(MIN, Math.min(300, r));
    this.fGoal = null;
  }

  // 双击：飞到被点中的物体跟前
  focusAt(e) {
    this.setNdc(e);
    this.raycaster.setFromCamera(this.ndc, this.camera);
    const hit = this.raycaster.intersectObjects(this.scene.children, true).find((h) => h.object.visible && h.object.isMesh && (h.object.material?.opacity ?? 1) > 0.05 && !h.object.userData.noFocus);
    if (!hit) return;
    this.enterFree();
    this.fGoal = { target: hit.point.clone(), radius: Math.max(0.35, Math.min(this.fSph.radius, hit.distance * 0.45)) };
  }

  setNdc(e) {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
  }

  bindInput() {
    const el = this.renderer.domElement;
    el.style.touchAction = 'none';
    const pts = new Map();
    let mode = null, moved = 0, last = null, pinchD = 0, pinchMid = null;
    const two = () => { const [a, b] = [...pts.values()]; return { d: Math.hypot(a.x - b.x, a.y - b.y), m: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 } }; };
    el.addEventListener('contextmenu', (e) => e.preventDefault());
    el.addEventListener('pointerdown', (e) => {
      el.setPointerCapture(e.pointerId);
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pts.size === 1) {
        mode = e.button === 2 || e.button === 1 || e.shiftKey ? 'pan' : 'orbit';
        last = { x: e.clientX, y: e.clientY };
        moved = 0;
        this.downButton = e.button;
      } else if (pts.size === 2) {
        mode = 'pinch';
        ({ d: pinchD, m: pinchMid } = two());
      }
      this.host.classList.add('grabbing');
      this.onHover?.(null);
    });
    el.addEventListener('pointermove', (e) => {
      if (!pts.has(e.pointerId)) { this.pick(e, false); return; }
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (mode === 'pinch' && pts.size >= 2) {
        const { d, m } = two();
        this.enterFree();
        if (d > 0 && pinchD > 0) this.dolly(pinchD / d);
        this.pan(m.x - pinchMid.x, m.y - pinchMid.y);
        pinchD = d; pinchMid = m; moved += 10;
        return;
      }
      const dx = e.clientX - last.x, dy = e.clientY - last.y;
      last = { x: e.clientX, y: e.clientY };
      moved += Math.abs(dx) + Math.abs(dy);
      if (moved < 4) return;
      this.enterFree();
      if (mode === 'pan') this.pan(dx, dy); else this.orbit(dx, dy);
    });
    const end = (e) => {
      if (!pts.has(e.pointerId)) return;
      pts.delete(e.pointerId);
      if (pts.size === 0) {
        if (moved < 5 && e.type === 'pointerup' && this.downButton === 0) this.pick(e, true);
        mode = null;
        this.host.classList.remove('grabbing');
      } else if (pts.size === 1) { mode = 'orbit'; last = [...pts.values()][0]; }
    };
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);
    el.addEventListener('pointerleave', () => { if (!pts.size) this.onHover?.(null); });
    el.addEventListener('wheel', (e) => {
      e.preventDefault();
      this.enterFree();
      this.dolly(Math.exp(Math.max(-60, Math.min(60, e.deltaY)) * 0.0022));
    }, { passive: false });
    el.addEventListener('dblclick', (e) => this.focusAt(e));
    addEventListener('keydown', (e) => {
      if (!this.active || e.ctrlKey || e.metaKey || e.altKey || e.target.closest?.('input, textarea, [contenteditable]')) return;
      const k = e.key.toLowerCase();
      if (k.length === 1 && 'wasdqe'.includes(k)) { this.keys.add(k); e.preventDefault(); }
      if (k === 'shift') this.keys.add('shift');
    });
    addEventListener('keyup', (e) => this.keys.delete(e.key.toLowerCase()));
    addEventListener('blur', () => this.keys.clear());
  }

  pick(e, click) {
    this.setNdc(e);
    this.raycaster.setFromCamera(this.ndc, this.camera);
    const hits = this.raycaster.intersectObjects(this.pickables.filter((o) => o.visible && o.userData.pickOn !== false), false);
    const hit = hits.find((h) => h.object.material?.opacity !== 0) || null;
    let info = null;
    if (hit) {
      const pk = hit.object.userData.pick;
      info = typeof pk === 'function' ? pk(hit) : pk;
    }
    if (click) this.onPick?.(info, e);
    else this.onHover?.(info, e);
    this.renderer.domElement.style.cursor = info?.click ? 'pointer' : '';
  }

  loop(t) {
    const dt = Math.min(0.05, (t - this.last) / 1000);
    this.last = t;
    this.clock += dt;
    if (this.ovGoal) {
      const ov = (this.ov ||= [0, 0]), k = 1 - Math.exp(-dt * 5);
      let moved = false;
      for (let i = 0; i < 2; i++) {
        const d = this.ovGoal[i] - ov[i];
        if (Math.abs(d) > 0.5) { ov[i] += d * k; moved = true; } else if (d) { ov[i] = this.ovGoal[i]; moved = true; }
      }
      if (moved) this.applyInsets();
    }
    if (this.active) {
      this.onFrame?.(dt, this.clock);
      const moveKeys = ['w', 'a', 's', 'd', 'q', 'e'].filter((k) => this.keys.has(k));
      if (moveKeys.length) this.enterFree();
      if (this.free) {
        if (moveKeys.length) {
          const sp = Math.max(0.35, Math.min(25, this.fSph.radius * 0.9)) * (this.keys.has('shift') ? 3 : 1) * dt;
          const fw = this.fTarget.clone().sub(this.camera.position).normalize();
          const right = new THREE.Vector3().crossVectors(fw, UP).normalize();
          const mv = new THREE.Vector3();
          if (this.keys.has('w')) mv.add(fw);
          if (this.keys.has('s')) mv.sub(fw);
          if (this.keys.has('d')) mv.add(right);
          if (this.keys.has('a')) mv.sub(right);
          if (this.keys.has('e')) mv.y += 1;
          if (this.keys.has('q')) mv.y -= 1;
          if (mv.lengthSq() > 0) { this.fTarget.addScaledVector(mv.normalize(), sp); this.fGoal = null; }
        }
        if (this.fGoal) {
          const k = 1 - Math.exp(-dt * 4);
          this.fTarget.lerp(this.fGoal.target, k);
          this.fSph.radius += (this.fGoal.radius - this.fSph.radius) * k;
          if (this.fTarget.distanceTo(this.fGoal.target) < 0.002) this.fGoal = null;
        }
        this.camera.position.copy(this.fTarget).add(new THREE.Vector3().setFromSpherical(this.fSph));
        this.camera.lookAt(this.fTarget);
      } else {
        const k = 1 - Math.exp(-dt * this.camSpeed);
        this.pos.lerp(this.tPos, k);
        this.look.lerp(this.tLook, k);
        this.camera.position.copy(this.pos);
        this.camera.lookAt(this.look);
      }
      this.composer.render();
      this.css.render(this.scene, this.camera);
    }
    requestAnimationFrame((tt) => this.loop(tt));
  }
}

// ---------------------------------------------------------------- 小工具

const texCache = new Map();
export function textTexture(text, { color = '#e9eff9', bg = null, font = '600 76px "PingFang SC","Microsoft YaHei","Noto Sans SC",sans-serif', w = 256, h = 160, border = null } = {}) {
  const key = [text, color, bg, font, w, h, border].join('|');
  if (texCache.has(key)) return texCache.get(key);
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const g = c.getContext('2d');
  if (bg) { g.fillStyle = bg; g.fillRect(0, 0, w, h); }
  if (border) { g.strokeStyle = border; g.lineWidth = 6; g.strokeRect(3, 3, w - 6, h - 6); }
  g.fillStyle = color;
  g.textAlign = 'center';
  g.textBaseline = 'middle';
  let size = parseInt(font.match(/(\d+)px/)[1], 10);
  const f = (s) => font.replace(/\d+px/, `${s}px`);
  g.font = f(size);
  while (g.measureText(text).width > w * 0.86 && size > 18) { size -= 4; g.font = f(size); }
  g.fillText(text, w / 2, h / 2 + size * 0.04);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = 4;
  texCache.set(key, t);
  return t;
}

export function label(html, cls = 'lbl') {
  const el = document.createElement('div');
  el.className = cls;
  el.innerHTML = html;
  const o = new CSS2DObject(el);
  o.el = el;
  return o;
}

export const easeOut = (t) => 1 - Math.pow(1 - Math.min(1, Math.max(0, t)), 3);
export const easeInOut = (t) => { t = Math.min(1, Math.max(0, t)); return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2; };
export const seg = (p, a, b) => Math.min(1, Math.max(0, (p - a) / (b - a)));
