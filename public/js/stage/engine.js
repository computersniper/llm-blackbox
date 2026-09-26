// Three.js 舞台：渲染器、泛光、CSS2D 标签、带阻尼的相机、拖动环视、滚轮/捏合换深度、拾取。
import * as THREE from '../vendor/three/three.module.js';
import { EffectComposer } from '../vendor/three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from '../vendor/three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from '../vendor/three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from '../vendor/three/addons/postprocessing/OutputPass.js';
import { CSS2DRenderer, CSS2DObject } from '../vendor/three/addons/renderers/CSS2DRenderer.js';
import { RoomEnvironment } from '../vendor/three/addons/environments/RoomEnvironment.js';

export { THREE, CSS2DObject };

const BG = 0x050b17;

export class Engine {
  constructor(host, { onFrame, onWheelDepth, onHover, onPick } = {}) {
    this.host = host;
    this.onFrame = onFrame;
    this.onWheelDepth = onWheelDepth;
    this.onHover = onHover;
    this.onPick = onPick;
    this.active = false;
    this.pickables = [];

    const small = matchMedia('(max-width: 900px)').matches;
    const r = (this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' }));
    r.setPixelRatio(Math.min(devicePixelRatio || 1, small ? 1.5 : 1.75));
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 1.05;
    host.append(r.domElement);

    const scene = (this.scene = new THREE.Scene());
    scene.background = new THREE.Color(BG);
    scene.fog = new THREE.Fog(BG, 40, 120);
    this.camera = new THREE.PerspectiveCamera(34, 1, 0.004, 400);

    const pmrem = new THREE.PMREMGenerator(r);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environmentIntensity = 0.32;
    scene.add(new THREE.HemisphereLight(0x9fb8ff, 0x0a0f1c, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 1.1);
    key.position.set(8, 14, 12);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x5ef0d4, 0.7);
    rim.position.set(-10, 6, -12);
    scene.add(rim);

    this.composer = new EffectComposer(r);
    this.composer.addPass(new RenderPass(scene, this.camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), small ? 0.5 : 0.62, 0.5, 0.34);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());

    this.css = new CSS2DRenderer();
    this.css.domElement.className = 'css2d';
    host.append(this.css.domElement);

    // 相机：目标位姿 + 阻尼 + 用户拖动的环视偏移
    this.pos = new THREE.Vector3(60, 30, 90);
    this.look = new THREE.Vector3(0, 4, 0);
    this.tPos = this.pos.clone();
    this.tLook = this.look.clone();
    this.yaw = 0; this.pitch = 0; this.tYaw = 0; this.tPitch = 0;
    this.camSpeed = 3.2;

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
  applyInsets() {
    const w = this.w, h = this.h;
    if (!w) return;
    this.camera.setViewOffset(w, h, (this.insetR || 0) / 2, (this.insetB || 0) / 2, w, h);
    this.camera.updateProjectionMatrix();
  }

  // 让宽 width、高 height 的区域刚好装进画面所需的距离
  // 只按没被面板挡住的那块区域来算
  fitDistance(width, height, margin = 1.15) {
    const vw = Math.max(200, this.w - (this.insetR || 0)), vh = Math.max(200, this.h - (this.insetB || 0));
    const vf = (this.camera.fov * Math.PI) / 180;
    const tv = Math.tan(vf / 2) * (vh / this.h);
    const th = Math.tan(vf / 2) * (vw / this.h);
    return Math.max((height * margin) / 2 / tv, (width * margin) / 2 / th);
  }

  setView(pos, look, { snap = false, speed = 3.2, keepOrbit = false } = {}) {
    this.tPos.copy(pos);
    this.tLook.copy(look);
    this.camSpeed = speed;
    if (!keepOrbit) { this.tYaw = 0; this.tPitch = 0; }
    if (snap) { this.pos.copy(pos); this.look.copy(look); }
  }

  project(v) {
    const p = v.clone().project(this.camera);
    const r = this.renderer.domElement.getBoundingClientRect();
    return { x: r.left + ((p.x + 1) / 2) * r.width, y: r.top + ((1 - p.y) / 2) * r.height, behind: p.z > 1 };
  }

  bindInput() {
    const el = this.renderer.domElement;
    let drag = null, moved = 0, pinch = null, wheelAcc = 0, wheelLock = 0;
    el.addEventListener('pointerdown', (e) => {
      if (e.pointerType === 'touch' && pinch) return;
      drag = { x: e.clientX, y: e.clientY, yaw: this.tYaw, pitch: this.tPitch };
      moved = 0;
      this.host.classList.add('grabbing');
    });
    addEventListener('pointermove', (e) => {
      if (drag) {
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        moved = Math.max(moved, Math.abs(dx) + Math.abs(dy));
        this.tYaw = Math.max(-0.9, Math.min(0.9, drag.yaw - dx * 0.005));
        this.tPitch = Math.max(-0.35, Math.min(0.5, drag.pitch + dy * 0.004));
      }
    });
    addEventListener('pointerup', (e) => {
      if (drag && moved < 5 && e.target === el) this.pick(e, true);
      drag = null;
      this.host.classList.remove('grabbing');
    });
    el.addEventListener('pointermove', (e) => { if (!drag) this.pick(e, false); });
    el.addEventListener('pointerleave', () => this.onHover?.(null));
    el.addEventListener('wheel', (e) => {
      e.preventDefault();
      const now = performance.now();
      if (now < wheelLock) return;
      wheelAcc += e.deltaY;
      if (Math.abs(wheelAcc) > 90) {
        this.onWheelDepth?.(wheelAcc < 0 ? 1 : -1);
        wheelAcc = 0;
        wheelLock = now + 550;
      }
    }, { passive: false });
    el.addEventListener('touchstart', (e) => { if (e.touches.length === 2) { pinch = this.touchDist(e); drag = null; } }, { passive: true });
    el.addEventListener('touchmove', (e) => {
      if (pinch && e.touches.length === 2) {
        const d = this.touchDist(e);
        if (d / pinch > 1.35) { this.onWheelDepth?.(1); pinch = d; }
        else if (d / pinch < 0.74) { this.onWheelDepth?.(-1); pinch = d; }
      }
    }, { passive: true });
    el.addEventListener('touchend', (e) => { if (e.touches.length < 2) pinch = null; });
  }

  touchDist(e) { const [a, b] = e.touches; return Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY); }

  pick(e, click) {
    const r = this.renderer.domElement.getBoundingClientRect();
    this.ndc.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
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
    if (this.active) {
      this.onFrame?.(dt, this.clock);
      const k = 1 - Math.exp(-dt * this.camSpeed);
      this.pos.lerp(this.tPos, k);
      this.look.lerp(this.tLook, k);
      this.yaw += (this.tYaw - this.yaw) * (1 - Math.exp(-dt * 6));
      this.pitch += (this.tPitch - this.pitch) * (1 - Math.exp(-dt * 6));
      const off = this.pos.clone().sub(this.look);
      const sph = new THREE.Spherical().setFromVector3(off);
      sph.theta += this.yaw;
      sph.phi = Math.max(0.15, Math.min(Math.PI - 0.15, sph.phi - this.pitch));
      this.camera.position.copy(this.look).add(new THREE.Vector3().setFromSpherical(sph));
      this.camera.lookAt(this.look);
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
