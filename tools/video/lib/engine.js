// 电影用的渲染器：接口和站点的 Engine 一样（scene / camera / bloom / pickables / fitDistance），
// 这样 public/js/stage/ 里的机器可以原样复用；但没有自己的 rAF 循环、没有交互，
// 每一帧由 film.js 显式调用 render()，逐帧渲染完全确定。
import * as THREE from '/public/js/vendor/three/three.module.js';
import { EffectComposer } from '/public/js/vendor/three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from '/public/js/vendor/three/addons/postprocessing/RenderPass.js';
import { ShaderPass } from '/public/js/vendor/three/addons/postprocessing/ShaderPass.js';
import { UnrealBloomPass } from '/public/js/vendor/three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from '/public/js/vendor/three/addons/postprocessing/OutputPass.js';
import { CSS2DRenderer } from '/public/js/vendor/three/addons/renderers/CSS2DRenderer.js';
import { RoomEnvironment } from '/public/js/vendor/three/addons/environments/RoomEnvironment.js';

export { THREE };

export const BG = 0x03070f;

// 景深：按深度缓冲做一个轻量的“散景”模糊（圆盘采样），焦点平面以外逐渐变虚。
// 只在特写镜头里打开（strength > 0），其余时候直接跳过这一趟。
const DofShader = {
  uniforms: {
    tDiffuse: { value: null },
    tDepth: { value: null },
    focus: { value: 10 },      // 焦点距离（世界单位）
    range: { value: 6 },       // 景深范围：离焦点越远越虚
    maxBlur: { value: 0 },     // 最大模糊半径（像素）
    cameraNear: { value: 0.01 },
    cameraFar: { value: 400 },
    res: { value: new THREE.Vector2(1920, 1080) },
  },
  vertexShader: 'varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }',
  fragmentShader: `
    #include <packing>
    uniform sampler2D tDiffuse; uniform sampler2D tDepth;
    uniform float focus, range, maxBlur, cameraNear, cameraFar; uniform vec2 res;
    varying vec2 vUv;
    float viewZ(vec2 uv) { float d = texture2D(tDepth, uv).x; return -perspectiveDepthToViewZ(d, cameraNear, cameraFar); }
    float coc(vec2 uv) { float z = viewZ(uv); return clamp(abs(z - focus) / range, 0.0, 1.0); }
    void main() {
      vec4 base = texture2D(tDiffuse, vUv);
      float c = coc(vUv);
      if (maxBlur < 0.01 || c < 0.02) { gl_FragColor = base; return; }
      vec3 acc = base.rgb; float wsum = 1.0;
      const int N = 28;
      float r = c * maxBlur;
      for (int i = 0; i < N; i++) {
        float a = float(i) * 2.39996;            // 黄金角螺旋
        float rr = sqrt((float(i) + 0.5) / float(N));
        vec2 o = vec2(cos(a), sin(a)) * rr * r / res;
        float cs = coc(vUv + o);
        float w = cs + 0.15;                      // 前景清晰的像素不往背景里渗
        acc += texture2D(tDiffuse, vUv + o).rgb * w; wsum += w;
      }
      gl_FragColor = vec4(acc / wsum, base.a);
    }`,
};

export class FilmEngine {
  constructor(host, { width = 1920, height = 1080, pixelRatio = 1.5 } = {}) {
    this.host = host;
    this.w = width; this.h = height;
    this.pickables = [];
    this.insetR = 0; this.insetB = 0;
    const r = (this.renderer = new THREE.WebGLRenderer({ antialias: false, powerPreference: 'high-performance', preserveDrawingBuffer: true }));
    r.setPixelRatio(pixelRatio);
    r.setSize(width, height, false);
    r.toneMapping = THREE.ACESFilmicToneMapping;
    r.toneMappingExposure = 0.84;
    host.append(r.domElement);

    const scene = (this.scene = new THREE.Scene());
    scene.background = new THREE.Color(BG);
    scene.fog = new THREE.Fog(BG, 60, 200);
    this.camera = new THREE.PerspectiveCamera(34, width / height, 0.004, 400);

    const pmrem = new THREE.PMREMGenerator(r);
    scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    scene.environmentIntensity = 0.26;
    scene.add(new THREE.HemisphereLight(0x9fb8ff, 0x0a0f1c, 0.45));
    const key = new THREE.DirectionalLight(0xffffff, 0.95);
    key.position.set(8, 14, 12);
    scene.add(key);
    const rim = new THREE.DirectionalLight(0x5ef0d4, 0.55);
    rim.position.set(-10, 6, -12);
    scene.add(rim);

    // 合成：多重采样的渲染目标（后期处理链里没有默认抗锯齿）+ 深度纹理（景深用）
    const pw = Math.round(width * pixelRatio), ph = Math.round(height * pixelRatio);
    const rt = new THREE.WebGLRenderTarget(pw, ph, { type: THREE.HalfFloatType, samples: 4 });
    rt.depthTexture = new THREE.DepthTexture(pw, ph);
    rt.depthTexture.type = THREE.UnsignedIntType;
    this.composer = new EffectComposer(r, rt);
    this.composer.setPixelRatio(1);
    this.composer.setSize(pw, ph);
    this.renderPass = new RenderPass(scene, this.camera);
    this.composer.addPass(this.renderPass);
    this.dof = new ShaderPass(DofShader);
    this.dof.uniforms.res.value.set(pw, ph);
    this.dof.enabled = false;
    this.composer.addPass(this.dof);
    this.bloom = new UnrealBloomPass(new THREE.Vector2(pw / 2, ph / 2), 0.36, 0.42, 0.55);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());
    this.rt = rt;

    this.css = new CSS2DRenderer();
    this.css.domElement.className = 'css2d';
    this.css.setSize(width, height);
    host.append(this.css.domElement);

    // 字幕区留白：投影中心往上挪一点，让主体落在字幕上方；算式板出现时再往左挪（shiftX）
    this.shiftY = 0;
    this.shiftX = 0;
    this.applyOffset();
  }

  applyOffset() {
    const { w, h } = this;
    this.camera.aspect = w / h;
    if (this.shiftY || this.shiftX) this.camera.setViewOffset(w, h, this.shiftX, this.shiftY, w, h);
    else this.camera.clearViewOffset();
    this.camera.updateProjectionMatrix();
  }

  // 网站的算式板会告诉 Engine 它挡住了哪块画面；片子里的机位是手写的，这里只记下来
  setOverlay(top = 0, bottom = 0) { this.ov = [top, bottom]; }

  // 让宽 width、高 height 的区域刚好装进画面所需的距离（和站点 Engine 同名同义，给机器的取景用）
  fitDistance(width, height, margin = 1.15) {
    const vf = (this.camera.fov * Math.PI) / 180;
    const tv = Math.tan(vf / 2) * 0.86; // 底部留出字幕区
    const th = Math.tan(vf / 2) * (this.w / this.h);
    return Math.max((height * margin) / 2 / tv, (width * margin) / 2 / th);
  }

  setCamera(pos, look, fov = 34, roll = 0) {
    const c = this.camera;
    if (Math.abs(c.fov - fov) > 1e-6) { c.fov = fov; c.updateProjectionMatrix(); }
    c.position.copy(pos);
    c.up.set(Math.sin(roll), Math.cos(roll), 0);
    c.lookAt(look);
  }

  setDof(focus, range, maxBlur) {
    const on = maxBlur > 0.05;
    this.dof.enabled = on;
    if (!on) return;
    const u = this.dof.uniforms;
    u.focus.value = focus; u.range.value = range; u.maxBlur.value = maxBlur;
    u.cameraNear.value = this.camera.near; u.cameraFar.value = this.camera.far;
    u.tDepth.value = this.rt.depthTexture;
  }

  render() {
    this.composer.render();
    this.css.render(this.scene, this.camera);
  }
}
