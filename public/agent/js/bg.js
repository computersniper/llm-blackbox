// 背景：纯色底 + 缓慢上浮的暗淡粒子（从 ../../js/bg.js 简化而来：去掉了中心的径向渐变和声呐圈，避免“背景光源”）。

const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

export class Background {
  constructor(canvas) {
    this.c = canvas;
    this.g = canvas.getContext('2d');
    this.parts = [];
    this.resize();
    addEventListener('resize', () => this.resize());
    this.last = performance.now();
    requestAnimationFrame((t) => this.frame(t));
  }

  resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    this.w = innerWidth; this.h = innerHeight;
    this.c.width = this.w * dpr; this.c.height = this.h * dpr;
    this.c.style.width = this.w + 'px'; this.c.style.height = this.h + 'px';
    this.g.setTransform(dpr, 0, 0, dpr, 0, 0);
    const n = Math.round(Math.min(120, (this.w * this.h) / 12000));
    while (this.parts.length < n) this.parts.push(this.spawn(true));
    this.parts.length = n;
  }

  spawn(anywhere) {
    const z = 0.2 + Math.random() * 0.8;
    return { x: Math.random() * this.w, y: anywhere ? Math.random() * this.h : this.h + 10, z, vy: -(3 + 10 * z), sway: Math.random() * Math.PI * 2, r: 0.5 + z * 1.1 };
  }

  frame(now) {
    const dt = Math.min(0.05, (now - this.last) / 1000);
    this.last = now;
    const g = this.g;
    g.fillStyle = '#060d1a';
    g.fillRect(0, 0, this.w, this.h);
    for (const p of this.parts) {
      if (!reduced) {
        p.sway += dt * (0.3 + p.z * 0.4);
        p.x += Math.sin(p.sway) * 5 * dt * p.z;
        p.y += p.vy * dt;
        if (p.y < -12) Object.assign(p, this.spawn(false));
      }
      g.fillStyle = `rgba(120, 200, 220, ${0.06 + p.z * 0.22})`;
      g.beginPath();
      g.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      g.fill();
    }
    requestAnimationFrame((t) => this.frame(t));
  }
}
