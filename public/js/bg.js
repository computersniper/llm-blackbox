// 背景：缓慢上浮的"海雪"粒子。下潜时粒子沿径向拉成光线（跃迁），上浮时反向收拢。

const TINTS = [
  [6, 13, 26], [6, 13, 25], [6, 12, 24], [7, 11, 23], [8, 11, 22],
  [9, 10, 21], [10, 9, 20], [11, 8, 17], [6, 14, 23],
];
const DOTS = [
  [120, 230, 220], [120, 220, 235], [130, 200, 255], [140, 180, 255], [160, 170, 255],
  [190, 160, 255], [225, 170, 240], [255, 190, 150], [130, 240, 220],
];

const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

export class Background {
  constructor(canvas) {
    this.c = canvas;
    this.g = canvas.getContext('2d');
    this.depth = 0;
    this.tint = [...TINTS[0]];
    this.dot = [...DOTS[0]];
    this.warpStart = -1e9;
    this.warpDir = 1;
    this.cx = 0; this.cy = 0;
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
    const n = Math.round(Math.min(190, (this.w * this.h) / 8000));
    while (this.parts.length < n) this.parts.push(this.spawn(true));
    this.parts.length = n;
    this.cx = this.w / 2; this.cy = this.h / 2;
  }

  spawn(anywhere) {
    const z = 0.2 + Math.random() * 0.8;
    return {
      x: Math.random() * this.w,
      y: anywhere ? Math.random() * this.h : this.h + 10,
      z,
      vy: -(4 + 14 * z),
      sway: Math.random() * Math.PI * 2,
      r: 0.5 + z * 1.4,
      px: 0, py: 0,
    };
  }

  setDepth(d) { this.depth = d; }

  warp(dir, x, y) {
    if (reduced) return;
    this.warpStart = performance.now();
    this.warpDir = dir;
    this.cx = x ?? this.w / 2;
    this.cy = y ?? this.h / 2;
  }

  frame(now) {
    const dt = Math.min(0.05, (now - this.last) / 1000);
    this.last = now;
    const g = this.g;
    const tt = TINTS[this.depth] ?? TINTS[0];
    const dd = DOTS[this.depth] ?? DOTS[0];
    for (let i = 0; i < 3; i++) {
      this.tint[i] += (tt[i] - this.tint[i]) * Math.min(1, dt * 1.5);
      this.dot[i] += (dd[i] - this.dot[i]) * Math.min(1, dt * 1.5);
    }
    const [r, gg, b] = this.tint.map(Math.round);
    // 均匀的深色空间：中间不提亮（提亮会像背后有一盏灯），只在最边缘略微压暗
    const grad = g.createRadialGradient(this.w * 0.5, this.h * 0.5, 0, this.w * 0.5, this.h * 0.5, Math.max(this.w, this.h) * 0.75);
    grad.addColorStop(0, `rgb(${r},${gg},${b})`);
    grad.addColorStop(0.6, `rgb(${r},${gg},${b})`);
    grad.addColorStop(1, `rgb(${Math.max(0, r - 2)},${Math.max(0, gg - 4)},${Math.max(0, b - 7)})`);
    g.fillStyle = grad;
    g.fillRect(0, 0, this.w, this.h);

    // 跃迁强度：先急速上升，再缓慢衰减
    const wt = (now - this.warpStart) / 1300;
    const k = wt >= 0 && wt < 1 ? Math.sin(Math.PI * Math.pow(wt, 0.6)) : 0;
    const [dr, dg, db] = this.dot.map(Math.round);

    for (const p of this.parts) {
      p.px = p.x; p.py = p.y;
      if (k > 0.02) {
        const dx = p.x - this.cx, dy = p.y - this.cy;
        const f = this.warpDir * k * 5.5 * (0.4 + p.z);
        p.x += dx * f * dt;
        p.y += dy * f * dt;
        if (this.warpDir > 0 && (p.x < -20 || p.x > this.w + 20 || p.y < -20 || p.y > this.h + 20)) {
          const a = Math.random() * Math.PI * 2, rr = 10 + Math.random() * 60;
          p.x = p.px = this.cx + Math.cos(a) * rr;
          p.y = p.py = this.cy + Math.sin(a) * rr;
        }
        if (this.warpDir < 0 && Math.hypot(dx, dy) < 30) {
          const a = Math.random() * Math.PI * 2, rr = Math.max(this.w, this.h) * 0.7;
          p.x = p.px = this.cx + Math.cos(a) * rr;
          p.y = p.py = this.cy + Math.sin(a) * rr;
        }
      } else if (!reduced) {
        p.sway += dt * (0.3 + p.z * 0.4);
        p.x += Math.sin(p.sway) * 6 * dt * p.z;
        p.y += p.vy * dt;
        if (p.y < -12) Object.assign(p, this.spawn(false));
      }
      const alpha = 0.12 + p.z * 0.45;
      if (k > 0.05) {
        g.strokeStyle = `rgba(${dr},${dg},${db},${alpha * (0.6 + k)})`;
        g.lineWidth = p.r;
        g.beginPath();
        g.moveTo(p.px, p.py);
        const sx = p.x + (p.x - p.px) * 3 * k, sy = p.y + (p.y - p.py) * 3 * k;
        g.lineTo(sx, sy);
        g.stroke();
      } else {
        g.fillStyle = `rgba(${dr},${dg},${db},${alpha})`;
        g.beginPath();
        g.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        g.fill();
      }
    }

    // 一圈极淡的声呐波
    const sonar = (now / 7000) % 1;
    if (!reduced && sonar < 0.6) {
      const rad = sonar / 0.6 * Math.max(this.w, this.h) * 0.6;
      g.strokeStyle = `rgba(${dr},${dg},${db},${0.07 * (1 - sonar / 0.6)})`;
      g.lineWidth = 1;
      g.beginPath();
      g.arc(this.w / 2, this.h * 0.52, rad, 0, Math.PI * 2);
      g.stroke();
    }
    requestAnimationFrame((t) => this.frame(t));
  }
}
