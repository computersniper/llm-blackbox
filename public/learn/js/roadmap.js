// 学习路线图：6 段、20 站，像地铁线一样蛇形排开。
// 宽屏时每一段是一行，奇数段反向排，线在行尾绕一个圆角弯接到下一行；窄屏时一列排下来，线走在左边。
// 线随着页面往下滚一段段“画”出来，画到哪一站，那一站的卡片才浮现；学完的相邻两站之间的线段变成青色。

import { $, esc, reducedMotion } from '../../js/ui.js';
import { loadDone, saveDone, onDoneChange } from './store.js';
import { isEn, L, t, tx } from './lang.js';

const NS = 'http://www.w3.org/2000/svg';
const CHECK = '<svg viewBox="0 0 12 12" aria-hidden="true"><path d="M2.4 6.3 5 8.8 9.7 3.4" fill="none" stroke="#04121a" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/></svg>';
const pad2 = (n) => String(n).padStart(2, '0');
const stop = (no) => L(`第 ${pad2(no)} 站`, `Stop ${pad2(no)}`);

export class Roadmap {
  constructor(root, RM, RES, { onProgress } = {}) {
    this.root = root;
    this.svg = $('#rmSvg', root);
    this.res = new Map(RES.items.map((x) => [x.id, x]));
    this.platforms = RES.platforms_en || {};
    this.stages = RM.stages;
    this.nodes = [];
    this.stages.forEach((s, si) => s.nodes.forEach((n) => this.nodes.push({ ...n, si, no: this.nodes.length + 1 })));
    this.done = loadDone();
    this.onProgress = onProgress || (() => {});
    this.drawn = 0;
    this.total = 1;
    this.visible = false;
    this.anim = !reducedMotion;
    this.render();
    this.bind();
    this.layout();
    this.refresh(false);
    onDoneChange((set) => { this.done = set; this.refresh(false); });
    new ResizeObserver(() => this.queueLayout()).observe(root);
    document.fonts?.ready.then(() => this.queueLayout());
    new IntersectionObserver((es) => { this.visible = es[0].isIntersecting; if (this.visible) this.kick(); }, { rootMargin: '200px 0px' }).observe(root);
    addEventListener('scroll', () => this.kick(), { passive: true });
  }

  /* ------------------------------------------------ 结构 */

  render() {
    const last = this.stages.length - 1;
    const html = this.stages.map((s, si) => {
      const nodes = this.nodes.filter((n) => n.si === si);
      const end = si === last
        ? L(`<div class="rm-end" id="rmEnd"><span class="rm-dot" aria-hidden="true"></span><b>终点</b><br>走到这里，你已经可以自己读论文、跑实验了。<br><a href="#try">去「动手试试」→</a></div>`,
          `<div class="rm-end" id="rmEnd"><span class="rm-dot" aria-hidden="true"></span><b>The end</b><br>By now you can read papers and run experiments on your own.<br><a href="#try">Go to Hands-on →</a></div>`)
        : '';
      return `<section class="rm-stage${si % 2 ? ' rev' : ''}" id="st-${esc(s.id)}" aria-label="${L(`第 ${s.no} 段 ${esc(s.title)}`, `Stage ${s.no}: ${esc(tx(s, 'title'))}`)}">
        <div class="rm-stage-h"><span class="rm-no">${esc(s.no)}</span><h3>${esc(tx(s, 'title'))}</h3><span class="sub">${esc(tx(s, 'sub'))}</span><span class="cnt" data-cnt="${si}"></span></div>
        <div class="rm-row">${nodes.map((n) => this.nodeHTML(n)).join('')}${end}</div>
      </section>`;
    }).join('');
    $('.lp-loading', this.root)?.remove();
    this.root.insertAdjacentHTML('beforeend', html);
    this.els = this.nodes.map((n) => $(`#n-${n.id}`, this.root));
    this.endEl = $('#rmEnd', this.root);
    if (this.anim) this.root.classList.add('anim');
    // 首屏的深度计
    this.gauge = $('#heroGauge');
    if (this.gauge) {
      this.gauge.innerHTML = `<div class="gauge-h"><span>${L('DEPTH · 路线', 'DEPTH · ROUTE')}</span><span>${L('学完', 'Done')}</span></div><ol>${this.stages.map((s, si) => `<li style="--i:${si}" data-g="${si}"><a href="#st-${esc(s.id)}"><span class="g-dot"></span><span class="g-no">${esc(s.no)}</span><span class="g-t">${esc(tx(s, 'title'))}</span><span class="g-cells">${s.nodes.map(() => '<i></i>').join('')}</span></a></li>`).join('')}</ol><div class="gauge-f">${L('从上往下，一段比一段深。进度只存在这台设备的浏览器里。', 'Top to bottom, each stage goes deeper. Progress is saved only in this browser.')}</div>`;
    }
  }

  nodeHTML(n) {
    const site = n.site.map((x) => `<a class="site-chip" href="${esc(x.href)}">${esc(tx(x, 'label'))}</a>`).join('');
    const pf = (r) => (isEn ? `${esc(this.platforms[r.platform] || r.platform)}·${r.lang === 'zh' ? 'ZH' : 'EN'}` : `${esc(r.platform)}·${r.lang === 'zh' ? '中' : 'EN'}`);
    const res = this.resOf(n).map((id) => this.res.get(id)).filter(Boolean).map((r) => `<li><a href="${esc(r.url)}" target="_blank" rel="noopener" title="${esc(tx(r, 'title'))}"><span class="pf${r.lang === 'zh' ? ' zh' : ''}">${pf(r)}</span><span class="t">${esc(tx(r, 'title'))}</span></a></li>`).join('');
    return `<article class="rm-node" id="n-${esc(n.id)}" data-id="${esc(n.id)}">
      <span class="rm-dot" aria-hidden="true">${CHECK}</span>
      <div class="rm-k">${stop(n.no)}</div>
      <h4>${esc(tx(n, 'title'))}</h4>
      <p class="rm-goal">${esc(tx(n, 'goal'))}</p>
      <div class="rm-site"><span class="rm-lab">${L('在本站看', 'See it on this site')}</span>${site}</div>
      <span class="rm-lab">${L('延伸', 'Go further')}</span>
      <ul class="rm-res">${res}</ul>
      <div class="rm-foot"><button class="rm-done" type="button" aria-pressed="false" aria-label="${L(`第 ${pad2(n.no)} 站 ${esc(n.title)}：标记为学完`, `Stop ${pad2(n.no)} ${esc(tx(n, 'title'))}: mark as done`)}"><span class="box">${CHECK}</span><span class="rm-lbl">${L('标记学完', 'Mark as done')}</span></button></div>
    </article>`;
  }

  bind() {
    this.root.addEventListener('click', (e) => {
      const btn = e.target.closest('.rm-done, .rm-node .rm-dot');
      if (!btn) return;
      const node = btn.closest('.rm-node');
      this.toggle(node.dataset.id);
    });
    const reset = $('#rmReset');
    let armed = 0;
    reset?.addEventListener('click', () => {
      if (!this.done.size) return;
      if (Date.now() - armed > 3000) {
        armed = Date.now();
        reset.textContent = t('lp.resetConfirm');
        setTimeout(() => { if (Date.now() - armed >= 2900) reset.textContent = t('lp.reset'); }, 3000);
        return;
      }
      armed = 0;
      reset.textContent = t('lp.reset');
      this.done.clear();
      saveDone(this.done);
      this.refresh(true);
    });
  }

  toggle(id) {
    const on = !this.done.has(id);
    if (on) this.done.add(id); else this.done.delete(id);
    saveDone(this.done);
    const el = $(`#n-${id}`, this.root);
    if (on && el && this.anim) {
      el.classList.remove('pop');
      void el.offsetWidth;
      el.classList.add('pop');
      setTimeout(() => el.classList.remove('pop'), 900);
    }
    this.refresh(true);
  }

  refresh() {
    this.nodes.forEach((n, i) => {
      const on = this.done.has(n.id);
      const el = this.els[i];
      el.classList.toggle('done', on);
      const b = $('.rm-done', el);
      b.setAttribute('aria-pressed', String(on));
      $('.rm-lbl', b).textContent = on ? L('已学完', 'Done') : L('标记学完', 'Mark as done');
      b.setAttribute('aria-label', L(`第 ${pad2(n.no)} 站 ${n.title}：${on ? '已学完，点一下取消' : '标记为学完'}`, `Stop ${pad2(n.no)} ${tx(n, 'title')}: ${on ? 'done — click to undo' : 'mark as done'}`));
    });
    this.stages.forEach((s, si) => {
      const ns = this.nodes.filter((n) => n.si === si);
      const k = ns.filter((n) => this.done.has(n.id)).length;
      const c = $(`[data-cnt="${si}"]`, this.root);
      c.textContent = `${k} / ${ns.length}`;
      c.classList.toggle('full', k === ns.length);
      const g = this.gauge && $(`[data-g="${si}"]`, this.gauge);
      if (g) {
        g.classList.toggle('full', k === ns.length);
        [...g.querySelectorAll('.g-cells i')].forEach((cell, j) => cell.classList.toggle('on', this.done.has(ns[j].id)));
        $('a', g).setAttribute('aria-label', L(`第 ${s.no} 段 ${s.title}，学完 ${k} / ${ns.length} 站`, `Stage ${s.no} ${tx(s, 'title')}: ${k} / ${ns.length} stops done`));
      }
    });
    this.paintSegs();
    const k = this.nodes.filter((n) => this.done.has(n.id)).length;
    this.onProgress(k, this.nodes.length);
  }

  /* ------------------------------------------------ 线 */

  queueLayout() {
    if (this.lq) return;
    this.lq = requestAnimationFrame(() => { this.lq = 0; this.layout(); });
  }

  // 相对路线容器的坐标（用 offset* 量，不受卡片浮现动画的位移影响）
  center(el) {
    let x = el.offsetWidth / 2, y = el.offsetHeight / 2;
    for (let e = el; e && e !== this.root; e = e.offsetParent) { x += e.offsetLeft; y += e.offsetTop; }
    return [x, y];
  }

  layout() {
    const W = this.root.clientWidth, H = this.root.clientHeight;
    if (!W) return;
    const dots = [...this.els.map((el) => $('.rm-dot', el)), $('.rm-dot', this.endEl)];
    const P = dots.map((d) => this.center(d));
    const column = Math.abs(P[0][0] - P[1][0]) < 3; // 窄屏：一列
    const frac = this.total > 1 ? this.drawn / this.total : 0;
    // 起点：宽屏从左边缘引进来，窄屏从第一站上方
    const [x0, y0] = P[0];
    const start = column ? [x0, Math.max(0, y0 - 46)] : [-22, y0];
    const segs = [`M ${start[0]} ${start[1]} L ${x0} ${y0}`];
    for (let i = 1; i < P.length; i++) {
      const [px, py] = P[i - 1], [qx, qy] = P[i];
      let s = `M ${px} ${py} `;
      if (Math.abs(qy - py) < 3 || Math.abs(qx - px) < 3) s += `L ${qx} ${qy}`;
      else {
        const right = px > W / 2;
        const xe = right ? W + 22 : -22;
        const r = Math.min(34, (qy - py) / 2);
        const sw = right ? 1 : 0, dx = right ? -r : r;
        s += `L ${xe + dx} ${py} A ${r} ${r} 0 0 ${sw} ${xe} ${py + r} L ${xe} ${qy - r} A ${r} ${r} 0 0 ${sw} ${xe + dx} ${qy} L ${qx} ${qy}`;
      }
      segs.push(s);
    }
    this.svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    this.svg.innerHTML = '';
    const mk = (cls, d) => { const p = document.createElementNS(NS, 'path'); p.setAttribute('class', cls); p.setAttribute('d', d); this.svg.appendChild(p); return p; };
    this.track = mk('rm-track', segs.map((s, i) => (i ? s.replace(/^M [^L]+/, '') : s)).join(' '));
    this.segEls = segs.map((s) => mk('rm-seg', s));
    this.segLen = this.segEls.map((p) => p.getTotalLength());
    this.cum = [];
    let acc = 0;
    this.segLen.forEach((l) => { acc += l; this.cum.push(acc); }); // cum[i]：画到第 i 个点（0 起，最后一个是终点）
    this.total = acc;
    this.trackLen = this.track.getTotalLength();
    this.track.style.strokeDasharray = `${this.trackLen} ${this.trackLen + 10}`;
    this.segEls.forEach((p, i) => { p.style.strokeDasharray = `${this.segLen[i]} ${this.segLen[i] + 10}`; });
    const st = document.createElementNS(NS, 'circle');
    st.setAttribute('class', 'rm-start'); st.setAttribute('cx', start[0]); st.setAttribute('cy', start[1]); st.setAttribute('r', 4);
    this.svg.appendChild(st);
    this.trav = document.createElementNS(NS, 'circle');
    this.trav.setAttribute('class', 'rm-trav'); this.trav.setAttribute('r', 2.6); this.trav.style.opacity = 0;
    this.svg.appendChild(this.trav);
    this.P = P;
    this.drawn = this.anim ? frac * this.total : this.total;
    this.paintSegs(true);
    this.paintTrack();
    this.kick();
  }

  // 学完的相邻两站之间的线段亮起来（起点 → 第 1 站、第 20 站 → 终点 也算）
  paintSegs(instant = false) {
    if (!this.segEls) return;
    const ids = this.nodes.map((n) => n.id);
    const d = (k) => this.done.has(ids[k]);
    const N = ids.length;
    this.segEls.forEach((p, i) => {
      const on = i === 0 ? d(0) : i === N ? d(N - 1) : d(i - 1) && d(i);
      if (instant) p.style.transition = 'none';
      p.style.strokeDashoffset = on ? '0' : String(this.segLen[i] + 10);
      if (instant) { void p.getBoundingClientRect(); p.style.transition = ''; }
    });
  }

  paintTrack() {
    const off = Math.max(0, this.trackLen - this.drawn * (this.trackLen / this.total));
    this.track.style.strokeDashoffset = String(off);
    this.els.forEach((el, i) => { if (this.drawn >= this.cum[i] - 4) el.classList.add('lit'); });
    if (this.drawn >= this.total - 4) this.endEl.classList.add('lit');
  }

  /* ------------------------------------------------ 动画：线跟着滚动画出来，一个小光点沿着已画好的线走 */

  target() {
    const top = this.root.getBoundingClientRect().top;
    const line = innerHeight * 0.86 - top;
    let k = -1;
    this.P.forEach(([, y], i) => { if (y <= line) k = i; });
    return k < 0 ? 0 : this.cum[k];
  }

  kick() {
    if (this.raf || !this.P) return;
    this.last = performance.now();
    this.raf = requestAnimationFrame((t) => this.frame(t));
  }

  frame(now) {
    this.raf = 0;
    const dt = Math.min(0.05, (now - this.last) / 1000);
    this.last = now;
    let busy = false;
    if (this.anim) {
      const tgt = this.target();
      if (tgt > this.drawn + 0.5) {
        const step = Math.max((tgt - this.drawn) * (1 - Math.exp(-dt * 2.6)), 420 * dt);
        this.drawn = Math.min(tgt, this.drawn + step);
        this.paintTrack();
        busy = true;
      }
    }
    // 小光点：从起点沿着已经画出来的线走到头，再从头开始
    if (this.anim && this.visible && this.drawn > 40) {
      this.tp = ((this.tp || 0) + dt * 110) % (this.drawn + 160);
      const s = Math.min(this.tp, this.drawn);
      const pt = this.track.getPointAtLength(s * (this.trackLen / this.total));
      this.trav.setAttribute('cx', pt.x);
      this.trav.setAttribute('cy', pt.y);
      const fade = Math.min(1, s / 60, (this.drawn - s) / 60);
      this.trav.style.opacity = String(0.85 * Math.max(0, fade));
      busy = true;
    }
    if (busy) this.kick();
  }

  // 从资源库 / 术语表跳回某一站：滚过去，闪一下边框
  focus(id) {
    const el = $(`#n-${id}`, this.root);
    if (!el) return;
    el.classList.add('lit');
    el.scrollIntoView({ behavior: reducedMotion ? 'auto' : 'smooth', block: 'center' });
    el.classList.remove('flash');
    void el.offsetWidth;
    el.classList.add('flash');
  }

  stationOf(resId) {
    return this.nodes.find((n) => this.resOf(n).includes(resId));
  }

  // 英文模式下每一站挂的是 res_en（全部英文资源）
  resOf(n) {
    return (isEn && n.res_en) || n.res;
  }
}
