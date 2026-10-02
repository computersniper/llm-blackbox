// D0 · 流水线：预训练 → 监督微调 → 偏好对齐 / 强化学习。前两段是本页的真实训练，第三段只是示意（DPO 的数是真实算的）。
import { COL, text, rr, card, line, dot, hexA, wrap, badge, fmtInt, arrow, measure, clamp, ease } from '../draw.js';
import { isEn, L, compact, SFT_EN } from '../lang.js';

const disp = (s) => (s === '\n' ? '↵' : s === '\n\n' ? '↵↵' : String(s).replace(/\n/g, '↵'));
const ROLE_COL = { user: '#5ee4f0', answer: COL.amber, tpl: COL.violet };

export class Pipeline {
  constructor(app) { this.app = app; this.t0 = 0; }

  layout(env) {
    this.portrait = env.portrait;
    if (!this.portrait) {
      const w = 344, gap = 36;
      this.cards = [0, 1, 2].map((i) => ({ x: i * (w + gap), y: 96, w, h: 492 }));
      this.title = { x: 0, y: 0 };
      this.W = 3 * w + 2 * gap;
    } else {
      this.cards = [0, 1, 2].map((i) => ({ x: 0, y: 120 + i * 572, w: 380, h: 548 }));
      this.W = 380;
    }
  }

  focus(st) {
    if (!this.portrait) return { x: -16, y: -6, w: this.W + 32, h: 604 };
    const c = this.cards[st.step.st];
    const y0 = st.step.st === 0 ? -4 : c.y - 10;
    return { x: -8, y: y0, w: 396, h: c.y + c.h + 10 - y0 };
  }

  draw(g, st, env) {
    const cur = st.step.st;
    const P = this.portrait;
    text(g, L('TRAINING PIPELINE · 一个大模型的三段训练', 'TRAINING PIPELINE · THREE STAGES OF TRAINING AN LLM'), 0, P ? 18 : 22, { size: 10.5, kind: 'mono', color: COL.dim });
    text(g, L('一个大模型是怎么训练出来的', 'How an LLM gets trained'), 0, P ? 54 : 64, { size: P ? 25 : 32, kind: 'serif', weight: 900, color: COL.ink });
    if (isEn) {
      if (!P) wrap(g, 'One goal — guess the next token — with three kinds of data and three algorithms, one stage after another. The first two stages are real training runs recorded for this page.', this.W, 46, 600, 18, { size: 13, color: COL.ink2, align: 'right' });
      else wrap(g, 'One goal — guess the next token — with three kinds of data and three algorithms, one stage after another.', 0, 84, 380, 17, { size: 12, color: COL.ink2, maxLines: 2 });
    } else if (!P) text(g, '同一个“猜下一个词元”的目标，换三种数据、三种算法，一段接一段地训。前两段是这一页上的真实训练记录。', this.W, 64, { size: 13, color: COL.ink2, align: 'right' });
    else wrap(g, '同一个“猜下一个词元”的目标，换三种数据、三种算法，一段接一段地训。', 0, 84, 380, 18, { size: 12.5, color: COL.ink2 });
    for (let i = 0; i < 3; i++) {
      const c = this.cards[i];
      g.save();
      g.globalAlpha = i === cur ? 1 : 0.62;
      [this.pre, this.sft, this.pref][i].call(this, g, c, i === cur, env, st);
      g.restore();
      if (i < 2) {
        if (!P) arrow(g, c.x + c.w + 8, c.y + c.h / 2, c.x + c.w + 34, c.y + c.h / 2, i < cur ? COL.cyan : COL.line3, 1.5, 7);
        else arrow(g, c.x + c.w / 2, c.y + c.h + 4, c.x + c.w / 2, c.y + c.h + 20, i < cur ? COL.cyan : COL.line3, 1.5, 7);
      }
      env.hit(c.x, c.y, c.w, c.h - 60, { click: true, act: () => { const tl = this.app.tl; tl.pause(); tl.seekIndex(i); this.app.sfx('click'); }, tip: L(`<span class="k">第 ${i + 1} 段</span>点一下选中，再按 ＋ 进去`, `<span class="k">Stage ${i + 1}</span>Click to select it, then press + to go in`) });
    }
  }

  button(g, c, label, on, env, act, color = COL.cyan) {
    const bw = c.w - 28, by = c.y + c.h - 52;
    rr(g, c.x + 14, by, bw, 38, 12);
    g.fillStyle = on ? hexA(color, 0.16) : 'rgba(255,255,255,0.03)';
    g.fill();
    g.strokeStyle = on ? hexA(color, 0.7) : COL.line2;
    g.lineWidth = 1.2;
    g.stroke();
    text(g, label, c.x + c.w / 2, by + 24, { size: 13.5, color: on ? color : COL.ink2, align: 'center', weight: 600 });
    if (act) env.hit(c.x + 14, by, bw, 38, { click: true, act, tip: L('进入这一段真实训练', 'Enter this real training run') });
  }

  facts(g, x, y, w, rows) {
    rr(g, x, y, w, rows.length * 21 + 32, 10);
    g.fillStyle = 'rgba(94,240,212,0.04)';
    g.fill();
    g.strokeStyle = 'rgba(94,240,212,0.18)';
    g.stroke();
    text(g, L('本页实测', 'MEASURED HERE'), x + 12, y + 18, { size: 10, kind: 'mono', color: COL.cyan });
    rows.forEach(([k, v], i) => {
      text(g, k, x + 12, y + 40 + i * 21, { size: 11.5, color: COL.dim });
      text(g, v, x + w - 12, y + 40 + i * 21, { size: 11.5, color: COL.ink, align: 'right', kind: /^[\d.→ ×e+-]+$/.test(v) ? 'mono' : 'sans', max: w - 80 });
    });
    return rows.length * 21 + 32;
  }

  spark(g, x, y, w, h, vals, color) {
    const lo = Math.min(...vals), hi = Math.max(...vals);
    const pts = vals.map((v, i) => [x + (i / (vals.length - 1)) * w, y + h - ((v - lo) / (hi - lo || 1)) * h]);
    line(g, pts, color, 1.6);
    dot(g, ...pts[pts.length - 1], 2.5, color);
  }

  pre(g, c, on, env) {
    const tm = this.app.tiny.D.meta;
    card(g, c.x, c.y, c.w, c.h, { eyebrow: 'STAGE 1 · PRETRAINING', title: L('预训练 · 学会语言', 'Pretraining · learning language'), accent: COL.cyan, active: on });
    wrap(g, L('把海量文本切成词元，一遍遍“猜下一个”。每个位置都算损失：语言、知识、文风，都是这一段学到的。', 'Chop vast amounts of text into tokens and “guess the next one” over and over. Every position counts: language, knowledge and style are all learned here.'), c.x + 14, c.y + 70, c.w - 28, 19, { size: isEn ? 12 : 12.5, color: COL.ink2, maxLines: isEn ? 3 : 0 });
    // 一行真实的训练数据：每个字下面都有一条线（都算损失）
    const D = this.app.tiny.D, k = D.K - 1;
    let x = c.x + 14;
    const y = c.y + 130;
    for (let i = 0; i < 20; i++) {
      const id = D.rowId(k, i);
      const ch = id === 0 ? '⏎' : D.ch(id);
      const w = 15;
      text(g, ch, x + w / 2, y, { size: 13, color: id === 0 ? COL.violet : COL.ink, align: 'center' });
      g.fillStyle = COL.cyan;
      g.fillRect(x + 1, y + 6, w - 2, 2);
      x += w + 0.5;
    }
    text(g, L('↑ 每个字都要猜，都算损失', '↑ every character is a guess, and every guess counts'), c.x + 14, y + 26, { size: 10.5, color: COL.cyan });
    const h = this.facts(g, c.x + 14, c.y + 178, c.w - 28, isEn ? [
      ['Corpus', `${compact(tm.corpus.poems.train, 0)} Tang & Song poems`],
      ['', `${compact(tm.corpus.chars.train, 1)} chars · chinese-poetry (MIT)`],
      ['Model', `Qwen3 arch · ${tm.model.layers} layers · ${compact(tm.model.params)} params`],
      ['Training', `${fmtInt(tm.train.steps)} steps · ${tm.train.seconds} s · RTX 5090`],
      ['Val loss', `${tm.ckpts[0].valLoss.toFixed(2)} → ${tm.train.finalVal.toFixed(2)}`],
    ] : [
      ['语料', `唐宋格律诗 ${(tm.corpus.poems.train / 1e4).toFixed(1)} 万首`],
      ['', `${(tm.corpus.chars.train / 1e4).toFixed(0)} 万字 · chinese-poetry（MIT）`],
      ['模型', `Qwen3 同构 · ${tm.model.layers} 层 · ${(tm.model.params / 1e4).toFixed(0)} 万参数`],
      ['训练', `${fmtInt(tm.train.steps)} 步 · ${tm.train.seconds} 秒 · RTX 5090`],
      ['验证损失', `${tm.ckpts[0].valLoss.toFixed(2)} → ${tm.train.finalVal.toFixed(2)}`],
    ]);
    text(g, L('对照（据 Qwen3 技术报告）：约 36 万亿词元，119 种语言', 'Qwen3 itself: ~36 trillion tokens, 119 languages'), c.x + 14, c.y + 178 + h + 22, { size: 10.5, color: COL.dim, max: c.w - 28 });
    if (isEn) text(g, 'Trained on Chinese poems, so it writes Chinese.', c.x + 14, c.y + 178 + h + 40, { size: 10.5, color: COL.dim, max: c.w - 28 });
    this.button(g, c, L('＋ 进去，看它从乱码学会写诗', '+ Go in: from gibberish to poems'), on, env, () => this.app.enterRun('tiny'));
  }

  sft(g, c, on, env) {
    const q = this.app.qwen.D, qm = q.meta;
    card(g, c.x, c.y, c.w, c.h, { eyebrow: 'STAGE 2 · SUPERVISED FINE-TUNING', title: L('监督微调 · 学会对话', 'Fine-tuning · learning to chat'), accent: COL.amber, active: on });
    wrap(g, L('用一问一答的对话继续训练。损失函数不变，只是把提示部分遮掉：只对助手的回答算损失。', 'Keep training on Q&A chats. Same loss, but the prompt is masked: only the answer counts.'), c.x + 14, c.y + 70, c.w - 28, 19, { size: isEn ? 12 : 12.5, color: COL.ink2, maxLines: isEn ? 3 : 0 });
    // 聊天模板里的真实词元：只有回答下面有线
    let x = c.x + 14, y = c.y + 128;
    for (let i = 0; i < qm.ids.length; i++) {
      const s = disp(q.tok(qm.ids[i]));
      const role = qm.roles[i];
      const short = s.replace(/^<\|(.+)\|>$/, '⟨$1⟩');
      const w = measure(g, short, 10.5, role === 'tpl' ? 'mono' : 'sans') + 6;
      if (x + w > c.x + c.w - 14) { x = c.x + 14; y += 24; }
      rr(g, x, y - 13, w, 18, 4);
      g.fillStyle = hexA(ROLE_COL[role], 0.1);
      g.fill();
      text(g, short, x + 3, y, { size: 10.5, color: ROLE_COL[role], kind: role === 'tpl' ? 'mono' : 'sans' });
      if (i >= 1 && qm.sftMask[i - 1]) { g.fillStyle = COL.cyan; g.fillRect(x + 1, y + 7, w - 2, 2); }
      x += w + 3;
    }
    text(g, L('↑ 只有回答（和结束标记）下面有线', '↑ only the answer (and the end marker) is underlined'), c.x + 14, y + 24, { size: 10.5, color: COL.cyan });
    const s = qm.states.map((v) => v.lossSft);
    const h = this.facts(g, c.x + 14, y + 40, c.w - 28, isEn ? [
      ['Model', 'Qwen3-0.6B · 596M params'],
      ['Method', 'full-parameter · fp32 · AdamW lr 1e-5'],
      ['Data', `「${qm.question}」→「${qm.answer}」`],
      ['Answer loss', s.map((v, i) => v.toFixed(i === 3 ? 3 : 2)).join(' → ')],
    ] : [
      ['模型', 'Qwen3-0.6B · 5.96 亿参数'],
      ['方式', '全参数 · fp32 · AdamW lr 1e-5'],
      ['数据', `「${qm.question}」→「${qm.answer}」`],
      ['回答的损失', s.map((v, i) => v.toFixed(i === 3 ? 3 : 2)).join(' → ')],
    ]);
    if (isEn) {
      let ty = y + 40 + h + 20;
      ty += 16 * wrap(g, `Translation: “${SFT_EN.question}” → “${SFT_EN.answer}”`, c.x + 14, ty, c.w - 28, 16, { size: 10.5, color: COL.ink2, maxLines: 2 }) + 6;
      wrap(g, `Before training, the model would say: “${SFT_EN.rejected}” (translated from Chinese)`, c.x + 14, ty, c.w - 28, 16, { size: 10.5, color: COL.dim, maxLines: 3 });
    } else wrap(g, `原模型自己会说：“${qm.dpo.rejected}”`, c.x + 14, y + 40 + h + 22, c.w - 28, 16, { size: 10.5, color: COL.dim, maxLines: 2 });
    this.button(g, c, L('＋ 进去，看一步训练细到单个权重', '+ Go in: one step, down to one weight'), on, env, () => this.app.enterRun('qwen'), COL.amber);
  }

  pref(g, c, on, env) {
    const d = this.app.qwen.D.meta.dpo;
    card(g, c.x, c.y, c.w, c.h, { eyebrow: 'STAGE 3 · ALIGNMENT / RL', title: L('偏好对齐 · 学会答得更好', 'Alignment · answering better'), accent: COL.rose, active: on, demo: true });
    wrap(g, L('不再给标准答案，而是告诉它“这个回答比那个好”，或者给回答打分、用奖励来强化。', 'No reference answers: tell it “this answer beats that one”, or score answers and reinforce with rewards.'), c.x + 14, c.y + 70, c.w - 28, 19, { size: isEn ? 12 : 12.5, color: COL.ink2, maxLines: isEn ? 3 : 0 });
    let y = c.y + 124;
    text(g, L('DPO 的一个例子（对数概率是真实算的）', 'A DPO example (the log-probabilities are real)'), c.x + 14, y, { size: 11, color: COL.ink });
    y += 10;
    const rows = [
      [L('好', 'good'), d.chosen, d.policy.chosen, d.ref.chosen, COL.cyan, SFT_EN.answer],
      [L('坏', 'bad'), d.rejected, d.policy.rejected, d.ref.rejected, COL.rose, SFT_EN.rejected],
    ];
    const tx = c.x + (isEn ? 58 : 46);
    for (const [tag, s, pol, ref, col, en] of rows) {
      rr(g, c.x + 14, y, c.w - 28, 54, 8);
      g.fillStyle = hexA(col, 0.05);
      g.fill();
      g.strokeStyle = hexA(col, 0.3);
      g.stroke();
      text(g, tag, c.x + 24, y + 20, { size: isEn ? 11 : 13, color: col, weight: 700 });
      text(g, s, tx, y + 20, { size: 11.5, color: COL.ink, max: c.x + c.w - 30 - tx });
      text(g, L(`log π ${pol.toFixed(2)}　log π_ref ${ref.toFixed(2)}　提升 ${(pol - ref >= 0 ? '+' : '')}${(pol - ref).toFixed(2)}`, `log π ${pol.toFixed(2)}  log π_ref ${ref.toFixed(2)}  gain ${(pol - ref >= 0 ? '+' : '')}${(pol - ref).toFixed(2)}`), tx, y + 42, { size: 10.5, kind: 'mono', color: COL.dim, max: c.x + c.w - 30 - tx });
      if (isEn) env.hit(c.x + 14, y, c.w - 28, 54, { tip: `<span class="k">${tag} answer · translation</span>“${en}”` });
      y += 62;
    }
    text(g, L(`L = −log σ(β·[提升(好) − 提升(坏)])`, `L = −log σ(β·[gain(good) − gain(bad)])`), c.x + 14, y + 12, { size: 11.5, kind: 'mono', color: COL.ink2 });
    text(g, `= −log σ(${d.beta} × ${d.margin.toFixed(1)}) = ${d.loss.toFixed(4)}`, c.x + 14, y + 32, { size: 11.5, kind: 'mono', color: COL.amber });
    wrap(g, L('π = 走完 3 步 SFT 的模型，π_ref = 原模型。只算了这一次损失，没有真的做偏好训练。', 'π = model after 3 SFT steps, π_ref = original. Just this one loss — no real preference training.'), c.x + 14, y + 54, c.w - 28, 16, { size: 10.5, color: COL.dim });
    y += 100;
    const items = isEn ? [
      ['RLHF', 'Train a reward model, then reinforce with PPO'],
      ['GRPO', 'Score a group of answers against each other'],
      ['Distill', 'Small models mostly learn from big ones'],
    ] : [
      ['RLHF', '先训一个奖励模型给回答打分，再用 PPO 强化'],
      ['GRPO', '同一题生成一组回答，组内互相比较来打分'],
      ['蒸馏', 'Qwen3-0.6B 这样的小模型主要从大模型蒸馏'],
    ];
    for (const [k, v] of items) {
      text(g, k, c.x + 14, y, { size: 11, kind: 'mono', color: COL.rose });
      text(g, v, c.x + 64, y, { size: 11, color: COL.ink2, max: c.w - 78 });
      y += 20;
    }
    text(g, L('（后两条据 Qwen3 技术报告）', '(last two per the Qwen3 technical report)'), c.x + 14, y + 2, { size: 10, color: COL.faint });
    this.button(g, c, L('这一段没有真实训练，只有示意', 'No real training here — illustration only'), false, env, null, COL.rose);
  }
}
