// 一个"示意"的小型 GPT。
// 结构参数取自真实的 GPT-2 Small（12 层 / 12 头 / 768 维 / 3072 个 MLP 神经元），
// 数值由确定性伪随机数和少量启发式规则生成：同样的输入永远得到同样的画面。
// 它不是真的在跑一个模型，而是把真实模型里"会发生什么"做成可以摸得到的样子。

export const CFG = {
  name: 'Mini-GPT',
  arch: 'GPT-2 Small 结构',
  layers: 12,
  heads: 12,
  dModel: 768,
  dHead: 64,
  dFF: 3072,
  vocab: 50257,
  params: 124439808,
  paramsPerLayer: 7087872,
};

/* ------------------------------------------------------------------ */
/* 确定性随机                                                          */
/* ------------------------------------------------------------------ */

export function hashStr(str, seed = 0) {
  let h1 = 0xdeadbeef ^ seed, h2 = 0x41c6ce57 ^ seed;
  for (let i = 0; i < str.length; i++) {
    const ch = str.charCodeAt(i);
    h1 = Math.imul(h1 ^ ch, 2654435761);
    h2 = Math.imul(h2 ^ ch, 1597334677);
  }
  h1 = Math.imul(h1 ^ (h1 >>> 16), 2246822507) ^ Math.imul(h2 ^ (h2 >>> 13), 3266489909);
  h2 = Math.imul(h2 ^ (h2 >>> 16), 2246822507) ^ Math.imul(h1 ^ (h1 >>> 13), 3266489909);
  return 4294967296 * (2097151 & h2) + (h1 >>> 0);
}

export function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export const rngFor = (...parts) => mulberry32(hashStr(parts.join('␟')) % 4294967296);

export function gaussian(r) {
  let u = 0, v = 0;
  while (u === 0) u = r();
  while (v === 0) v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export const clamp = (x, a, b) => Math.min(b, Math.max(a, x));

/* ------------------------------------------------------------------ */
/* 词汇与语义类别                                                      */
/* ------------------------------------------------------------------ */

export const CATS = {
  animal:  { name: '动物',   c: [-0.64, 0.50] },
  nature:  { name: '自然',   c: [-0.16, 0.74] },
  color:   { name: '颜色',   c: [0.36, 0.72] },
  place:   { name: '地点',   c: [0.74, 0.30] },
  people:  { name: '人物',   c: [0.62, -0.30] },
  tech:    { name: '科技',   c: [0.14, -0.70] },
  emotion: { name: '情感',   c: [-0.40, -0.16] },
  time:    { name: '时间',   c: [-0.80, 0.02] },
  poetry:  { name: '诗意',   c: [-0.34, 0.32] },
  num:     { name: '数字',   c: [0.10, 0.18] },
  func:    { name: '虚词',   c: [-0.64, -0.60] },
  punct:   { name: '标点',   c: [-0.16, -0.86] },
  misc:    { name: '其他',   c: [0.30, -0.06] },
  byte:    { name: '字节',   c: [0.84, -0.84] },
};

const LEX = {
  animal: '猫 狗 鸟 鱼 马 老虎 熊猫 兔子 cat dog bird fish horse',
  nature: '天气 天空 太阳 月亮 星星 雨 雪 风 云 霜 晴朗 大海 山 河 阳光 空气 春天 夏天 秋天 冬天 热 冷 水 sky sun rain snow water',
  color: '红 红色 蓝 蓝色 蓝光 绿 绿色 黄 黑 白 紫 red blue green black white',
  place: '北京 上海 中国 法国 巴黎 伦敦 日本 美国 故乡 地上 France Paris London China Lyon Tokyo city',
  people: '国王 女王 男人 女人 王子 公主 老师 学生 朋友 我们 你们 他们 我 你 他 她 人 king queen man woman people',
  tech: '人工智能 大模型 模型 语言模型 神经网络 机器学习 深度学习 计算机 电脑 手机 数据 算法 程序 代码 网络 智能 技术 科学 黑箱 AI GPT model models token network',
  emotion: '好 开心 难过 爱 喜欢 真 不错 美 快乐 讨厌 害怕 糟糕 棒 good love happy sad great',
  time: '今天 明天 昨天 现在 未来 过去 时候 时间 世纪 夜 today tomorrow future time',
  poetry: '明月 月 光 床前 举头 望 低头 思 诗 诗歌 moon',
  num: '一 二 三 四 五 六 七 八 九 十 百 千 万 one two three',
  func: '的 了 是 在 和 与 就 都 也 很 这 那 这个 那个 一个 为什么 什么 因为 所以 但是 如果 可以 将 由 被 得 最 中的 the of is a an to and in on it that was for as are be with at by this not',
};

const WORD_CAT = new Map();
for (const [cat, words] of Object.entries(LEX)) {
  for (const w of words.split(' ')) WORD_CAT.set(w.toLowerCase(), cat);
}

// 用来做"国王 − 男人 + 女人 ≈ 女王"的平行四边形
const FIXED_POS = {
  '男人': [0.44, -0.50], '女人': [0.80, -0.48], '国王': [0.42, -0.16], '女王': [0.78, -0.14],
  'man': [0.50, -0.56], 'woman': [0.86, -0.54], 'king': [0.48, -0.22], 'queen': [0.84, -0.20],
};

// 语义地图的背景词
export const MAP_WORDS = [
  '猫', '狗', '鸟', '熊猫', 'dog', '马',
  '天空', '太阳', '雨', '雪', '云', '霜', '大海', '春天', '风',
  '红色', '蓝色', '绿色', '黄', '白', 'blue',
  '北京', '中国', '法国', '巴黎', '伦敦', 'Paris', 'Tokyo',
  '国王', '女王', '男人', '女人', '老师', '朋友',
  '人工智能', '大模型', '计算机', '数据', '算法', '代码', 'AI',
  '开心', '难过', '喜欢', '爱', '害怕', 'love',
  '今天', '明天', '未来', '过去', '现在',
  '明月', '故乡', '诗歌', '举头', '低头',
  '1', '2', '3', '百', '万',
  '的', '了', '是', 'the', 'of', '因为',
  '，', '。', '？', '!', ',',
];

const RELATED = { 'nature|poetry': 0.8, 'nature|time': 0.35, 'place|people': 0.3, 'tech|people': 0.2, 'poetry|place': 0.35, 'emotion|people': 0.3, 'color|nature': 0.45 };
export function catSim(a, b) {
  if (a === b) return a === 'func' || a === 'punct' || a === 'misc' ? 0.35 : 1;
  return RELATED[`${a}|${b}`] ?? RELATED[`${b}|${a}`] ?? 0;
}

/* ------------------------------------------------------------------ */
/* 分词器（示意，接近 Qwen / GPT-4 这类 BPE 分词器的行为）               */
/* ------------------------------------------------------------------ */

const ZH_WORDS = new Set((
  '人工智能 大模型 语言模型 神经网络 机器学习 深度学习 计算机 为什么 什么 我们 你们 他们 今天 明天 昨天 天气 天空 蓝色 蓝光 ' +
  '未来 世界 中国 北京 上海 学习 知道 喜欢 因为 所以 但是 如果 可以 没有 一个 这个 那个 自己 时候 现在 问题 东西 电脑 ' +
  '手机 朋友 老师 学生 明月 故乡 地上 思考 智能 模型 数据 黑箱 不错 晴朗 散步 适合 出去 发展 共同 塑造 阳光 空气 散射 ' +
  '国王 女王 男人 女人 巴黎 法国 你好 谢谢 开心 难过 一起 非常 已经 正在 应该 需要 觉得 起来 出来 主要 重要 世纪 科学 ' +
  '技术 社会 生活 工作 研究 方法 能力 文化 历史 经济 信息 系统 程序 代码 游戏 音乐 电影 故事 小说 诗歌 春天 夏天 秋天 ' +
  '冬天 月亮 太阳 星星 大海 举头 低头 床前 熊猫 老虎 兔子 红色 绿色 伦敦 日本 美国 王子 公主 算法 网络 快乐 讨厌 害怕 ' +
  '糟糕 过去 时间 中的 充满 答案 解释 意思 如何 怎么 怎么样 预报 天安门 大学 哈哈 其实 翻译 写作 编程 宇宙 地球 生命'
).split(' '));

const EN_WORDS = new Set((
  'the of and to in is it that was for on are as with his they at be this have from or one had by word but not what all ' +
  'were we when your can said there use an each which she do how their if will up other about out many then them these so ' +
  'some her would make like him into time has look two more write go see number no way could people my than first water ' +
  'been call who its now find long down day did get come made may part capital france paris london city world hello cat ' +
  'dog sat mat love model models language learning machine neural network sky blue why because king queen man woman good ' +
  'great know think future today known eiffel tower located very new sun moon where after before little ai gpt token ' +
  'tokens attention inside black box deep red green white happy sad lyon tokyo china once upon'
).split(' '));

const SUFFIXES = ['ization', 'ational', 'fulness', 'ousness', 'iveness', 'ation', 'ness', 'ment', 'able', 'ible', 'tion', 'sion', 'ing', 'est', 'ful', 'less', 'ous', 'ive', 'ize', 'ly', 'ed', 'er'];
const PREFIXES = ['inter', 'trans', 'under', 'over', 'anti', 'dis', 'pre', 'un', 're', 'mis'];

function splitWord(w) {
  const lw = w.toLowerCase();
  if (w.length <= 7 || EN_WORDS.has(lw)) return [w];
  for (const p of PREFIXES) {
    if (lw.startsWith(p) && w.length - p.length >= 4) return [w.slice(0, p.length), ...splitWord(w.slice(p.length))];
  }
  for (const s of SUFFIXES) {
    if (lw.endsWith(s) && w.length - s.length >= 3) return [...splitWord(w.slice(0, w.length - s.length)), w.slice(w.length - s.length)];
  }
  return [w.slice(0, 4), ...splitWord(w.slice(4))];
}

const KNOWN_IDS = {
  '，': 3837, '。': 1773, ',': 11, '.': 13, ' the': 262, ' Paris': 6342, ' is': 318, ' of': 286, 'The': 464,
  ' France': 4881, ' capital': 3139, '\n': 198, ' ': 220, '1': 16, ' 2': 362, ' 3': 513, ' 4': 604, ' 5': 642,
};

const isHan = (c) => /\p{Script=Han}/u.test(c);
const isLatin = (c) => /[A-Za-zÀ-ɏ']/.test(c);
const isDigit = (c) => /[0-9]/.test(c);
const isEmojiLike = (c) => /\p{Extended_Pictographic}/u.test(c) || c.codePointAt(0) > 0xffff;
const enc = new TextEncoder();

function catOf(text, kind) {
  if (kind === 'byte') return 'byte';
  if (kind === 'punct' || kind === 'space') return 'punct';
  if (kind === 'num') return 'num';
  const key = text.trim().toLowerCase();
  if (WORD_CAT.has(key)) return WORD_CAT.get(key);
  if (kind === 'zh' && key.length > 1) {
    // 未收录的词：看里面有没有认识的字
    for (const ch of key) if (WORD_CAT.has(ch)) return WORD_CAT.get(ch);
  }
  return 'misc';
}

function makeTok(text, kind, extra = {}) {
  const id = extra.id ?? KNOWN_IDS[text] ?? (300 + (hashStr(text, 7) % 49900));
  return { text, kind, id, cat: catOf(text, kind), ...extra };
}

export function tokenize(input) {
  const chars = Array.from(input);
  const out = [];
  let i = 0;
  while (i < chars.length) {
    const c = chars[i];

    if (isHan(c)) {
      let w = c;
      for (let L = 4; L >= 2; L--) {
        const cand = chars.slice(i, i + L).join('');
        if (Array.from(cand).length === L && ZH_WORDS.has(cand)) { w = cand; break; }
      }
      out.push(makeTok(w, 'zh'));
      i += Array.from(w).length;
      continue;
    }

    // 单词：前导空格并入单词，这是 BPE 分词器的典型行为
    let lead = '';
    let j = i;
    if (c === ' ' && i + 1 < chars.length && (isLatin(chars[i + 1]) || isDigit(chars[i + 1]))) { lead = ' '; j = i + 1; }

    if (isLatin(chars[j])) {
      let k = j;
      while (k < chars.length && isLatin(chars[k])) k++;
      const pieces = splitWord(chars.slice(j, k).join(''));
      pieces.forEach((p, n) => out.push(makeTok((n === 0 ? lead : '') + p, 'en', { sub: pieces.length > 1 })));
      i = k;
      continue;
    }

    if (isDigit(chars[j])) {
      // 很多分词器把数字逐位切开
      out.push(makeTok(lead + chars[j], 'num'));
      i = j + 1;
      continue;
    }

    if (c === '\n') { out.push(makeTok('\n', 'space')); i++; continue; }
    if (/\s/.test(c)) {
      let k = i;
      while (k < chars.length && /[ \t]/.test(chars[k]) && !(k + 1 < chars.length && chars[k] === ' ' && (isLatin(chars[k + 1]) || isDigit(chars[k + 1])))) k++;
      if (k === i) k = i + 1;
      out.push(makeTok(chars.slice(i, k).join(''), 'space'));
      i = k;
      continue;
    }

    if (isEmojiLike(c)) {
      // 词表里没有：退回到 UTF-8 字节
      const bytes = enc.encode(c);
      const group = out.length;
      bytes.forEach((b) => out.push(makeTok(`<0x${b.toString(16).toUpperCase().padStart(2, '0')}>`, 'byte', { id: b, of: c, group })));
      i++;
      continue;
    }

    if (/[\p{P}\p{S}]/u.test(c)) { out.push(makeTok(c, 'punct')); i++; continue; }

    out.push(makeTok(c, 'other'));
    i++;
  }
  out.forEach((t, n) => { t.i = n; });
  return out;
}

export function tokenHue(t) {
  return hashStr(t.text, 3) % 360;
}

/* ------------------------------------------------------------------ */
/* 嵌入                                                                */
/* ------------------------------------------------------------------ */

const embCache = new Map();
export function embedding(t) {
  const key = t.text;
  if (embCache.has(key)) return embCache.get(key);
  const d = CFG.dModel;
  const v = new Float32Array(d);
  const rc = rngFor('cat', t.cat);
  const rt = rngFor('emb', key);
  const wc = t.cat === 'misc' ? 0.25 : t.cat === 'func' || t.cat === 'punct' ? 0.45 : 0.62;
  for (let i = 0; i < d; i++) v[i] = 0.9 * (wc * gaussian(rc) + (1 - wc * 0.6) * 0.55 * gaussian(rt));
  embCache.set(key, v);
  return v;
}

export function posEnc(pos, i, d = CFG.dModel) {
  const k = Math.floor(i / 2);
  const ang = pos / Math.pow(10000, (2 * k) / d);
  return i % 2 === 0 ? Math.sin(ang) : Math.cos(ang);
}

export function mapPos(t) {
  const key = t.text.trim();
  if (FIXED_POS[key]) return FIXED_POS[key];
  const cat = CATS[t.cat] ?? CATS.misc;
  const r = rngFor('map', key);
  if (t.cat === 'misc') return [-0.1 + (r() - 0.5) * 1.6, -0.2 + (r() - 0.5) * 1.2];
  const a = r() * Math.PI * 2, rad = 0.06 + Math.sqrt(r()) * 0.17;
  return [cat.c[0] + Math.cos(a) * rad, cat.c[1] + Math.sin(a) * rad];
}

/* ------------------------------------------------------------------ */
/* 注意力                                                              */
/* ------------------------------------------------------------------ */

export const HEAD_TYPES = {
  prev:   { name: '前一词头', en: 'Previous-token', desc: '几乎总是盯着紧挨着的前一个词元，帮模型记住“刚才说到哪儿了”。' },
  sink:   { name: '汇聚头', en: 'Attention sink', desc: '把大部分注意力“停”在第一个词元上。没有值得看的东西时，它就看开头，当作一个安全的空位。' },
  induct: { name: '归纳头', en: 'Induction', desc: '找到当前词上一次出现的位置，再看它后面跟着什么，然后照着预测。这是“上下文学习”的核心回路之一。' },
  punct:  { name: '标点头', en: 'Punctuation', desc: '盯着最近的标点符号，用来感知句子和分句的边界。' },
  sem:    { name: '语义头', en: 'Semantic', desc: '去看意思相关的词。比如“霜”会看“月”“光”，“Paris”会看“France”。' },
  self:   { name: '自我头', en: 'Self', desc: '主要看自己，把当前词元的信息再强化一遍。' },
  broad:  { name: '广角头', en: 'Broad', desc: '把注意力均匀地摊开，相当于给整段上下文取个平均。' },
  recent: { name: '近邻头', en: 'Local window', desc: '只看最近的几个词元，越近权重越高，像一个滑动窗口。' },
};

const HEAD_PLANS = {
  early: ['prev', 'self', 'recent', 'broad', 'punct', 'prev', 'recent', 'self', 'sink', 'punct', 'broad', 'prev'],
  mid:   ['induct', 'prev', 'sink', 'punct', 'sem', 'recent', 'induct', 'sink', 'broad', 'prev', 'sem', 'punct'],
  late:  ['sem', 'sink', 'sink', 'induct', 'broad', 'sem', 'punct', 'sink', 'sem', 'recent', 'sink', 'broad'],
};

export function headType(layer, head) {
  const plan = layer <= 2 ? HEAD_PLANS.early : layer <= 7 ? HEAD_PLANS.mid : HEAD_PLANS.late;
  return plan[(head + layer * 5) % 12];
}

export function findHead(layer, type) {
  for (let h = 0; h < CFG.heads; h++) if (headType(layer, h) === type) return h;
  return -1;
}

const isPunctTok = (t) => t.kind === 'punct' || t.kind === 'space';

export function attention(tokens, layer, head) {
  const type = headType(layer, head);
  const T = tokens.length;
  const r = rngFor('attn', layer, head, tokens.map((t) => t.text).join('|'));
  const W = [];
  for (let i = 0; i < T; i++) {
    const s = new Array(T).fill(-Infinity);
    let prevOcc = -1;
    if (type === 'induct') for (let p = i - 1; p >= 0; p--) if (tokens[p].text.trim() === tokens[i].text.trim() && !isPunctTok(tokens[i])) { prevOcc = p; break; }
    let lastPunct = -1;
    if (type === 'punct') for (let p = i - 1; p >= 0; p--) if (isPunctTok(tokens[p])) { lastPunct = p; break; }
    for (let j = 0; j <= i; j++) {
      let v = 0;
      switch (type) {
        case 'prev': v = j === i - 1 || (i === 0 && j === 0) ? 4.2 : 0; break;
        case 'sink': v = j === 0 ? 3.6 : 0; break;
        case 'induct': v = prevOcc >= 0 ? (j === prevOcc + 1 ? 5.2 : j === 0 ? 1.2 : 0) : (j === 0 ? 2.8 : 0); break;
        case 'punct': v = lastPunct >= 0 ? (j === lastPunct ? 4 : 0) : (j === 0 ? 1.5 : 0); break;
        case 'sem': v = j === i ? 0.6 : 3.1 * catSim(tokens[i].cat, tokens[j].cat) + (j === 0 ? 0.8 : 0); break;
        case 'self': v = j === i ? 4 : 0; break;
        case 'broad': v = 0; break;
        case 'recent': v = -0.9 * (i - j); break;
      }
      s[j] = v + 0.45 * gaussian(r);
    }
    W.push(softmax(s));
  }
  return { type, W };
}

export function softmax(arr, T = 1) {
  let m = -Infinity;
  for (const v of arr) if (v > m) m = v;
  const e = arr.map((v) => (v === -Infinity ? 0 : Math.exp((v - m) / T)));
  const z = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / z);
}

/* ------------------------------------------------------------------ */
/* MLP 神经元                                                          */
/* ------------------------------------------------------------------ */

export const CONCEPTS = [
  { id: 417,  label: '夜晚 · 月亮', test: (t) => t.cat === 'poetry' || /月|夜|霜|moon/i.test(t.text) },
  { id: 1203, label: '句子边界', test: (t) => t.cat === 'punct' },
  { id: 2951, label: '数字 · 计数', test: (t) => t.cat === 'num' },
  { id: 88,   label: '国家与城市', test: (t) => t.cat === 'place' },
  { id: 1777, label: '正面情绪', test: (t) => t.cat === 'emotion' },
  { id: 2310, label: '科技与计算', test: (t) => t.cat === 'tech' },
  { id: 666,  label: '英文单词', test: (t) => t.kind === 'en' },
  { id: 3001, label: '天气', test: (t) => t.cat === 'nature' },
  { id: 999,  label: '人物与身份', test: (t) => t.cat === 'people' },
  { id: 2048, label: '句首位置', test: (t) => t.i === 0 },
  { id: 5,    label: '时间表达', test: (t) => t.cat === 'time' },
  { id: 1314, label: '动物', test: (t) => t.cat === 'animal' },
  { id: 2600, label: '颜色', test: (t) => t.cat === 'color' },
  { id: 72,   label: '虚词 · 语法', test: (t) => t.cat === 'func' },
  { id: 1460, label: '多义：月亮 / 法国 / 数字', poly: true, test: (t) => t.cat === 'poetry' || t.cat === 'place' || t.cat === 'num' },
  { id: 2222, label: '子词拼接', test: (t) => !!t.sub },
  { id: 256,  label: '字节碎片', test: (t) => t.kind === 'byte' },
];
export const CONCEPT_BY_ID = new Map(CONCEPTS.map((c) => [c.id, c]));

export const ACTS = {
  GELU: (x) => 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x))),
  ReLU: (x) => Math.max(0, x),
  SiLU: (x) => x / (1 + Math.exp(-x)),
};

const actCache = new Map();
export function neuronPre(t, layer) {
  const key = `${layer}|${t.i}|${t.text}`;
  if (actCache.has(key)) return actCache.get(key);
  const n = CFG.dFF;
  const z = new Float32Array(n);
  const r = rngFor('act', t.text, layer, t.i);
  for (let j = 0; j < n; j++) {
    z[j] = r() < 0.018 ? 0.6 + 1.9 * r() : -1.15 + 0.7 * gaussian(r);
  }
  const strength = 0.75 + 0.5 * Math.sin((Math.PI * (layer + 1)) / (CFG.layers + 1));
  for (const c of CONCEPTS) {
    if (c.test(t)) z[c.id] = (2.4 + 1.3 * r()) * strength;
    else z[c.id] = Math.min(z[c.id], -0.4);
  }
  if (actCache.size > 400) actCache.clear();
  actCache.set(key, z);
  return z;
}

/* ------------------------------------------------------------------ */
/* 预测下一个词元                                                      */
/* ------------------------------------------------------------------ */

export const PRESETS = [
  { label: '床前明月光', prompt: '床前明月光，疑是地上', cont: ['霜', '。', '举头', '望', '明月', '，', '低头', '思', '故乡', '。'],
    alts: [['霜', 0.86], ['雪', 0.05], ['光', 0.03], ['月', 0.02], ['白', 0.012], ['水', 0.008]] },
  { label: 'The capital of France', prompt: 'The capital of France is', cont: [' Paris', '.', ' It', ' is', ' known', ' for', ' the', ' Eiffel', ' Tower', '.'],
    alts: [[' Paris', 0.82], [' a', 0.05], [' the', 0.04], [' located', 0.025], [' Lyon', 0.012], [' not', 0.008]] },
  { label: '今天天气真', prompt: '今天天气真', cont: ['好', '，', '适合', '出去', '散步', '。'],
    alts: [['好', 0.71], ['不错', 0.14], ['热', 0.06], ['冷', 0.04], ['晴朗', 0.02], ['糟糕', 0.01]] },
  { label: '1, 2, 3, 4,', prompt: '1, 2, 3, 4,', cont: [' 5', ',', ' 6', ',', ' 7', ',', ' 8', ','],
    alts: [[' 5', 0.93], [' 4', 0.02], [' 6', 0.015], [' 10', 0.008], [' 1', 0.006], ['\n', 0.004]] },
  { label: '天空为什么是蓝色的？', prompt: '天空为什么是蓝色的？', cont: ['因为', '阳光', '中的', '蓝光', '被', '空气', '散射', '得', '最', '多', '。'],
    alts: [['因为', 0.46], ['这', 0.14], ['天空', 0.11], ['\n', 0.08], ['主要', 0.05], ['其实', 0.04]] },
  { label: '人工智能的未来', prompt: '人工智能的未来', cont: ['将', '由', '我们', '共同', '塑造', '。'],
    alts: [['将', 0.24], ['是', 0.21], ['，', 0.14], ['发展', 0.09], ['在', 0.07], ['充满', 0.05]] },
];

const NEXT = {
  '天': ['气', '空', '下', '上', '啊', '的'], '人': ['们', '类', '工', '生', '的', '和'], '我': ['们', '的', '是', '想', '觉得', '很'],
  '你': ['好', '的', '是', '们', '在', '说'], '他': ['们', '的', '说', '是', '在', '很'], '大': ['家', '学', '模型', '的', '小', '量'],
  '中': ['国', '的', '文', '心', '间', '午'], '学': ['习', '生', '校', '者', '会', '问'], '今天': ['天气', '是', '我', '的', '，', '很'],
  '天气': ['真', '很', '好', '预报', '不错', '怎么样'], '明天': ['会', '是', '的', '我', '早上', '见'], '我们': ['的', '是', '一起', '可以', '要', '在'],
  '人工智能': ['的', '是', '将', '技术', '正在', '和'], '大模型': ['的', '是', '如何', '可以', '在', '正在'], '模型': ['的', '是', '会', '可以', '在', '输出'],
  '的': ['人', '时候', '是', '问题', '东西', '世界'], '是': ['一个', '我', '什么', '的', '不', '在'], '在': ['这里', '我', '家', '中国', '一起', '的'],
  '了': ['。', '，', '一个', '我', '吗', '！'], '很': ['好', '多', '大', '重要', '难', '喜欢'], '真': ['好', '的', '是', '不错', '美', '棒'],
  '什么': ['？', '是', '样', '时候', '东西', '呢'], '为什么': ['会', '是', '要', '？', '不', '呢'], '因为': ['我', '他', '这', '它', '天空', '没有'],
  '世界': ['上', '的', '是', '，', '和', '。'], '中国': ['的', '是', '人', '有', '，', '在'], '北京': ['是', '的', '，', '天安门', '大学', '市'],
  '爱': ['你', '情', '的', '是', '上', '着'], '喜欢': ['你', '的', '吃', '看', '这', '听'], '好': ['的', '，', '。', '吗', '！', '像'],
  '未来': ['的', '是', '将', '，', '会', '已经'], '黑箱': ['里', '的', '是', '，', '中', '模型'], '猫': ['咪', '在', '是', '的', '和', '坐'],
  '你好': ['，', '！', '。', '呀', '我', '啊'], '谢谢': ['你', '！', '，', '。', '大家', '啦'],
  'the': [' world', ' first', ' same', ' best', ' most', ' end'], 'is': [' a', ' the', ' not', ' very', ' an', ' one'],
  'i': [' am', ' think', ' have', "'m", ' was', ' love'], 'hello': [',', '!', ' world', ' there', '.', ' everyone'],
  'of': [' the', ' a', ' his', ' this', ' all', ' its'], 'a': [' lot', ' new', ' good', ' very', ' few', ' little'],
  'to': [' the', ' be', ' make', ' get', ' see', ' do'], 'and': [' the', ' I', ' a', ' then', ' it', ' we'],
  'cat': [' is', ' sat', ' was', "'s", ' and', ' in'], 'sat': [' on', ' down', ' in', ' at', ' by', ' there'],
  'on': [' the', ' a', ' his', ' top', ' my', ' it'], 'love': [' you', ' the', ' it', ' is', ' to', ' and'],
  'ai': [' is', ' will', ' can', ' has', ' models', ','], 'once': [' upon', ' again', ' more', ',', ' a', ' the'],
  'upon': [' a', ' the', ' time', ' his', ' her', ' it'],
  '，': ['我', '但', '他', '这', '所以', '也'], '。': ['\n', '我', '他', '这', '但是', '在'], '？': ['\n', '因为', '这', '我', '答', '其实'],
  '！': ['\n', '我', '你', '这', '哈哈', '真'], ',': [' and', ' I', ' but', ' the', ' which', ' it'], '.': ['\n', ' The', ' I', ' It', ' This', ' We'],
  '?': ['\n', ' I', ' The', ' What', ' It', ' Yes'], '!': ['\n', ' I', ' The', ' You', ' It', ' We'],
};
const ZH_DEFAULT = ['的', '，', '是', '了', '。', '在', '和', '我'];
const EN_DEFAULT = [' the', ',', ' and', ' to', '.', ' of', ' a', ' is'];
export const FILLER = ['泉', ' quantum', 'ing', '<0xE5>', '@', '…', ' banana', '彼', 'の', ' 然后', 'Ω', ' Tokyo', '器', ' seventeen', '#', '逻辑', ' pixel', '氢', '::', ' und', '阿', ' Mozart', '鲸', '{', ' tomato', '辑', ' sql', '凹'];

function isEnglishish(tokens) {
  const tail = tokens.slice(-4);
  return tail.filter((t) => t.kind === 'en').length >= tail.filter((t) => t.kind === 'zh').length && tail.some((t) => t.kind === 'en' || t.kind === 'num');
}

function heuristicCands(tokens) {
  const last = tokens[tokens.length - 1];
  if (!last) return ZH_DEFAULT.slice(0, 6);
  const key = last.text.trim().toLowerCase();
  let cands = NEXT[last.text] || NEXT[key];
  if (!cands && last.kind === 'zh') {
    const ch = Array.from(last.text).pop();
    cands = NEXT[ch];
  }
  if (!cands && last.kind === 'num') {
    const d = Number(key);
    cands = [String((d + 1) % 10), '0', ',', '.', '年', ' '];
  }
  if (!cands) cands = isEnglishish(tokens) ? EN_DEFAULT : ZH_DEFAULT;
  // 归纳头的行为：如果当前词元之前出现过，下一个词很可能重复上次跟在后面的那个
  for (let p = tokens.length - 2; p >= 0; p--) {
    if (tokens[p].text.trim() === last.text.trim() && !isPunctTok(last) && p + 1 < tokens.length - 1) {
      const nxt = tokens[p + 1].text;
      cands = [nxt, ...cands.filter((c) => c !== nxt)];
      break;
    }
  }
  return cands.slice(0, 6);
}

function distFrom(list, tail, seedKey, topBoost = null) {
  const r = rngFor('dist', seedKey);
  const base = [0.4, 0.2, 0.12, 0.08, 0.06, 0.045];
  let items = list.map((text, k) => ({ text, w: base[k] * (0.7 + 0.6 * r()) }));
  if (topBoost != null) items[0].w = topBoost;
  items.sort((a, b) => b.w - a.w);
  const sum = items.reduce((a, b) => a + b.w, 0);
  return items.map((it) => ({ text: it.text, p: (it.w / sum) * (1 - tail) }));
}

export function predict(prompt, tokens) {
  let cands = null;
  let source = 'heuristic';
  for (const pre of PRESETS) {
    if (!prompt.startsWith(pre.prompt)) continue;
    const rest = prompt.slice(pre.prompt.length);
    if (rest === '') {
      const tail = 1 - pre.alts.reduce((a, b) => a + b[1], 0);
      cands = pre.alts.map(([text, p]) => ({ text, p }));
      cands.tail = tail;
      source = 'curated';
      break;
    }
    let acc = '', k = 0;
    while (k < pre.cont.length && acc.length < rest.length) acc += pre.cont[k++];
    if (acc === rest && k < pre.cont.length) {
      const top = pre.cont[k];
      const r = rngFor('cur', prompt);
      const others = heuristicCands(tokens).filter((c) => c !== top).slice(0, 5);
      const tail = 0.03 + 0.04 * r();
      cands = distFrom([top, ...others], tail, prompt, 3.2 + 3 * r());
      cands.tail = tail;
      source = 'curated';
    }
    break;
  }
  if (!cands) {
    const r = rngFor('tail', prompt);
    const tail = 0.04 + 0.05 * r();
    cands = distFrom(heuristicCands(tokens), tail, prompt);
    cands.tail = tail;
  }
  const tail = cands.tail;
  const out = cands.map((c) => ({ ...c, logit: Math.log(c.p) }));
  // 剩下的五万多个词元分成三档，温度升高时它们会迅速"涨潮"
  const rest = CFG.vocab - out.length;
  const buckets = [
    { count: 100, mass: tail * 0.8 },
    { count: 2000, mass: tail * 0.17 },
    { count: rest - 2100, mass: tail * 0.03 },
  ].map((b) => ({ ...b, logit: Math.log(b.mass / b.count) }));
  return { cands: out, buckets, source };
}

export function withTemperature(dist, T) {
  if (T < 0.06) {
    const cands = dist.cands.map((c, k) => ({ ...c, q: k === 0 ? 1 : 0 }));
    return { cands, other: 0 };
  }
  let m = -Infinity;
  for (const c of dist.cands) m = Math.max(m, c.logit / T);
  const ws = dist.cands.map((c) => Math.exp(c.logit / T - m));
  const bw = dist.buckets.reduce((a, b) => a + b.count * Math.exp(b.logit / T - m), 0);
  const z = ws.reduce((a, b) => a + b, 0) + bw;
  return { cands: dist.cands.map((c, k) => ({ ...c, q: ws[k] / z })), other: bw / z };
}

export function sampleFrom(distT, r = Math.random) {
  let x = r();
  for (let k = 0; k < distT.cands.length; k++) {
    x -= distT.cands[k].q;
    if (x <= 0) return { k, text: distT.cands[k].text };
  }
  return { k: -1, text: FILLER[Math.floor(r() * FILLER.length)] };
}

// 逻辑透镜：把每一层的中间状态直接接上输出头，看看"此刻它会猜什么"
export function logitLens(tokens, dist) {
  const last = tokens[tokens.length - 1];
  const echo = [last?.text ?? '', '的', '，', ' the', '。', '是', FILLER[hashStr(last?.text ?? '') % FILLER.length]];
  const rows = [];
  for (let L = 0; L < CFG.layers; L++) {
    const a = Math.pow(clamp((L - 2.5) / 8.5, 0, 1), 1.5);
    const r = rngFor('lens', L, last?.text ?? '');
    const pool = new Map();
    dist.cands.forEach((c) => pool.set(c.text, (pool.get(c.text) || 0) + a * Math.pow(c.p, 0.35 + 0.65 * a)));
    echo.forEach((e, k) => pool.set(e, (pool.get(e) || 0) + (1 - a) * (k === 0 ? 0.5 : 0.25 * r())));
    const arr = [...pool.entries()].filter(([t]) => t !== '').map(([text, w]) => ({ text, w }));
    const sum = arr.reduce((s, x) => s + x.w, 0) * (1.25 - 0.2 * a);
    arr.forEach((x) => { x.p = x.w / sum; });
    arr.sort((x, y) => y.p - x.p);
    rows.push(arr.slice(0, 5));
  }
  return rows;
}

/* ------------------------------------------------------------------ */
/* 汇总：一次"前向传播"                                                */
/* ------------------------------------------------------------------ */

export function analyze(prompt) {
  const tokens = tokenize(prompt);
  const dist = predict(prompt, tokens);
  return {
    prompt,
    tokens,
    dist,
    lens: logitLens(tokens, dist),
    hasRepeat: tokens.some((t, i) => t.kind !== 'punct' && t.kind !== 'space' && tokens.slice(0, i).some((u) => u.text.trim() === t.text.trim())),
  };
}

/* ------------------------------------------------------------------ */
/* 浮点数                                                              */
/* ------------------------------------------------------------------ */

const f32 = new Float32Array(1);
const u32 = new Uint32Array(f32.buffer);

export function floatBits(x, fmt) {
  f32[0] = x;
  const b = u32[0];
  if (fmt === 'FP32') return Array.from({ length: 32 }, (_, i) => (b >>> (31 - i)) & 1);
  if (fmt === 'BF16') {
    const r = (b + 0x7fff + ((b >>> 16) & 1)) >>> 16; // 就近舍入到偶数
    return Array.from({ length: 16 }, (_, i) => (r >>> (15 - i)) & 1);
  }
  if (fmt === 'FP16') return fp16Bits(x);
  return null;
}

function fp16Bits(x) {
  const s = x < 0 || Object.is(x, -0) ? 1 : 0;
  let a = Math.abs(x), e = 0, m = 0;
  if (a === 0) { e = 0; m = 0; }
  else if (a >= 65520) { e = 31; m = 0; }
  else {
    let ex = Math.floor(Math.log2(a));
    if (ex < -14) { e = 0; m = Math.round(a / Math.pow(2, -24)); }
    else {
      m = Math.round((a / Math.pow(2, ex) - 1) * 1024);
      if (m === 1024) { m = 0; ex++; }
      e = ex + 15;
    }
  }
  const v = (s << 15) | (e << 10) | m;
  return Array.from({ length: 16 }, (_, i) => (v >>> (15 - i)) & 1);
}

export const FORMATS = {
  FP32: { bits: 32, exp: 8, man: 23, bias: 127, name: '单精度浮点', note: '训练时的“全精度”' },
  BF16: { bits: 16, exp: 8, man: 7, bias: 127, name: 'Brain Float 16', note: '当今大模型最常用的存储格式' },
  FP16: { bits: 16, exp: 5, man: 10, bias: 15, name: '半精度浮点', note: '范围更小，精度更高' },
  INT8: { bits: 8, name: '8 位整数量化', note: '数值 ≈ 整数 × 缩放系数' },
  INT4: { bits: 4, name: '4 位整数量化', note: '只剩 16 个可能的取值' },
};
export const QUANT_SCALE = 0.25; // 假设这一行权重的最大绝对值

export function bitsToValue(bits, fmt) {
  const F = FORMATS[fmt];
  if (fmt === 'INT8' || fmt === 'INT4') {
    let q = 0;
    for (const b of bits) q = q * 2 + b;
    if (bits[0]) q -= 1 << bits.length; // 补码
    const qmax = (1 << (bits.length - 1)) - 1;
    return { value: (q / qmax) * QUANT_SCALE, q, qmax };
  }
  const s = bits[0];
  let e = 0;
  for (let i = 1; i <= F.exp; i++) e = e * 2 + bits[i];
  let m = 0;
  for (let i = F.exp + 1; i < F.bits; i++) m = m * 2 + bits[i];
  const emax = (1 << F.exp) - 1;
  let value;
  if (e === emax) value = m === 0 ? Infinity : NaN;
  else if (e === 0) value = (m / Math.pow(2, F.man)) * Math.pow(2, 1 - F.bias);
  else value = (1 + m / Math.pow(2, F.man)) * Math.pow(2, e - F.bias);
  if (s) value = -value;
  return { value, s, e, m, bias: F.bias, man: F.man };
}

export function quantize(x, fmt) {
  const nb = FORMATS[fmt].bits;
  const qmax = (1 << (nb - 1)) - 1;
  const q = clamp(Math.round((x / QUANT_SCALE) * qmax), -qmax, qmax);
  const u = q < 0 ? q + (1 << nb) : q;
  return Array.from({ length: nb }, (_, i) => (u >>> (nb - 1 - i)) & 1);
}
