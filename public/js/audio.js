// 全部用 WebAudio 即时合成，没有音频文件。默认静音。

let ctx = null;
let on = false;

function ac() {
  if (!ctx) ctx = new (window.AudioContext || window.webkitAudioContext)();
  if (ctx.state === 'suspended') ctx.resume();
  return ctx;
}

export const soundOn = () => on;
export function setSound(v) { on = !!v; if (on) ac(); }

function tone(freq, { t = 0, dur = 0.12, type = 'sine', gain = 0.08, to = null } = {}) {
  if (!on) return;
  const a = ac();
  const o = a.createOscillator();
  const g = a.createGain();
  const t0 = a.currentTime + t;
  o.type = type;
  o.frequency.setValueAtTime(freq, t0);
  if (to) o.frequency.exponentialRampToValueAtTime(to, t0 + dur);
  g.gain.setValueAtTime(0, t0);
  g.gain.linearRampToValueAtTime(gain, t0 + 0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  o.connect(g).connect(a.destination);
  o.start(t0);
  o.stop(t0 + dur + 0.02);
}

function whoosh(from, to, dur, gain = 0.1) {
  if (!on) return;
  const a = ac();
  const len = Math.floor(a.sampleRate * dur);
  const buf = a.createBuffer(1, len, a.sampleRate);
  const d = buf.getChannelData(0);
  for (let i = 0; i < len; i++) d[i] = Math.random() * 2 - 1;
  const src = a.createBufferSource();
  src.buffer = buf;
  const f = a.createBiquadFilter();
  f.type = 'lowpass';
  f.Q.value = 6;
  const t0 = a.currentTime;
  f.frequency.setValueAtTime(from, t0);
  f.frequency.exponentialRampToValueAtTime(to, t0 + dur);
  const g = a.createGain();
  g.gain.setValueAtTime(0, t0);
  g.gain.linearRampToValueAtTime(gain, t0 + dur * 0.25);
  g.gain.exponentialRampToValueAtTime(0.0001, t0 + dur);
  src.connect(f).connect(g).connect(a.destination);
  src.start(t0);
}

export const sfx = {
  click: () => tone(880, { dur: 0.06, to: 1320, gain: 0.05 }),
  hover: () => tone(1760, { dur: 0.03, gain: 0.015 }),
  tick: () => tone(2200, { dur: 0.025, type: 'square', gain: 0.02 }),
  dive: () => { whoosh(2400, 160, 1.1, 0.09); tone(140, { dur: 1, to: 45, gain: 0.08 }); },
  rise: () => { whoosh(200, 2600, 1.1, 0.08); tone(60, { dur: 1, to: 220, gain: 0.06 }); },
  discover: () => [660, 830, 990, 1320].forEach((f, i) => tone(f, { t: i * 0.07, dur: 0.35, type: 'triangle', gain: 0.05 })),
  land: () => [523, 659, 784].forEach((f) => tone(f, { dur: 0.6, type: 'triangle', gain: 0.04 })),
  bit: () => tone(440 + Math.random() * 400, { dur: 0.05, type: 'square', gain: 0.025 }),
};
