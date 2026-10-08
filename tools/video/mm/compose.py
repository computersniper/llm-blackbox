#!/usr/bin/env python3
"""多模态视频的原创配乐（从 ../compose.py 改来）：全部用 numpy / scipy 现场合成，不用任何采样或下载的音乐。

速度 96 BPM（一拍 0.625 秒），段落和分镜表（mm/score.js）用同一张时间表：render.mjs events 导出的 events.json
里有每一段的起止和能量，以及画面里的事件（点选图片、敲出问题的每个词元、切块、ViT 的每一层、28 层……）。

  聊天     低频嗡鸣 + 空气声，选图、敲字的轻响；按下发送，和弦进来
  飞行     心跳、上行；片名处一记重拍，黑箱打开的机械声
  切块/乘加 讲解段：安静的铺底，稀疏的琶音，切块时水滴声
  ViT      鼓、贝斯进场；每过一层一个音，沿 D 小调五声音阶爬上去
  合并/插入 心跳，往下一段推
  看图回答  心跳；逐层看「3」的注意力时律动进来，盯住绿苹果那一层一声钟
  透镜/回答 律动回来，最后回到主和弦
  片尾      和弦解决到 D 大调，混响拖尾，淡出

母带：K 加权积分响度（ITU-R BS.1770）归一到 −14 LUFS，4 倍过采样的真峰值限制在 −2 dBTP。

    /mnt/d/cjc/venvs/blackbox/bin/python tools/video/mm/compose.py --events tools/video/mm/events.json --out /mnt/d/cjc/videos/multimodal/score.wav
"""
import argparse
import json
import math

import numpy as np
import scipy.signal as ss
from scipy.io import wavfile
from scipy.ndimage import minimum_filter1d

SR = 48000
BPM = 96
BEAT = 60 / BPM
BAR = 4 * BEAT
RNG = np.random.default_rng(20261001)


# ---------------------------------------------------------------- 基础

def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def n_of(sec):
    return int(round(sec * SR))


def tvec(n):
    return np.arange(n) / SR


def pan_gain(p):
    """-1 左 … +1 右，等功率"""
    a = (p + 1) * math.pi / 4
    return math.cos(a), math.sin(a)


class Bus:
    """立体声总线：按秒往里叠加单声道或立体声片段"""

    def __init__(self, dur):
        self.n = n_of(dur) + SR * 6
        self.x = np.zeros((2, self.n), np.float64)

    def add(self, t, sig, gain=1.0, pan=0.0):
        i = n_of(t)
        if i >= self.n:
            return
        if sig.ndim == 1:
            gl, gr = pan_gain(pan)
            sig = np.stack([sig * gl, sig * gr])
        j = min(self.n, i + sig.shape[1])
        if i < 0:
            sig = sig[:, -i:]
            j = min(self.n, sig.shape[1])
            i = 0
        self.x[:, i:j] += gain * sig[:, : j - i]


def sos(kind, f, order=2, q=None):
    if kind == 'bp':
        return ss.butter(order, [f[0], f[1]], 'bandpass', fs=SR, output='sos')
    return ss.butter(order, f, kind, fs=SR, output='sos')


def filt(x, kind, f, order=2):
    return ss.sosfilt(sos(kind, f, order), x, axis=-1)


def shelf(x, f0, gain_db, high=True, q=0.707):
    """RBJ 搁架滤波（母带上做一点整体的高低音倾斜）"""
    A = 10 ** (gain_db / 40)
    w = 2 * math.pi * f0 / SR
    al = math.sin(w) / (2 * q)
    c = math.cos(w)
    s = 1 if high else -1
    b0 = A * ((A + 1) + s * (A - 1) * c + 2 * math.sqrt(A) * al)
    b1 = -2 * s * A * ((A - 1) + s * (A + 1) * c)
    b2 = A * ((A + 1) + s * (A - 1) * c - 2 * math.sqrt(A) * al)
    a0 = (A + 1) - s * (A - 1) * c + 2 * math.sqrt(A) * al
    a1 = 2 * s * ((A - 1) - s * (A + 1) * c)
    a2 = (A + 1) - s * (A - 1) * c - 2 * math.sqrt(A) * al
    return ss.lfilter([b0 / a0, b1 / a0, b2 / a0], [1, a1 / a0, a2 / a0], x, axis=-1)


def saw(freq, n, phase=0.0):
    """抗混叠锯齿波（polyBLEP），freq 可以是常数或逐样本数组"""
    f = np.broadcast_to(np.asarray(freq, np.float64), (n,))
    dt = f / SR
    ph = (phase + np.cumsum(dt)) % 1.0
    y = 2 * ph - 1
    m = ph < dt
    x = ph[m] / dt[m]
    y[m] -= x + x - x * x - 1
    m = ph > 1 - dt
    x = (ph[m] - 1) / dt[m]
    y[m] -= x * x + x + x + 1
    return y


def sine(freq, n, phase=0.0):
    f = np.broadcast_to(np.asarray(freq, np.float64), (n,))
    return np.sin(2 * np.pi * (phase + np.cumsum(f) / SR))


def env_ar(n, a, r, curve=1.0):
    t = tvec(n)
    e = np.minimum(1, t / max(a, 1e-4))
    rel = np.exp(-np.maximum(0, t - a) / max(r, 1e-4))
    return (e ** curve) * rel


def env_asr(n, a, r):
    """起 a 秒、收 r 秒，中间保持"""
    t = tvec(n)
    dur = n / SR
    e = np.minimum(1, t / max(a, 1e-4)) * np.minimum(1, np.maximum(0, (dur - t) / max(r, 1e-4)))
    return e * e * (3 - 2 * e)


def noise(n):
    return RNG.standard_normal(n)


def soft(y, ms=1.5):
    """开头 ms 毫秒的淡入：去掉起音瞬间的阶跃"""
    k = min(len(y), max(2, int(SR * ms / 1000)))
    y = y.copy()
    y[:k] *= np.sin(np.linspace(0, np.pi / 2, k)) ** 2
    return y


# ---------------------------------------------------------------- 混响 / 延迟

def make_ir(rt60=3.2, pre=0.018, dur=4.5, dark=5200):
    n = n_of(dur)
    t = tvec(n)
    ir = np.zeros((2, n))
    for c in range(2):
        nz = noise(n)
        hi = filt(nz, 'high', 1800) * np.exp(-t * 6.9 / (rt60 * 0.45))
        lo = filt(nz, 'low', dark) * np.exp(-t * 6.9 / rt60)
        ir[c] = lo + 0.35 * hi
        # 早期反射
        for k in range(10):
            d = n_of(pre + RNG.uniform(0.004, 0.09))
            ir[c, d] += RNG.uniform(-0.6, 0.6) * (1 - k / 12)
    ir[:, : n_of(pre)] = 0
    ir /= np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True))
    return ir


def reverb(x, ir, wet=1.0):
    out = np.zeros_like(x)
    for c in range(2):
        out[c] = ss.oaconvolve(x[c], ir[c])[: x.shape[1]]
    return out * wet


def pingpong(x, delay, fb=0.38, taps=7, lp=3800):
    """乒乓延迟：回声左右交替，越往后越暗"""
    out = np.zeros_like(x)
    d = n_of(delay)
    mono = 0.5 * (x[0] + x[1])
    for k in range(1, taps + 1):
        g = fb ** k
        sh = d * k
        if sh >= x.shape[1]:
            break
        y = filt(mono[:-sh], 'low', max(900, lp / (1 + 0.35 * k)))
        out[k % 2, sh:] += g * y
    return out


# ---------------------------------------------------------------- 乐器

def pad_voice(m, dur, bright=0.5, detune=0.08):
    n = n_of(dur)
    f = mtof(m)
    s = np.zeros(n)
    for k, dc in enumerate((-detune, 0, detune)):
        ff = f * 2 ** (dc / 12)
        lfo = 1 + 0.0016 * np.sin(2 * np.pi * (0.13 + 0.05 * k) * tvec(n) + k)
        s += saw(ff * lfo, n, RNG.random())
    s /= 3
    dark = filt(s, 'low', 700 + 300 * bright, 2)
    brt = filt(s, 'low', 2200 + 2600 * bright, 2)
    return dark * (1 - bright) + brt * bright


def pad_chord(notes, dur, bright, a=0.45, r=1.3):
    n = n_of(dur + r)
    out = np.zeros((2, n))
    for i, m in enumerate(notes):
        v = pad_voice(m, dur + r, bright)
        v *= env_asr(n, a, r)
        p = -0.55 + 1.1 * (i / max(1, len(notes) - 1))
        gl, gr = pan_gain(p)
        out[0] += v * gl
        out[1] += v * gr
    out = filt(out, 'high', 140)
    sub = sine(mtof(notes[0] - 12), n) * env_asr(n, a, r) * 0.12
    out += sub
    return out / max(1, len(notes)) * 1.6


def pluck(m, dur=0.5, bright=0.5, vel=1.0):
    n = n_of(dur)
    t = tvec(n)
    f = mtof(m)
    idx = (1.6 + 2.4 * bright) * np.exp(-t / 0.05)
    mod = np.sin(2 * np.pi * f * 2 * t) * idx
    car = np.sin(2 * np.pi * f * t + mod)
    car += 0.25 * np.sin(2 * np.pi * f * 2 * t + mod * 0.5)
    e = np.minimum(1, t / 0.003) * np.exp(-t / (0.14 + 0.12 * bright))
    return car * e * vel


def bell(m, dur=3.0, vel=1.0, ratio=3.5):
    n = n_of(dur)
    t = tvec(n)
    f = mtof(m)
    idx = 2.2 * np.exp(-t / 0.9)
    car = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * ratio * t))
    car += 0.35 * np.sin(2 * np.pi * f * 2.0 * t) * np.exp(-t / 0.6)
    car += 0.18 * np.sin(2 * np.pi * f * 4.07 * t) * np.exp(-t / 0.25)
    e = np.minimum(1, t / 0.002) * np.exp(-t / 1.15)
    return car * e * vel


def blip(m, dur=0.09, vel=1.0):
    n = n_of(dur)
    t = tvec(n)
    f = mtof(m)
    y = np.sin(2 * np.pi * f * t) + 0.3 * np.sin(2 * np.pi * f * 3 * t) * np.exp(-t / 0.01)
    return y * np.minimum(1, t / 0.002) * np.exp(-t / 0.028) * vel


def glass(m, vel=1.0):
    n = n_of(1.4)
    t = tvec(n)
    f = mtof(m)
    y = np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * f * 2.76 * t) * np.exp(-t / 0.18) + 0.25 * np.sin(2 * np.pi * f * 5.4 * t) * np.exp(-t / 0.07)
    return y * np.minimum(1, t / 0.002) * np.exp(-t / 0.42) * vel


def kick(vel=1.0):
    n = n_of(0.6)
    t = tvec(n)
    f = 46 + 110 * np.exp(-t / 0.035)
    y = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.32)
    click = filt(noise(n), 'high', 2500) * np.exp(-t / 0.004) * 0.25
    return soft(np.tanh(1.4 * (y + click)) * vel, 1.0)


def snare(vel=1.0, clap=True):
    n = n_of(0.5)
    t = tvec(n)
    nz = filt(noise(n), 'bp', (1100, 7000))
    if clap:
        e = np.zeros(n)
        for k, d in enumerate((0, 0.011, 0.023, 0.034)):
            i = n_of(d)
            e[i:] += np.exp(-(t[: n - i]) / (0.006 if k < 3 else 0.16)) * (0.8 if k < 3 else 1.0)
    else:
        e = np.exp(-t / 0.11)
    body = np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.06) * 0.5
    return soft((nz * e + body) * vel * 0.6)


def hat(vel=1.0, open_=False):
    n = n_of(0.35 if open_ else 0.08)
    t = tvec(n)
    y = filt(noise(n), 'high', 7200, 2)
    # 2 毫秒的起音：瞬间起振的噪声在 AAC 里会冲出很高的峰值
    return soft(y * np.exp(-t / (0.11 if open_ else 0.022)) * np.minimum(1, t / 0.002) * vel * 0.42)


def shaker(vel=1.0):
    n = n_of(0.07)
    t = tvec(n)
    y = filt(noise(n), 'bp', (4500, 11000))
    e = np.sin(np.pi * np.minimum(1, t / 0.07)) ** 2
    return y * e * vel * 0.22


def bass_note(m, dur, vel=1.0):
    n = n_of(dur)
    t = tvec(n)
    f = mtof(m)
    y = np.sin(2 * np.pi * f * t) + 0.22 * np.sin(2 * np.pi * 2 * f * t) + 0.08 * saw(f, n)
    y = np.tanh(1.6 * y) / 1.2
    e = np.minimum(1, t / 0.006) * np.minimum(1, np.maximum(0, (dur - t) / 0.04))
    return filt(y * e, 'low', 900) * vel


def impact(vel=1.0, dur=3.0):
    n = n_of(dur)
    t = tvec(n)
    f = 38 + 44 * np.exp(-t / 0.22)
    sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.8) * 0.75
    crack = filt(noise(n), 'low', 3000) * np.exp(-t / 0.09) * 0.5
    air = filt(noise(n), 'bp', (300, 2500)) * np.exp(-t / 0.7) * 0.18
    return soft(np.tanh(1.3 * (sub + crack + air)) * vel, 2.0)


def whoosh(dur=1.3, vel=1.0, up=True):
    """噪声穿过一串带通，中心频率依次扫过（上扬或下坠）"""
    n = n_of(dur)
    t = tvec(n) / dur
    nz = noise(n)
    bands = [(200, 500), (400, 900), (800, 1700), (1500, 3200), (3000, 6500)]
    y = np.zeros(n)
    for i, b in enumerate(bands):
        c = (i + 0.5) / len(bands)
        if not up:
            c = 1 - c
        w = np.exp(-((t - c) / 0.22) ** 2)
        y += filt(nz, 'bp', b) * w
    return y * np.sin(np.pi * np.clip(t, 0, 1)) ** 1.5 * vel * 0.8


def riser(dur=3.0, vel=1.0, m0=50, m1=74):
    n = n_of(dur)
    t = tvec(n)
    k = t / dur
    nz = noise(n)
    lo = filt(nz, 'bp', (400, 1200)) * (1 - k)
    hi = filt(nz, 'bp', (2500, 9000)) * k
    f = mtof(m0 + (m1 - m0) * k ** 1.5)
    tone = saw(f, n) * 0.15 + sine(f * 2, n) * 0.1
    tone = filt(tone, 'low', 3500)
    return (lo + hi + tone) * (k ** 2.2) * vel


def reverse_swell(dur=1.2, vel=1.0):
    n = n_of(dur)
    t = tvec(n)
    y = filt(noise(n), 'bp', (500, 9000)) * np.exp(-(dur - t) / 0.35)
    return y * vel * 0.6


def keyclick(vel=1.0, pitch=0.0):
    n = n_of(0.12)
    t = tvec(n)
    y = filt(noise(n), 'bp', (1800 * 2 ** pitch, 5200 * 2 ** pitch)) * np.exp(-t / 0.008)
    thump = np.sin(2 * np.pi * 180 * t) * np.exp(-t / 0.02) * 0.6
    return soft((y + thump) * vel)


def wood(vel=1.0, m=84):
    n = n_of(0.08)
    t = tvec(n)
    f = mtof(m)
    y = np.sin(2 * np.pi * f * t) * np.exp(-t / 0.012) + 0.4 * filt(noise(n), 'bp', (2000, 6000)) * np.exp(-t / 0.004)
    return soft(y * vel)


def droplet(vel=1.0, m=86):
    n = n_of(0.16)
    t = tvec(n)
    f = mtof(m) * (1 + 1.4 * np.exp(-t / 0.012))
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.045) * vel


def dice(vel=1.0):
    n = n_of(0.75)
    y = np.zeros(n)
    times = np.cumsum(RNG.uniform(0.04, 0.12, 8))
    for k, tt in enumerate(times):
        if tt > 0.68:
            break
        i = n_of(tt)
        m = n_of(0.05)
        tk = tvec(m)
        c = filt(noise(m), 'bp', (1400, 4200)) * np.exp(-tk / 0.006) + np.sin(2 * np.pi * RNG.uniform(700, 1100) * tk) * np.exp(-tk / 0.012) * 0.6
        c = soft(c)
        y[i: i + m] += c[: n - i] * (1 - k * 0.09)
    return soft(y * vel)


def bitblip(k, vel=1.0):
    n = n_of(0.05)
    t = tvec(n)
    f = 1200 if k % 2 == 0 else 1800
    sq = np.sign(np.sin(2 * np.pi * f * t))
    return soft(filt(sq, 'low', 6000) * np.exp(-t / 0.014) * vel * 0.35)


def glitch(vel=1.0):
    n = n_of(0.6)
    t = tvec(n)
    f = 2400 * np.exp(-t / 0.08) + 70
    y = saw(f, n)
    crush = np.round(y * 4) / 4
    return filt(crush, 'low', 5000) * np.exp(-t / 0.2) * vel * 0.5


def mech_open(vel=1.0):
    n = n_of(2.6)
    t = tvec(n)
    thunk = np.sin(2 * np.pi * (55 + 30 * np.exp(-t / 0.05)) * t) * np.exp(-t / 0.35)
    metal = sum(np.sin(2 * np.pi * f * t) * np.exp(-t / d) * a for f, d, a in ((223, 1.2, 0.3), (571, 0.8, 0.18), (913, 0.5, 0.1), (1377, 0.3, 0.06)))
    return (thunk + metal) * vel


# ---------------------------------------------------------------- 和声

CH = {
    'Dm9': ([50, 53, 57, 60, 64], 38),
    'Bbmaj7': ([46, 50, 53, 57, 60], 34),
    'Fmaj7': ([53, 57, 60, 64, 67], 41),
    'Csus2': ([48, 55, 60, 62, 67], 36),
    'Dmadd9': ([50, 57, 62, 64, 65], 38),
    'Gm9': ([55, 58, 62, 65, 69], 43),
    'A7sus4': ([45, 50, 52, 55, 62], 33),
    'A': ([45, 49, 52, 57, 61], 33),
    'C': ([48, 52, 55, 60, 64], 36),
    'Bbmaj9': ([46, 50, 53, 57, 60, 62], 34),
    'FmajA': ([45, 53, 57, 60, 64], 33),
    'Dsus2': ([50, 57, 62, 64, 69], 38),
    'D': ([50, 54, 57, 62, 64, 69], 38),
}
P1 = ['Dm9', 'Bbmaj7', 'Fmaj7', 'Csus2']
P2 = ['Dmadd9', 'Gm9', 'Bbmaj7', 'A7sus4']


# 每段的和弦进行（按段内的第几小节取；比段落长就循环）
SEC_CHORDS = {
    'chat': [None, None, None, 'Dm9', 'Dm9', 'Csus2'],
    'patch': ['Dm9', 'Bbmaj7', 'Fmaj7', 'Csus2'],
    'calc': P2,                                  # 一次乘加：讲解段，安静
    'vit': P1,
    'merge': ['Gm9', 'Bbmaj7', 'Dm9', 'A7sus4', 'A7sus4'],
    'splice': ['Bbmaj7', 'Fmaj7', 'Gm9', 'Csus2', 'A7sus4'],
    'answer': ['Dm9', 'Bbmaj7', 'Fmaj7', 'Csus2'],
    'lens': ['Bbmaj7', 'C', 'Dm9', 'Bbmaj7', 'A7sus4'],
    'reply': P1,  # 打字时只有嗡鸣；按下发送，和弦从那一下进来；推近、裂开时往上走
    'fly': ['Dm9', 'Bbmaj7', 'Csus2'],          # 穿入：一路往上走
    'land': ['Dm9', 'Bbmaj7', 'Fmaj7'],
    'embed': P1, 'layers1': P1,
    'attn': P2,                                 # 注意力：讲解段，安静
    'ffn': ['Gm9', 'Bbmaj7', 'Dm9', 'A7sus4'],  # 前馈
    'sample1': ['Bbmaj7', 'Bbmaj7', 'Gm9', 'Gm9', 'A7sus4', 'A'],
    'loop1': P1,
    'layersK': ['Bbmaj7', 'C', 'Dm9', 'Bbmaj7', 'C', 'A7sus4'],
    'sampleK': ['Gm9', 'Gm9', 'A7sus4', 'A'],
    'loop2': P1,
    'pick': ['Bbmaj7', 'Bbmaj7', 'Gm9', 'Gm9', 'A7sus4', 'A7sus4', 'A', 'A'],  # 选字：理解完之后怎么把词说出来
    'end': ['Bbmaj9', 'Bbmaj9', 'FmajA', 'Gm9', 'Bbmaj9', 'FmajA', 'Gm9', 'Dsus2', 'D', 'D'],  # 答案卡 → 推理页演示 → 落版收尾
}


def bar_info(sections):
    """每一小节：(段名, 段内第几小节, 这段一共几小节)"""
    out = []
    for s_ in sections:
        nb = int(round((s_['t1'] - s_['t0']) / BAR))
        for k in range(nb):
            out.append((s_['name'], k, nb))
    return out


def chord_plan(sections):
    plan = []
    for name, k, nb in bar_info(sections):
        seq = SEC_CHORDS.get(name, P1)
        plan.append(seq[min(k, len(seq) - 1)] if name in ('end', 'chat', 'fly', 'merge', 'splice', 'lens') else seq[k % len(seq)])
    return plan


# ---------------------------------------------------------------- 编曲

def section_at(sections, t):
    for s in sections:
        if s['t0'] <= t < s['t1']:
            return s
    return sections[-1]


def compose(ev, out_wav, stems_dir=None, ceiling=-2.0):
    sections = ev['score']['sections']
    dur = ev['duration']
    plan = chord_plan(sections)
    binfo = bar_info(sections)
    nbars = len(plan)
    events = ev['events']
    sec = lambda t: section_at(sections, t)['name']
    energy = lambda t: section_at(sections, t)['energy']

    pad = Bus(dur)
    arp = Bus(dur)
    bass = Bus(dur)
    drums = Bus(dur)
    fx = Bus(dur)      # 音效（干声多一点）
    bells = Bus(dur)   # 铃声、钟（混响多一点）
    kicks = []          # 侧链压缩用

    # 1. 冷开场的嗡鸣：D 的根音 + 五度，缓慢起伏，加一层空气声
    n = n_of(15.0)
    t = tvec(n)
    hum = (sine(mtof(38), n) * 0.35 + sine(mtof(50) * (1 + 0.002 * np.sin(2 * np.pi * 0.07 * t)), n) * 0.3 + sine(mtof(57) * 1.001, n) * 0.18)
    hum += 0.12 * filt(saw(mtof(38), n), 'low', 400)
    air = filt(noise(n), 'bp', (250, 900)) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.11 * t)) * 0.12
    shimmer = sum(sine(mtof(m), n) * 0.04 * (0.5 + 0.5 * np.sin(2 * np.pi * r * t + m)) for m, r in ((81, 0.09), (86, 0.13), (88, 0.07)))
    send_t = next((e_['t'] for e_ in events if e_['type'] == 'send'), 7.5)
    climbs = [(e_['t'], e_['t'] + e_.get('d', 0)) for e_ in events if e_['type'] == 'climb']
    QUIET = ('patch', 'calc', 'splice')
    e = np.clip(t / 3.0, 0, 1) ** 2 * (1 + 0.5 * np.clip((t - (send_t - 2.5)) / 2.5, 0, 1) ** 2)
    e *= np.clip(1 - (t - send_t) / 2.2, 0, 1) ** 1.5
    pad.add(0, (hum + air * 1.6 + shimmer * 1.5) * e * 0.4)

    # 2. 铺底和弦：每小节一个，按段落能量调亮度和音量
    for b, c in enumerate(plan):
        if not c:
            continue
        t0 = b * BAR
        en = energy(t0 + 0.1) if not (sec(t0 + 0.1) == "chat" and t0 + BAR > send_t) else 0.5  # 发送以后：和弦亮起来
        notes, root = CH[c]
        nm = sec(t0 + 0.1)
        bright = 0.25 + 0.6 * en if nm not in QUIET else 0.22
        vol = 0.32 + 0.35 * en
        if nm == 'end':
            vol = 0.55
        hold = BAR + 0.3 + (BAR * 2 if b == nbars - 1 else 0)
        start = t0 - 0.3
        if nm == 'chat' and t0 < send_t < t0 + BAR:
            # 发送落在这一小节中间：和弦从发送那一下才进来
            start = send_t - 0.05
            hold = t0 + BAR + 0.3 - start
        pad.add(start, pad_chord(notes, hold, bright), vol)

    # 3. 琶音：和弦音在两个八度之间上下走；能量高时 16 分音符，低时 8 分音符
    for b, c in enumerate(plan):
        if not c:
            continue
        t0 = b * BAR
        nm = sec(t0 + 0.1)
        en = energy(t0 + 0.1) if not (nm == "chat" and t0 + BAR > send_t) else 0.5
        if nm == 'end' or (nm == 'chat' and t0 + BAR <= send_t + 0.5):
            continue
        notes, _ = CH[c]
        tones = sorted(set([m + 12 for m in notes] + [m + 24 for m in notes[:3]]))
        sixteenth = nm in ('fly', 'reply')   # v2：视觉编码器那段讲解变多，琶音放慢
        step = BEAT / 4 if sixteenth else BEAT / 2
        steps = int(round(BAR / step))
        pattern = list(range(len(tones))) + list(range(len(tones) - 2, 0, -1))
        for k in range(steps):
            if nm == "chat" and t0 + k * step < send_t + 0.3:
                continue
            # 讲解段：琶音稀一点，留出思考的空间
            if nm in ('calc', 'patch') and k % 4 != 0:
                continue
            if nm in ('splice', 'merge') and k % 2 == 1:
                continue
            if nm == 'answer' and not any(a <= t0 + k * step < b for a, b in climbs) and k % 4 != 0:
                continue
            m = tones[pattern[(k + b * 3) % len(pattern)]]
            vel = (0.55 + 0.45 * (k % 4 == 0)) * (0.5 + 0.5 * en)
            if nm == 'chat':
                vel *= 0.55 + 0.45 * k / steps  # 推近气泡时一点点涨上去
            br = 0.35 + 0.5 * en if nm not in QUIET else 0.25
            arp.add(t0 + k * step, pluck(m, 0.6, br, vel), 0.42, pan=0.35 * math.sin(k * 1.3 + b))

    # 4. 贝斯与鼓
    def drum_bar(b, kind):
        t0 = b * BAR
        if kind == 'none':
            return
        if kind == 'heart':
            drums.add(t0, kick(0.8))
            drums.add(t0 + BEAT * 0.5, kick(0.4))
            kicks.extend([t0, t0 + BEAT * 0.5])
            return
        if kind == 'half':
            drums.add(t0, kick(0.7)); kicks.append(t0)
            drums.add(t0 + 2 * BEAT, snare(0.35, clap=False), pan=0.1)
            for k in range(8):
                drums.add(t0 + k * BEAT / 2, hat(0.35 + 0.25 * (k % 2)), pan=0.25)
            return
        four = kind in ('four', 'full')
        for k in range(4):
            if four or k in (0, 2):
                drums.add(t0 + k * BEAT, kick(1.0 if k == 0 else 0.85)); kicks.append(t0 + k * BEAT)
            if k in (1, 3):
                drums.add(t0 + k * BEAT, snare(0.7 if kind == 'full' else 0.5), pan=-0.05)
        if kind in ('groove', 'four', 'full'):
            div = 4 if kind in ('four', 'full') else 2
            for k in range(4 * div):
                v = 0.5 + 0.35 * (k % div == div // 2)
                drums.add(t0 + k * BEAT / div, hat(v * (0.9 if kind == 'full' else 0.7)), pan=0.3)
            drums.add(t0 + 3.5 * BEAT, hat(0.55, open_=True), pan=0.3)
        if kind == 'full':
            for k in range(16):
                drums.add(t0 + k * BEAT / 4 + 0.01, shaker(0.6 + 0.3 * (k % 2)), pan=-0.35)

    for b, c in enumerate(plan):
        t0 = b * BAR
        nm = sec(t0 + 0.1)
        kind = {'chat': 'none', 'fly': 'heart', 'patch': 'none', 'calc': 'none', 'vit': 'half', 'merge': 'heart', 'splice': 'none',
                'answer': 'heart', 'lens': 'half', 'reply': 'groove', 'end': 'none'}[nm]
        sname, k_in, nb_in = binfo[b]
        if nm == 'calc' and k_in % 4 == 2 and k_in < nb_in - 2:
            kind = 'heart'  # 讲解段偶尔垫一下心跳，别完全停住
        if nm == 'splice' and k_in % 2 == 1:
            kind = 'heart'
        if nm == 'vit' and k_in == 0:
            kind = 'heart'
        if nm == 'answer' and any(a < t0 + BAR and t0 < b2 for a, b2 in climbs):
            kind = 'groove'  # 逐层看「3」的注意力：律动进来
        if nm == 'chat' and k_in == nb_in - 1:
            kind = 'heart'  # 图和词元飞起来：心跳进来
        drum_bar(b, kind)
        # 进入下一段前一小节：军鼓滚奏
        if (sname in ('merge',) and k_in == nb_in - 1) or (sname == 'lens' and k_in == nb_in - 1):
            for k in range(16):
                drums.add(t0 + k * BEAT / 4, snare(0.15 + 0.5 * k / 16, clap=False), pan=0.05)
        if c and nm not in ('chat', 'end'):
            _, root = CH[c]
            if kind in ('groove', 'four', 'full'):
                for k in range(8):
                    bass.add(t0 + k * BEAT / 2, bass_note(root + (12 if k % 4 == 3 else 0), BEAT / 2 * 0.9, 0.9 if k % 2 == 0 else 0.7))
            else:
                bass.add(t0, bass_note(root, BAR * 0.95, 0.45 if nm in QUIET else 0.65))
    # 片尾最后一个低音
    bass.add((nbars - 1) * BAR, bass_note(38, 4.5, 0.6))

    # 5. 画面事件 → 音效 / 铃声
    penta = [62, 65, 67, 69, 72, 74, 77, 79, 81, 84, 86, 89]
    for e in events:
        t, ty, k = e['t'], e['type'], e.get('k', 1.0)
        b = int(t // BAR)
        c = plan[min(b, len(plan) - 1)] or 'Dm9'
        notes, root = CH[c]
        if ty == 'key':
            # 敲拼音：很轻的键盘声，左右略有变化
            fx.add(t, keyclick(0.22 + 0.06 * ((e.get('i', 0) * 5) % 3), pitch=0.05 * (e.get('i', 0) % 4)), pan=-0.25 + 0.05 * (e.get('i', 0) % 9))
        elif ty == 'type':
            fx.add(t, keyclick(0.55, pitch=0.1 * (e.get('i', 0) % 3)), pan=-0.2 + 0.08 * e.get('i', 0))
            bells.add(t, blip(74 + [0, 3, 5, 7, 10, 12][e.get('i', 0) % 6], 0.25, 0.25))
        elif ty == 'tick':
            fx.add(t, wood(0.18 * k + 0.05, 88 + (e.get('i', 0) % 5) * 2), pan=-0.6 + 1.2 * ((e.get('i', 0) % 40) / 40))
        elif ty == 'blip':
            bells.add(t, droplet(0.35, 84 + (e.get('i', 0) * 5) % 12), pan=-0.5 + (e.get('i', 0) % 39) / 39)
        elif ty == 'layer':
            L = e.get('L', 0)
            top = e.get('n', 28) - 1   # ViT 24 层、语言模型 28 层
            m = penta[min(len(penta) - 1, int(L * (len(penta) - 1) / top))]
            bells.add(t, blip(m, 0.12, 0.3 + 0.25 * k), pan=-0.4 + 0.8 * L / top)
        elif ty == 'reveal':
            for i, m in enumerate(sorted(notes)[-3:]):
                bells.add(t + i * 0.03, bell(m + 12, 3.5, 0.32 * k), pan=-0.3 + 0.3 * i)
        elif ty == 'step':
            bells.add(t, glass(81 + (int(t * 7) % 3) * 2, 0.22 * k), pan=0.2)
        elif ty == 'hit':
            fx.add(t, impact(0.9 * k))
            if k >= 0.7:
                fx.add(t - 1.0, reverse_swell(1.0, 0.5 * k))
        elif ty == 'whoosh':
            fx.add(t - 0.3, whoosh(1.6, 0.6 * k))
            fx.add(t - 2.5, riser(2.5, 0.35))
        elif ty == 'send':
            fx.add(t - 0.12, whoosh(0.9, 0.45, up=True))
            fx.add(t, impact(0.35, 2.0))
            bells.add(t, bell(74, 2.6, 0.38))
            bells.add(t + 0.06, bell(81, 2.2, 0.2), pan=0.2)
        elif ty == 'rise':
            fx.add(t, riser(e.get('d', 2.0), 0.45))
        elif ty == 'open':
            fx.add(t, mech_open(0.6))
            fx.add(t + 0.2, whoosh(2.0, 0.35, up=False))
        elif ty == 'dice':
            fx.add(t, dice(0.55), pan=0.15)
            fx.add(t - 2.0, riser(2.2, 0.25, 55, 70))
        elif ty == 'emit':
            g = e.get('g', 0)
            ct = sorted(set(m % 12 for m in notes))
            # 高概率的词落在和弦音上；没选概率最高的那个（rank > 0）落在色彩音（九音 / 六音）上
            rank = e.get('rank', 0)
            deg = ct[(g * 2) % len(ct)] if rank == 0 else (root + 2) % 12
            m = 72 + deg
            if m < 74:
                m += 12
            vel = 0.45 * k
            bells.add(t, bell(m, 3.0, vel), pan=-0.35 + 0.7 * ((g * 7) % 10) / 10)
            if k >= 1.1:
                fx.add(t, impact(0.5 * k, 2.5))
                bells.add(t, bell(m - 12, 4.0, 0.35))
        elif ty == 'end':
            for i, m in enumerate([62, 66, 69, 74]):
                bells.add(t + i * 0.06, bell(m, 4.0, 0.3), pan=-0.3 + 0.2 * i)
        elif ty == 'bit':
            fx.add(t, bitblip(e.get('k', 0), 0.5), pan=-0.5 + e.get('k', 0) / 15)
        elif ty == 'flip':
            fx.add(t, glitch(0.8))

    # 片尾：最后的大和弦 + 钟
    for i, m in enumerate([62, 69, 74, 78, 81]):
        bells.add((nbars - 1) * BAR + 0.05 * i, bell(m, 6.0, 0.22), pan=-0.4 + 0.2 * i)

    # 6. 侧链：底鼓响的时候，铺底 / 贝斯 / 琶音让一让
    duck = np.ones(pad.n)
    tt = tvec(pad.n)
    for kt in kicks:
        i = n_of(kt)
        m = min(pad.n - i, n_of(0.5))
        if m <= 0:
            continue
        duck[i: i + m] = np.minimum(duck[i: i + m], 1 - 0.42 * np.exp(-tt[:m] / 0.11))
    for bus in (pad, bass, arp):
        bus.x *= duck

    # 段落音量：冷开场、拆层段落更安静，28 层和最后的自回归最满
    LEVEL = {'chat': 0.7, 'fly': 0.95, 'patch': 0.78, 'calc': 0.72, 'vit': 0.84, 'merge': 0.86, 'splice': 0.78, 'answer': 0.86,
             'lens': 0.9, 'reply': 1.0, 'end': 0.7}
    lvl = np.ones(pad.n)
    for s_ in sections:
        lvl[n_of(s_['t0']): n_of(s_['t1'])] = LEVEL.get(s_['name'], 1.0)
    # 聊天段：打字时轻；发送以后逐渐推到飞行段的音量
    sc = next((s_ for s_ in sections if s_['name'] == 'chat'), None)
    if sc:
        a, b = n_of(send_t), n_of(sc['t1'])
        lvl[a:b] = np.linspace(LEVEL['chat'], LEVEL['fly'], b - a)
    lvl = ss.filtfilt([1 - math.exp(-1 / (SR * 0.25))], [1, -math.exp(-1 / (SR * 0.25))], lvl)
    for bus in (pad, bass, arp, drums):
        bus.x *= lvl

    # 7. 混音：混响、延迟
    ir = make_ir(3.4)
    ir_short = make_ir(1.4, dark=7000)
    arp_d = arp.x + pingpong(arp.x, BEAT * 0.75, 0.36)
    mix = (
        pad.x * 0.85 + reverb(pad.x, ir, 0.45)
        + arp_d * 0.7 + reverb(arp_d, ir, 0.38)
        + bass.x * 0.62
        + drums.x * 0.75 + reverb(drums.x, ir_short, 0.12)
        + bells.x * 0.95 + reverb(bells.x, ir, 0.65)
        + fx.x * 0.75 + reverb(fx.x, ir, 0.3)
    )
    mix = filt(mix, 'high', 30)
    mix = shelf(mix, 110, -3.0, high=False)
    mix = shelf(mix, 4500, 3.5, high=True)
    mix = filt(mix, 'low', 17000, 4)
    n_end = n_of(dur)
    mix = mix[:, : n_end]
    # 淡入淡出
    t = tvec(n_end)
    mix *= np.clip(t / 0.4, 0, 1)
    mix *= np.clip((dur - t) / 6.0, 0, 1) ** 1.5   # 片尾落版慢慢淡出

    if stems_dir:
        for name, bus in (('pad', pad), ('arp', arp), ('bass', bass), ('drums', drums), ('fx', fx), ('bells', bells)):
            wavfile.write(f'{stems_dir}/stem_{name}.wav', SR, (np.clip(bus.x[:, :n_end].T, -1, 1) * 32767).astype(np.int16))
    out = master(mix, ceiling=ceiling)
    wavfile.write(out_wav, SR, out.T.astype(np.float32))
    print(f'wrote {out_wav}: {out.shape[1] / SR:.2f}s, {lufs(out):.2f} LUFS, true peak {true_peak_db(out):.2f} dBTP')


def B(bar, beat=0):
    return bar * BAR + beat * BEAT


# ---------------------------------------------------------------- 母带

def k_weight(x):
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
    a1 = [1.0, -1.69065929318241, 0.73248077421585]
    b2 = [1.0, -2.0, 1.0]
    a2 = [1.0, -1.99004745483398, 0.99007225036621]
    return ss.lfilter(b2, a2, ss.lfilter(b1, a1, x, axis=-1), axis=-1)


def lufs(x):
    y = k_weight(x)
    blk, hop = n_of(0.4), n_of(0.1)
    zs = []
    for i in range(0, y.shape[1] - blk, hop):
        zs.append(np.sum(np.mean(y[:, i: i + blk] ** 2, axis=1)))
    zs = np.array(zs)
    l = -0.691 + 10 * np.log10(zs + 1e-20)
    zs = zs[l > -70]
    rel = -0.691 + 10 * np.log10(np.mean(zs)) - 10
    l = -0.691 + 10 * np.log10(zs + 1e-20)
    return -0.691 + 10 * np.log10(np.mean(zs[l > rel]))


def true_peak_db(x):
    up = ss.resample_poly(x, 4, 1, axis=-1)
    return 20 * np.log10(np.max(np.abs(up)) + 1e-12)


def limit(x, ceiling_db=-2.0):
    c = 10 ** (ceiling_db / 20)
    up = ss.resample_poly(x, 4, 1, axis=-1)
    pk = np.max(np.abs(up), axis=0)
    pk = pk[: x.shape[1] * 4].reshape(-1, 4).max(axis=1)
    g = np.minimum(1.0, c / np.maximum(pk, 1e-9))
    g = minimum_filter1d(g, size=n_of(0.004) * 2 + 1)          # 提前 4 毫秒压下去
    a = math.exp(-1 / (SR * 0.06))                               # 60 毫秒释放
    g2 = ss.lfilter([1 - a], [1, -a], g - 1.0) + 1.0
    g = np.minimum(g, g2)
    g = minimum_filter1d(g, size=n_of(0.002) * 2 + 1)
    return x * g


def master(mix, target=-14.0, ceiling=-2.0):
    # 轻微的软削波当“胶水”
    mix = np.tanh(mix * 0.9) / 0.9
    for _ in range(3):
        mix *= 10 ** ((target - lufs(mix)) / 20)
        mix = limit(mix, ceiling)
    return mix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default='tools/video/mm/events.json')
    ap.add_argument('--out', default='/mnt/d/cjc/videos/multimodal/score.wav')
    ap.add_argument('--stems', default=None)
    # 真峰值上限（dBTP）。AAC 解码后个别密集的瞬态会再冲高 2 dB 左右：英文版收尾蒙太奇那里用 −3.5 才压得住
    ap.add_argument('--ceiling', type=float, default=-2.0)
    a = ap.parse_args()
    ev = json.load(open(a.events))
    compose(ev, a.out, a.stems, a.ceiling)


if __name__ == '__main__':
    main()
