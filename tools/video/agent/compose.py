#!/usr/bin/env python3
"""智能体视频的原创配乐：全部用 numpy / scipy 现场合成，不用任何采样或下载的音乐。

音色（铺底、拨弦、贝斯、鼓、铃、音效）和母带处理直接用推理视频的 ../compose.py（只引用，不改它），
这里只写编曲：段落表和画面事件来自 render.mjs events 导出的 events.json（96 BPM，一小节 2.5 秒）。

  开场 chat   打字时只有嗡鸣和键盘声；按发送，和弦从那一下进来；片名一记重拍
  动手 work   轻律动：半速的鼓、八分音符琶音；每条命令一声轻响，报错是两个往下走的短音
  只会写字 loop / 越滚越长 ctx   讲解段：没有鼓，琶音很稀，只留铺底和数据包的玻璃声
  逐词生成 tok  更安静；每个词元一个很轻的音，关键决策处一声玻璃，「3」那一下和弦从 A7sus4 解决到 D
  自己改错 fix  律动回来，最满的一段；跑通时一组明亮的铃
  片尾 end    和弦落到 D 大调，混响拖尾，淡出

    /mnt/d/cjc/venvs/blackbox/bin/python tools/video/agent/compose.py --events tools/video/agent/events.json --out /mnt/d/cjc/videos/agent/score.wav
"""
import argparse
import json
import math
import pathlib
import sys

import numpy as np
import scipy.signal as ss
from scipy.io import wavfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import compose as C  # noqa: E402  推理视频的合成器（音色、混响、母带）

SR, BEAT, BAR = C.SR, C.BEAT, C.BAR
n_of, tvec, sine, saw, noise, filt = C.n_of, C.tvec, C.sine, C.saw, C.noise, C.filt
CH = C.CH
P1 = ['Dm9', 'Bbmaj7', 'Fmaj7', 'Csus2']
P2 = ['Dmadd9', 'Gm9', 'Bbmaj7', 'A7sus4']

SEC_CHORDS = {
    'chat': [None, None, None, 'Dm9', 'Bbmaj7', 'Csus2'],
    'work': P1,
    'loop': P2,
    'ctx': ['Gm9', 'Bbmaj7', 'Dm9', 'A7sus4'],
    # 逐词生成：python 那一步停在 A7sus4 上，「3」落下时解决到 Dmadd9
    'tok': ['Gm9', 'Gm9', 'Bbmaj7', 'Bbmaj7', 'A7sus4', 'Dmadd9', 'Dmadd9', 'Bbmaj7', 'Gm9', 'A7sus4', 'Dm9', 'Csus2'],
    'fix': P1,
    'end': ['Bbmaj9', 'Bbmaj9', 'FmajA', 'Gm9', 'Bbmaj9', 'FmajA', 'Gm9', 'Csus2', 'Bbmaj9', 'FmajA', 'Gm9', 'Dsus2', 'Bbmaj9', 'FmajA', 'Gm9', 'Dsus2', 'D', 'D'],
}
LOOPED = {'work', 'loop', 'ctx', 'fix'}
LEVEL = {'chat': 0.7, 'work': 0.95, 'loop': 0.72, 'ctx': 0.7, 'tok': 0.72, 'fix': 1.0, 'end': 0.78}


def bar_info(sections):
    out = []
    for s_ in sections:
        nb = int(round((s_['t1'] - s_['t0']) / BAR))
        for k in range(nb):
            out.append((s_['name'], k, nb))
    return out


def err_sound(vel=1.0):
    """报错：两个往下走的短音（小二度），不刺耳"""
    a = C.blip(70, 0.14, 0.5 * vel)
    b = C.blip(69, 0.22, 0.45 * vel)
    out = np.zeros(max(len(a), n_of(0.1) + len(b)))
    out[: len(a)] += a
    out[n_of(0.1): n_of(0.1) + len(b)] += b
    return filt(out, 'low', 3200)


def compose(ev, out_wav, ceiling=-2.0):
    sections = ev['score']['sections']
    dur = ev['duration']
    binfo = bar_info(sections)
    plan = []
    for name, k, nb in binfo:
        seq = SEC_CHORDS[name]
        plan.append(seq[k % len(seq)] if name in LOOPED else seq[min(k, len(seq) - 1)])
    nbars = len(plan)
    events = ev['events']
    sec_of = lambda t: C.section_at(sections, t)
    send_t = next((e['t'] for e in events if e['type'] == 'send'), 8.75)

    pad, arp, bass, drums, fx, bells = (C.Bus(dur) for _ in range(6))
    kicks = []

    # 1. 开场的嗡鸣：D 的根音 + 五度，缓慢起伏，加一层空气声；按发送以后退下去
    n = n_of(12.0)
    t = tvec(n)
    hum = sine(C.mtof(38), n) * 0.35 + sine(C.mtof(50) * (1 + 0.002 * np.sin(2 * np.pi * 0.07 * t)), n) * 0.3 + sine(C.mtof(57) * 1.001, n) * 0.18
    hum += 0.12 * filt(saw(C.mtof(38), n), 'low', 400)
    air = filt(noise(n), 'bp', (250, 900)) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.11 * t)) * 0.12
    e = np.clip(t / 3.0, 0, 1) ** 2 * np.clip(1 - (t - send_t) / 2.2, 0, 1) ** 1.5
    pad.add(0, (hum + air * 1.6) * e * 0.4)

    # 2. 铺底和弦
    for b, c in enumerate(plan):
        if not c:
            continue
        t0 = b * BAR
        s_ = sec_of(t0 + 0.1)
        nm, en = s_['name'], s_['energy']
        bright = 0.25 + 0.6 * en if nm not in ('loop', 'ctx', 'tok') else 0.22
        vol = 0.34 + 0.35 * en if nm != 'end' else 0.55
        hold = BAR + 0.3 + (BAR * 2 if b == nbars - 1 else 0)
        start = t0 - 0.3
        if nm == 'chat' and t0 < send_t < t0 + BAR:
            start = send_t - 0.05
            hold = t0 + BAR + 0.3 - start
        pad.add(start, C.pad_chord(CH[c][0], hold, bright), vol)

    # 3. 琶音：动手 / 改错段八分 / 十六分音符；讲解段很稀
    for b, c in enumerate(plan):
        if not c:
            continue
        t0 = b * BAR
        s_ = sec_of(t0 + 0.1)
        nm, en = s_['name'], s_['energy']
        if nm in ('end',) or (nm == 'chat' and t0 + BAR <= send_t + 0.5):
            continue
        notes = CH[c][0]
        tones = sorted(set([m + 12 for m in notes] + [m + 24 for m in notes[:3]]))
        sixteenth = nm == 'fix' and binfo[b][1] >= 2
        step = BEAT / 4 if sixteenth else BEAT / 2
        steps = int(round(BAR / step))
        pattern = list(range(len(tones))) + list(range(len(tones) - 2, 0, -1))
        for k in range(steps):
            tt = t0 + k * step
            if nm == 'chat' and tt < send_t + 0.3:
                continue
            if nm in ('loop', 'ctx') and k % 4 != 0:
                continue
            if nm == 'tok' and k % 4 != 0:
                continue
            m = tones[pattern[(k + b * 3) % len(pattern)]]
            vel = (0.55 + 0.45 * (k % 4 == 0)) * (0.5 + 0.5 * en)
            br = 0.35 + 0.5 * en if nm not in ('loop', 'ctx', 'tok') else 0.25
            arp.add(tt, C.pluck(m, 0.6, br, vel), 0.42 if nm not in ('tok',) else 0.3, pan=0.35 * math.sin(k * 1.3 + b))

    # 4. 鼓和贝斯
    def drum_bar(b, kind, vel=1.0):
        t0 = b * BAR
        if kind == 'none':
            return
        if kind == 'heart':
            drums.add(t0, C.kick(0.75 * vel)); drums.add(t0 + BEAT * 0.5, C.kick(0.38 * vel))
            kicks.extend([t0, t0 + BEAT * 0.5])
            return
        if kind == 'half':
            drums.add(t0, C.kick(0.75 * vel)); kicks.append(t0)
            drums.add(t0 + 2 * BEAT, C.snare(0.38 * vel, clap=False), pan=0.1)
            for k in range(8):
                drums.add(t0 + k * BEAT / 2, C.hat((0.35 + 0.25 * (k % 2)) * vel), pan=0.25)
            return
        four = kind in ('four', 'full')
        for k in range(4):
            if four or k in (0, 2):
                drums.add(t0 + k * BEAT, C.kick((1.0 if k == 0 else 0.85) * vel)); kicks.append(t0 + k * BEAT)
            if k in (1, 3):
                drums.add(t0 + k * BEAT, C.snare((0.7 if kind == 'full' else 0.5) * vel), pan=-0.05)
        div = 4 if four else 2
        for k in range(4 * div):
            v = 0.5 + 0.35 * (k % div == div // 2)
            drums.add(t0 + k * BEAT / div, C.hat(v * (0.9 if kind == 'full' else 0.7) * vel), pan=0.3)
        drums.add(t0 + 3.5 * BEAT, C.hat(0.55 * vel, open_=True), pan=0.3)
        if kind == 'full':
            for k in range(16):
                drums.add(t0 + k * BEAT / 4 + 0.01, C.shaker(0.6 + 0.3 * (k % 2)), pan=-0.35)

    for b, c in enumerate(plan):
        t0 = b * BAR
        nm, k_in, nb_in = binfo[b]
        kind = 'none'
        if nm == 'chat' and k_in == nb_in - 1:
            kind = 'heart'
        elif nm == 'work':
            kind = 'half' if k_in < 2 else 'groove'
        elif nm == 'tok' and k_in in (3, 4):
            kind = 'heart'   # python 那一步：心跳
        elif nm == 'fix':
            kind = 'groove' if k_in < 2 else 'four' if k_in < 8 else 'full'
        drum_bar(b, kind, 0.85 if nm == 'tok' else 1.0)
        if nm == 'fix' and k_in == nb_in - 1:
            for k in range(16):
                drums.add(t0 + k * BEAT / 4, C.snare(0.15 + 0.5 * k / 16, clap=False), pan=0.05)
        if c and nm not in ('chat', 'end'):
            root = CH[c][1]
            if kind in ('groove', 'four', 'full'):
                for k in range(8):
                    bass.add(t0 + k * BEAT / 2, C.bass_note(root + (12 if k % 4 == 3 else 0), BEAT / 2 * 0.9, 0.9 if k % 2 == 0 else 0.7))
            else:
                bass.add(t0, C.bass_note(root, BAR * 0.95, 0.45 if nm in ('loop', 'ctx', 'tok') else 0.6))
    bass.add((nbars - 1) * BAR, C.bass_note(38, 4.5, 0.6))

    # 5. 画面事件 → 音效 / 铃声
    penta = [62, 65, 67, 69, 72, 74, 77, 79, 81, 84, 86, 89]
    for e in events:
        t, ty, k = e['t'], e['type'], e.get('k', 1.0)
        b = min(int(t // BAR), nbars - 1)
        c = plan[b] or 'Dm9'
        notes, root = CH[c]
        i = e.get('i', 0) or e.get('j', 0) or 0
        if ty == 'key':
            fx.add(t, C.keyclick(0.22 + 0.06 * ((i * 5) % 3), pitch=0.05 * (i % 4)), pan=-0.25 + 0.05 * (i % 9))
        elif ty == 'type':
            fx.add(t, C.keyclick(0.5, pitch=0.1 * (i % 3)), pan=-0.2 + 0.04 * i)
            bells.add(t, C.blip(74 + [0, 3, 5, 7, 10, 12][i % 6], 0.25, 0.22))
        elif ty == 'send':
            fx.add(t - 0.12, C.whoosh(0.9, 0.45, up=True))
            fx.add(t, C.impact(0.35, 2.0))
            bells.add(t, C.bell(74, 2.6, 0.38))
            bells.add(t + 0.06, C.bell(81, 2.2, 0.2), pan=0.2)
        elif ty == 'whoosh':
            fx.add(t - 0.3, C.whoosh(1.6, 0.6 * k))
            fx.add(t - 2.0, C.riser(2.0, 0.3))
        elif ty == 'hit':
            fx.add(t, C.impact(0.9 * k))
            if k >= 0.7:
                fx.add(t - 1.0, C.reverse_swell(1.0, 0.5 * k))
        elif ty == 'tick':
            fx.add(t, C.wood(0.18 * k + 0.06, 88 + (i % 5) * 2), pan=-0.3 + 0.6 * ((i * 7) % 10) / 10)
        elif ty == 'cmd':
            fx.add(t, C.keyclick(0.45, pitch=0.15))
            bells.add(t + 0.02, C.blip(79, 0.1, 0.22), pan=0.15)
        elif ty == 'err':
            fx.add(t, err_sound(0.8 * k), pan=0.1)
        elif ty == 'ok':
            for j, m in enumerate([74, 78, 81, 86]):
                bells.add(t + j * 0.05, C.bell(m, 3.0, 0.3 * k), pan=-0.3 + 0.2 * j)
        elif ty == 'packet':
            bells.add(t, C.glass(81 + (int(t * 7) % 3) * 2, 0.24 * k), pan=0.2)
        elif ty == 'lap':
            fx.add(t, C.wood(0.3, 84))
            bells.add(t, C.bell(74, 2.0, 0.22))
        elif ty == 'seg':
            bells.add(t, C.droplet(0.4, 84 + i * 3), pan=-0.3 + 0.3 * i)
        elif ty == 'row':
            bells.add(t, C.blip(penta[min(len(penta) - 1, 2 + i)], 0.12, 0.32), pan=-0.4 + 0.1 * i)
        elif ty == 'step':
            bells.add(t, C.glass(81 + (int(t * 7) % 3) * 2, 0.22 * k), pan=0.2)
        elif ty == 'tok':
            ct = sorted(set(m % 12 for m in notes))
            m = 84 + ct[(i * 2) % len(ct)]
            bells.add(t, C.blip(m if m < 96 else m - 12, 0.07, 0.14), pan=-0.35 + 0.7 * ((i * 7) % 10) / 10)
        elif ty == 'dec':
            bells.add(t, C.glass(83, 0.3), pan=0.1)
            bells.add(t, C.bell(74, 2.2, 0.18))
        elif ty == 'climax':
            fx.add(t, C.impact(0.45, 2.5))
            for j, m in enumerate([62, 69, 74, 78]):
                bells.add(t + j * 0.04, C.bell(m + 12, 3.5, 0.3), pan=-0.3 + 0.2 * j)
        elif ty == 'end':
            for j, m in enumerate([62, 66, 69, 74]):
                bells.add(t + j * 0.06, C.bell(m, 4.0, 0.3), pan=-0.3 + 0.2 * j)

    for j, m in enumerate([62, 69, 74, 78, 81]):
        bells.add((nbars - 1) * BAR + 0.05 * j, C.bell(m, 6.0, 0.22), pan=-0.4 + 0.2 * j)

    # 6. 侧链
    duck = np.ones(pad.n)
    tt = tvec(pad.n)
    for kt in kicks:
        i0 = n_of(kt)
        mm = min(pad.n - i0, n_of(0.5))
        if mm > 0:
            duck[i0: i0 + mm] = np.minimum(duck[i0: i0 + mm], 1 - 0.42 * np.exp(-tt[:mm] / 0.11))
    for bus in (pad, bass, arp):
        bus.x *= duck

    # 段落音量：讲解段更安静
    lvl = np.ones(pad.n)
    for s_ in sections:
        lvl[n_of(s_['t0']): n_of(s_['t1'])] = LEVEL.get(s_['name'], 1.0)
    lvl = ss.filtfilt([1 - math.exp(-1 / (SR * 0.25))], [1, -math.exp(-1 / (SR * 0.25))], lvl)
    for bus in (pad, bass, arp, drums):
        bus.x *= lvl

    # 7. 混音
    ir = C.make_ir(3.4)
    ir_short = C.make_ir(1.4, dark=7000)
    arp_d = arp.x + C.pingpong(arp.x, BEAT * 0.75, 0.36)
    mix = (
        pad.x * 0.85 + C.reverb(pad.x, ir, 0.45)
        + arp_d * 0.7 + C.reverb(arp_d, ir, 0.38)
        + bass.x * 0.62
        + drums.x * 0.72 + C.reverb(drums.x, ir_short, 0.12)
        + bells.x * 0.95 + C.reverb(bells.x, ir, 0.65)
        + fx.x * 0.75 + C.reverb(fx.x, ir, 0.3)
    )
    mix = filt(mix, 'high', 30)
    mix = C.shelf(mix, 110, -3.0, high=False)
    mix = C.shelf(mix, 4500, 3.5, high=True)
    mix = filt(mix, 'low', 17000, 4)
    n_end = n_of(dur)
    mix = mix[:, :n_end]
    t = tvec(n_end)
    mix *= np.clip(t / 0.4, 0, 1)
    mix *= np.clip((dur - t) / 5.0, 0, 1) ** 1.5
    out = C.master(mix, ceiling=ceiling)
    wavfile.write(out_wav, SR, out.T.astype(np.float32))
    print(f'wrote {out_wav}: {out.shape[1] / SR:.2f}s, {C.lufs(out):.2f} LUFS, true peak {C.true_peak_db(out):.2f} dBTP')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default='tools/video/agent/events.json')
    ap.add_argument('--out', default='/mnt/d/cjc/videos/agent/score.wav')
    ap.add_argument('--ceiling', type=float, default=-2.0)
    a = ap.parse_args()
    compose(json.load(open(a.events)), a.out, a.ceiling)


if __name__ == '__main__':
    main()
