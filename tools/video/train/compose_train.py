#!/usr/bin/env python3
"""训练视频的原创配乐：和推理视频一样全部用 numpy / scipy 现场合成（不用任何采样或下载的音乐），
音色和母带处理直接借用 ../compose.py 里的函数（只引用、不改它）。

速度 96 BPM，段落和分镜（train/score.js）同一张时间表；render.mjs events 导出的 events.json 里有段落和画面事件：

  开场      低频嗡鸣 + 打字的轻响；“训练”两个字出来一声钟，片名一记重拍
  出厂状态  安静的铺底，好奇；直方图出来一串铃音
  喂一批    拨弦轻轻走起来，字块落下的木头声，心跳一样的底鼓
  前向      八分音符琶音，每过一层一个音往上爬；反向时倒着往下走
  损失      收住，低音
  更新      讲解段：只有铺底，算式每出一行一声玻璃音，方块挪动时一声
  重复 200 步  律动进来（鼓、贝斯、十六分音符琶音），每 10 步一下木鱼
  月→，     先压住（挂留和弦、只有心跳），一路升上去；学会的那一刻和弦解决到 D 大调、全套鼓进来
  全零对照  全部抽掉，只剩一根平直的长音
  一个参数的一生  温暖的收束
  片尾      和弦解决，混响拖尾，淡出

    /mnt/d/cjc/venvs/blackbox/bin/python tools/video/train/compose_train.py --events tools/video/train/events.json --out /mnt/d/cjc/videos/train-glass/score.wav
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
import compose as C  # noqa: E402
from compose import (SR, BEAT, BAR, CH, Bus, n_of, tvec, sine, filt, shelf, noise, mtof, pad_chord, pluck, bell, blip, glass, kick,  # noqa: E402
                     snare, hat, shaker, bass_note, impact, whoosh, riser, reverse_swell, keyclick, wood, droplet, make_ir, reverb, pingpong,
                     master, lufs, true_peak_db)

# 每段的和弦（按段内第几小节取，比段落短就停在最后一个）；None = 这一小节没有和弦
SEC_CHORDS = {
    'open': [None, None, None, 'Dm9', 'Bbmaj7'],
    'init': ['Dmadd9', 'Gm9', 'Bbmaj7', 'A7sus4', 'Dmadd9', 'Gm9'],
    'batch': ['Fmaj7', 'Csus2', 'Dm9', 'Bbmaj7'],
    'fwd': ['Dm9', 'Bbmaj7', 'Fmaj7', 'Csus2', 'Gm9'],
    'loss': ['Gm9', 'A7sus4', 'A7sus4'],
    'bwd': ['Dm9', 'Bbmaj7', 'Gm9', 'Gm9', 'A7sus4'],
    'upd': ['Dmadd9', 'Gm9', 'Bbmaj7', 'A7sus4', 'Dmadd9', 'Gm9', 'Bbmaj7', 'A7sus4', 'Dm9'],
    'run': ['Dm9', 'Bbmaj7', 'Fmaj7', 'Csus2', 'Dm9', 'Bbmaj7', 'Fmaj7', 'Csus2', 'Gm9', 'A7sus4'],
    'hard': ['Gm9', 'Bbmaj7', 'Gm9', 'A7sus4', 'A7sus4', 'D', 'Dsus2'],
    'zero': [None, None, None],
    'life': ['Bbmaj9', 'FmajA', 'Gm9', 'Dsus2', 'D'],
    'end': ['Bbmaj9', 'Bbmaj9', 'FmajA', 'Gm9', 'Bbmaj9', 'FmajA', 'Gm9', 'Dsus2', 'D'],
}
# 每段的鼓：none / heart（心跳）/ half / groove / four / full
DRUMS = {'open': 'none', 'init': 'none', 'batch': 'heart', 'fwd': 'half', 'loss': 'none', 'bwd': 'half', 'upd': 'none',
         'run': 'groove', 'hard': 'heart', 'zero': 'none', 'life': 'heart', 'end': 'none'}
# 琶音：0 = 不弹，2 = 八分音符，4 = 十六分音符，1 = 四分音符（讲解段）
ARP = {'open': 0, 'init': 1, 'batch': 2, 'fwd': 2, 'loss': 0, 'bwd': 2, 'upd': 0, 'run': 4, 'hard': 1, 'zero': 0, 'life': 2, 'end': 0}
LEVEL = {'open': 0.7, 'init': 0.74, 'batch': 0.86, 'fwd': 0.95, 'loss': 0.78, 'bwd': 0.95, 'upd': 0.72, 'run': 0.88, 'hard': 0.86,
         'zero': 0.5, 'life': 0.82, 'end': 0.72}


def compose(ev, out_wav, ceiling=-3.0):
    sections = ev['score']['sections']
    dur = ev['duration']
    events = ev['events']
    learn_t = next((e['t'] for e in events if e['type'] == 'learn'), None)
    nbars = int(round(dur / BAR))
    # 每小节：段名、段内第几小节、和弦
    binfo = []
    for s in sections:
        for k in range(s['bars']):
            seq = SEC_CHORDS.get(s['name'], [None])
            binfo.append((s['name'], k, s['bars'], seq[min(k, len(seq) - 1)], s['energy']))
    binfo = binfo[:nbars]

    pad, arp, bass, drums, fx, bells = (Bus(dur) for _ in range(6))
    kicks = []

    # 1. 开场的嗡鸣：D 的根音 + 五度，缓慢起伏，加一层空气声；片名落下以后淡出
    n = n_of(11.0)
    t = tvec(n)
    hum = sine(mtof(38), n) * 0.35 + sine(mtof(50) * (1 + 0.002 * np.sin(2 * np.pi * 0.07 * t)), n) * 0.3 + sine(mtof(57) * 1.001, n) * 0.18
    air = filt(noise(n), 'bp', (250, 900)) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.11 * t)) * 0.12
    e = np.clip(t / 2.5, 0, 1) ** 2 * np.clip((11.0 - t) / 3.5, 0, 1) ** 1.5
    pad.add(0, (hum + air * 1.6) * e * 0.4)

    # 2. 铺底和弦
    for b, (nm, k, nb, c, en) in enumerate(binfo):
        if not c:
            continue
        t0 = b * BAR
        start, hold = t0 - 0.3, BAR + 0.3 + (BAR * 2 if b == nbars - 1 else 0)
        # 学会的那一刻：D 大调从那一下进来（不等小节线）
        if nm == 'hard' and c == 'D' and learn_t and t0 > learn_t - 0.05:
            start = min(start, learn_t - 0.05)
            hold = t0 + BAR + 0.3 - start
        if nm == 'hard' and c == 'A7sus4' and learn_t and t0 < learn_t < t0 + BAR:
            hold = learn_t + 0.25 - start
        quiet = nm in ('init', 'upd', 'loss')
        bright = 0.22 if quiet else 0.25 + 0.6 * en
        vol = 0.3 + 0.35 * en if nm != 'end' else 0.55
        pad.add(start, pad_chord(CH[c][0], hold, bright), vol)

    # 3. 琶音
    for b, (nm, k, nb, c, en) in enumerate(binfo):
        div = ARP.get(nm, 0)
        if not c or not div:
            continue
        if nm == 'hard' and learn_t and b * BAR >= learn_t - 0.1:
            div = 4  # 学会以后十六分音符铺满
        t0 = b * BAR
        notes = CH[c][0]
        tones = sorted(set([m + 12 for m in notes] + [m + 24 for m in notes[:3]]))
        pattern = list(range(len(tones))) + list(range(len(tones) - 2, 0, -1))
        step = BEAT / div
        for j in range(int(round(BAR / step))):
            m = tones[pattern[(j + b * 3) % len(pattern)]]
            vel = (0.55 + 0.45 * (j % div == 0)) * (0.45 + 0.55 * en)
            br = 0.25 if div == 1 else 0.35 + 0.5 * en
            arp.add(t0 + j * step, pluck(m, 0.6, br, vel), 0.42, pan=0.35 * math.sin(j * 1.3 + b))

    # 4. 鼓和贝斯
    def drum_bar(t0, kind):
        if kind == 'none':
            return
        if kind == 'heart':
            drums.add(t0, kick(0.8)); drums.add(t0 + BEAT * 0.5, kick(0.4)); kicks.extend([t0, t0 + BEAT * 0.5])
            return
        if kind == 'half':
            drums.add(t0, kick(0.7)); kicks.append(t0)
            drums.add(t0 + 2 * BEAT, snare(0.35, clap=False), pan=0.1)
            for j in range(8):
                drums.add(t0 + j * BEAT / 2, hat(0.35 + 0.25 * (j % 2)), pan=0.25)
            return
        four = kind in ('four', 'full')
        for j in range(4):
            if four or j in (0, 2):
                drums.add(t0 + j * BEAT, kick(1.0 if j == 0 else 0.85)); kicks.append(t0 + j * BEAT)
            if j in (1, 3):
                drums.add(t0 + j * BEAT, snare(0.7 if kind == 'full' else 0.5), pan=-0.05)
        dv = 4 if four else 2
        for j in range(4 * dv):
            drums.add(t0 + j * BEAT / dv, hat((0.5 + 0.35 * (j % dv == dv // 2)) * (0.9 if kind == 'full' else 0.7)), pan=0.3)
        drums.add(t0 + 3.5 * BEAT, hat(0.55, open_=True), pan=0.3)
        if kind == 'full':
            for j in range(16):
                drums.add(t0 + j * BEAT / 4 + 0.01, shaker(0.6 + 0.3 * (j % 2)), pan=-0.35)

    for b, (nm, k, nb, c, en) in enumerate(binfo):
        t0 = b * BAR
        kind = DRUMS.get(nm, 'none')
        if nm == 'run':
            kind = 'groove' if k < 4 else 'four'
        if nm == 'hard' and learn_t:
            if t0 >= learn_t - 0.1:
                kind = 'full'
            elif t0 < learn_t < t0 + BAR:
                kind = 'none'
        drum_bar(t0, kind)
        # 学会之前那一小节：军鼓滚奏，滚到学会的那一下
        if nm == 'hard' and learn_t and t0 < learn_t < t0 + BAR:
            for j in range(16):
                tt = learn_t - BAR + j * BAR / 16
                if tt > t0 - BAR:
                    drums.add(tt, snare(0.12 + 0.5 * j / 16, clap=False), pan=0.05)
        # 前一段的最后一小节进律动段：一小串滚奏
        if nm == 'upd' and k == nb - 1:
            for j in range(8):
                drums.add(t0 + BAR / 2 + j * BAR / 16, snare(0.12 + 0.4 * j / 8, clap=False), pan=0.05)
        if c and nm not in ('open', 'zero', 'end'):
            root = CH[c][1]
            if kind in ('groove', 'four', 'full'):
                for j in range(8):
                    bass.add(t0 + j * BEAT / 2, bass_note(root + (12 if j % 4 == 3 else 0), BEAT / 2 * 0.9, 0.9 if j % 2 == 0 else 0.7))
            else:
                bass.add(t0, bass_note(root, BAR * 0.95, 0.4 if nm in ('upd', 'init', 'loss') else 0.6))
    bass.add((nbars - 1) * BAR, bass_note(38, 4.5, 0.6))

    # 5. 画面事件 → 音效 / 铃声
    penta = [62, 65, 67, 69, 72, 74, 77, 79, 81, 84, 86, 89]
    for e in events:
        t, ty = e['t'], e['type']
        b = min(len(binfo) - 1, int(t // BAR))
        c = binfo[b][3] or 'Dm9'
        notes, root = CH[c]
        if ty == 'key':
            i = e.get('k', 0)
            fx.add(t, keyclick(0.24 + 0.06 * ((i * 5) % 3), pitch=0.05 * (i % 4)), pan=-0.25 + 0.05 * (i % 9))
        elif ty == 'ans':
            bells.add(t, bell(74, 3.0, 0.32)); bells.add(t + 0.07, bell(81, 2.6, 0.16), pan=0.25)
        elif ty == 'hit':
            fx.add(t, impact(0.85)); fx.add(t - 1.0, reverse_swell(1.0, 0.45))
            for i, m in enumerate([62, 69, 74]):
                bells.add(t + 0.03 * i, bell(m, 4.0, 0.22), pan=-0.3 + 0.3 * i)
        elif ty == 'reveal':
            for i, m in enumerate(sorted(notes)[-3:]):
                bells.add(t + i * 0.05, bell(m + 12, 3.5, 0.28), pan=-0.3 + 0.3 * i)
        elif ty == 'whoosh':
            fx.add(t - 0.3, whoosh(1.4, 0.45))
        elif ty == 'drop':
            fx.add(t, wood(0.22, 84 + (e.get('b', 0) * 3) % 10), pan=-0.5 + e.get('b', 0) / 8)
        elif ty == 'shift':
            bells.add(t, glass(86, 0.3), pan=0.2)
        elif ty == 'layer':
            o, d = e.get('o', 0), e.get('dir', 1)
            m = penta[min(len(penta) - 1, o + 1)] if d > 0 else penta[min(len(penta) - 1, 10 - o)]
            bells.add(t, blip(m, 0.12, 0.42), pan=-0.4 + 0.08 * o)
            if d < 0:
                bells.add(t + 0.02, droplet(0.18, m + 12), pan=0.3)
        elif ty in ('focus',):
            fx.add(t - 0.6, reverse_swell(0.6, 0.3)); bells.add(t, bell(74, 3.0, 0.3))
        elif ty == 'blip':
            bells.add(t, glass(81 + (int(t * 7) % 3) * 2, 0.3), pan=0.2)
        elif ty == 'pop':
            bells.add(t, blip(81, 0.18, 0.5)); bells.add(t + 0.08, blip(86, 0.18, 0.35), pan=0.2)
        elif ty == 'popAll':
            for i, m in enumerate([74, 79, 81, 86]):
                bells.add(t + 0.04 * i, glass(m, 0.22), pan=-0.4 + 0.25 * i)
        elif ty == 'tick':
            s = e.get('s', 10)
            fx.add(t, wood(0.16 + 0.12 * s / 200, 84 + int(6 * s / 200)), pan=-0.6 + 1.2 * s / 200)
        elif ty == 'rise':
            d = max(1.0, e.get('dur', 3.0))
            fx.add(t, riser(d, 0.5, 50, 76))
        elif ty == 'learn':
            fx.add(t, impact(1.0, 3.5))
            for i, m in enumerate([62, 66, 69, 74, 78]):
                bells.add(t + 0.04 * i, bell(m + 12, 4.5, 0.3), pan=-0.4 + 0.2 * i)
        elif ty == 'flat':
            nn = n_of(5.5)
            tt = tvec(nn)
            tone = (sine(mtof(74), nn) * 0.5 + sine(mtof(62), nn) * 0.35) * np.clip(tt / 0.15, 0, 1) * np.clip((5.5 - tt) / 1.5, 0, 1)
            fx.add(t, tone * 0.16)
        elif ty == 'click':
            fx.add(t, keyclick(0.3, pitch=0.1))
        elif ty == 'final':
            for i, m in enumerate([62, 69, 74, 78, 81]):
                bells.add(t + 0.05 * i, bell(m, 6.0, 0.22), pan=-0.4 + 0.2 * i)

    # 6. 侧链：底鼓响时铺底 / 贝斯 / 琶音让一让
    duck = np.ones(pad.n)
    tt = tvec(pad.n)
    for kt in kicks:
        i = n_of(kt)
        m = min(pad.n - i, n_of(0.5))
        if m > 0:
            duck[i: i + m] = np.minimum(duck[i: i + m], 1 - 0.42 * np.exp(-tt[:m] / 0.11))
    for bus in (pad, bass, arp):
        bus.x *= duck

    # 段落音量（平滑过渡）；全零对照那一段几乎抽空
    lvl = np.ones(pad.n)
    for s_ in sections:
        lvl[n_of(s_['t0']): n_of(s_['t1'])] = LEVEL.get(s_['name'], 1.0)
    # 学会的那一刻起，比前面的律动段再满一点
    if learn_t:
        hs = next((s_ for s_ in sections if s_['name'] == 'hard'), None)
        if hs:
            lvl[n_of(learn_t - 0.05): n_of(hs['t1'])] = 1.1
    lvl = ss.filtfilt([1 - math.exp(-1 / (SR * 0.3))], [1, -math.exp(-1 / (SR * 0.3))], lvl)
    for bus in (pad, bass, arp, drums):
        bus.x *= lvl

    # 7. 混音
    ir = make_ir(3.4)
    ir_short = make_ir(1.4, dark=7000)
    arp_d = arp.x + pingpong(arp.x, BEAT * 0.75, 0.36)
    mix = (pad.x * 0.85 + reverb(pad.x, ir, 0.45) + arp_d * 0.7 + reverb(arp_d, ir, 0.38) + bass.x * 0.62
           + drums.x * 0.75 + reverb(drums.x, ir_short, 0.12) + bells.x * 0.95 + reverb(bells.x, ir, 0.65) + fx.x * 0.75 + reverb(fx.x, ir, 0.3))
    mix = filt(mix, 'high', 30)
    mix = shelf(mix, 110, -3.0, high=False)
    mix = shelf(mix, 4500, 3.0, high=True)
    mix = filt(mix, 'low', 17000, 4)
    n_end = n_of(dur)
    mix = mix[:, :n_end]
    t = tvec(n_end)
    mix *= np.clip(t / 0.4, 0, 1)
    mix *= np.clip((dur - t) / 5.0, 0, 1) ** 1.5
    out = master(mix, ceiling=ceiling)
    wavfile.write(out_wav, SR, out.T.astype(np.float32))
    print(f'wrote {out_wav}: {out.shape[1] / SR:.2f}s, {lufs(out):.2f} LUFS, true peak {true_peak_db(out):.2f} dBTP')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', default='tools/video/train/events.json')
    ap.add_argument('--out', default='/mnt/d/cjc/videos/train-glass/score.wav')
    # 真峰值上限：AAC 解码后密集的瞬态会再冲高 2 dB 左右，留足余量
    ap.add_argument('--ceiling', type=float, default=-3.5)
    a = ap.parse_args()
    compose(json.load(open(a.events)), a.out, a.ceiling)


if __name__ == '__main__':
    main()
