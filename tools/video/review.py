#!/usr/bin/env python3
"""质量检查：从渲染好的帧里每隔 N 秒抽一帧，拼成带时间码的联系表（tile），方便一眼看完整片。

    python tools/video/review.py --frames /mnt/d/cjc/videos/llm-inference/frames60 --fps 60 --every 2 --out /mnt/d/cjc/videos/llm-inference/review
"""
import argparse
import os
import subprocess

FONT = '/mnt/d/cjc/videos/llm-inference/fonts/NotoSerifSC-SemiBold.otf'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', required=True)
    ap.add_argument('--fps', type=int, default=60)
    ap.add_argument('--every', type=float, default=2.0)
    ap.add_argument('--from', dest='t0', type=float, default=0)
    ap.add_argument('--to', dest='t1', type=float, default=None)
    ap.add_argument('--cols', type=int, default=5)
    ap.add_argument('--rows', type=int, default=5)
    ap.add_argument('--width', type=int, default=480)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n = len([f for f in os.listdir(a.frames) if f.endswith('.jpg')])
    t1 = a.t1 if a.t1 is not None else n / a.fps
    per = a.cols * a.rows
    times = []
    t = a.t0
    while t < t1:
        times.append(t)
        t += a.every
    for s in range(0, len(times), per):
        chunk = times[s: s + per]
        inputs, filters = [], []
        for k, tt in enumerate(chunk):
            f = os.path.join(a.frames, f'f{int(round(tt * a.fps)):05d}.jpg')
            inputs += ['-i', f]
            filters.append(f"[{k}:v]scale={a.width}:-1,drawtext=fontfile={FONT}:text='{tt:.1f}s':x=8:y=8:fontsize=22:fontcolor=white:box=1:boxcolor=black@0.6[v{k}]")
        while len(chunk) < per:  # 补黑格
            k = len(chunk)
            filters.append(f"color=black:s={a.width}x{a.width * 9 // 16}:d=1[v{k}]")
            chunk.append(None)
        stack = ''.join(f'[v{k}]' for k in range(per))
        layout = '|'.join(f'{(k % a.cols) * a.width}_{(k // a.cols) * (a.width * 9 // 16)}' for k in range(per))
        filters.append(f'{stack}xstack=inputs={per}:layout={layout}[out]')
        out = os.path.join(a.out, f'sheet_{times[s]:06.1f}.jpg')
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', *inputs, '-filter_complex', ';'.join(filters), '-map', '[out]', '-frames:v', '1', '-q:v', '3', out], check=True)
        print(out)


if __name__ == '__main__':
    main()
