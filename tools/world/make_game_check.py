"""生成游戏一致性核对用例：Python 版跑若干局，记下每一步的动作、每一帧的哈希、撞车原因、启发式司机的选择。
tools/world/check_game.mjs 用 JS 版重放同样的种子和动作，逐帧比对。

    python tools/world/make_game_check.py --out /mnt/d/cjc/world-model/game_check.json
"""
import argparse
import hashlib
import json
import random

from game import Game, heuristic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/mnt/d/cjc/world-model/game_check.json')
    ap.add_argument('--episodes', type=int, default=60)
    args = ap.parse_args()
    rnd = random.Random(7)
    cases = []
    frames = 0
    for e in range(args.episodes):
        seed = rnd.randrange(1 << 32)
        g = Game(seed)
        acts, hashes, heur = [], [hashlib.md5(g.render().tobytes()).hexdigest()], []
        mode = e % 3   # 0 启发式，1 随机（带惯性），2 启发式 + 噪声
        a, hold = 0, 0
        while not g.done and g.t < 400:
            h = heuristic(g)
            heur.append(h)
            if mode == 1 or (mode == 2 and rnd.random() < 0.25):
                if hold <= 0:
                    a, hold = rnd.randrange(3), rnd.randrange(1, 7)
                hold -= 1
                act = a
            else:
                act = h
            acts.append(act)
            g.step(act)
            hashes.append(hashlib.md5(g.render().tobytes()).hexdigest())
        frames += len(hashes)
        cases.append({'seed': seed, 'acts': acts, 'hashes': hashes, 'heur': heur, 'why': g.why, 'T': g.t})
    with open(args.out, 'w') as f:
        json.dump(cases, f)
    lens = [c['T'] for c in cases]
    print(f'{len(cases)} 局，{frames} 帧，局长 {min(lens)}–{max(lens)}，撞车 {sum(c["why"] == "crash" for c in cases)}，冲出路面 {sum(c["why"] == "offroad" for c in cases)}')


if __name__ == '__main__':
    main()
