"""「夜路」小游戏的 Python 实现（采集训练数据用）。

和 public/world/js/game.js 是同一个游戏的两份实现：只用整数运算、随机数是 xorshift32，
同一个种子、同一串动作 → 两边逐像素相同（tools/world/check_game.mjs 核对）。
渲染结果是调色板编号（uint8, 64×64），转成 RGB 时查 PALETTE。
"""
import numpy as np

W = H = 64
HALF = 12
CAR_Y = 48
CAR_H, CAR_W = 8, 5
OB_H = 7
SPEED = 2
STEER = 2
FP = 16
LO, HI = (HALF + 3) * FP, (W - 1 - HALF - 3) * FP
TARGETS = [-10, -6, -3, 0, 0, 3, 6, 10]
SEG = 24
SPAWN = 30
GAP = 14
M32 = 0xFFFFFFFF

PALETTE = np.array([
    [15, 32, 29], [24, 49, 41], [35, 40, 51], [93, 101, 119], [184, 146, 63], [72, 214, 189],
    [11, 47, 44], [208, 86, 111], [77, 26, 39], [47, 53, 66], [22, 48, 43], [240, 211, 138],
], dtype=np.uint8)

_ = -1
CAR = np.array([
    [11, 5, 5, 5, 11],
    [5, 5, 5, 5, 5],
    [5, 6, 6, 6, 5],
    [5, 6, 6, 6, 5],
    [5, 5, 5, 5, 5],
    [5, 5, 5, 5, 5],
    [5, 6, 6, 6, 5],
    [5, 5, 5, 5, 5],
])
OB = np.array([
    [_, 7, 7, 7, _],
    [7, 8, 8, 8, 7],
    [7, 7, 7, 7, 7],
    [7, 7, 7, 7, 7],
    [7, 8, 8, 8, 7],
    [7, 7, 7, 7, 7],
    [7, _, _, _, 7],
])


def rng_init(seed):
    x = ((seed & M32) ^ 0x9E3779B9) & M32
    if x == 0:
        x = 1
    for _i in range(8):
        x = rng_next(x)
    return x


def rng_next(x):
    x ^= (x << 13) & M32
    x ^= x >> 17
    x ^= (x << 5) & M32
    return x & M32


def sign(v):
    return (v > 0) - (v < 0)


_XS = np.arange(W, dtype=np.uint64)


def tuft_rows(wr):
    """wr: (64,) 世界行号 → (64, 64) 布尔：这一格有没有草丛（和 JS 的 tuft() 一致）"""
    h = (_XS[None, :] * np.uint64(374761393) + wr.astype(np.uint64)[:, None] * np.uint64(668265263)) & np.uint64(M32)
    h = ((h ^ (h >> np.uint64(13))) * np.uint64(1274126177)) & np.uint64(M32)
    return (h >> np.uint64(24)) < 16


class Game:
    def __init__(self, seed):
        self.reset(seed)

    def reset(self, seed):
        self.seed = seed & M32
        self.rx = rng_init(self.seed)
        self.cx = np.zeros(64, dtype=np.int64)
        self.cxv = 32 * FP
        self.vx = 0
        self.tv = 0
        self.next_ok = 40
        self.obs = []          # [n, x]
        self.base = 0
        self.top = -1
        while self.top < 63:
            self.gen_row()
        self.car_x = self.center_at(self.base + 12)
        self.t = 0
        self.done = False
        self.why = ''
        return self

    def rand(self):
        self.rx = rng_next(self.rx)
        return self.rx

    def gen_row(self):
        self.top += 1
        n = self.top
        r1, r2, r3 = self.rand(), self.rand(), self.rand()
        if n % SEG == 0:
            self.tv = TARGETS[r1 % 8]
        self.vx += sign(self.tv - self.vx)
        self.cxv += self.vx
        if self.cxv < LO:
            self.cxv, self.vx, self.tv = LO, 0, abs(self.tv)
        if self.cxv > HI:
            self.cxv, self.vx, self.tv = HI, 0, -abs(self.tv)
        self.cx[n & 63] = self.cxv
        if n >= self.next_ok and r2 % 1000 < SPAWN:
            self.obs.append([n, ((self.cxv + 8) >> 4) + (r3 % 19) - 9])
            self.next_ok = n + GAP

    def center_at(self, wr):
        return (int(self.cx[wr & 63]) + 8) >> 4

    def step(self, a):
        if self.done:
            return
        if a == 1:
            self.car_x -= STEER
        elif a == 2:
            self.car_x += STEER
        self.car_x = max(2, min(W - 3, self.car_x))
        for _k in range(SPEED):
            self.base += 1
            self.gen_row()
        self.obs = [o for o in self.obs if o[0] + OB_H - 1 >= self.base]
        self.t += 1
        lo = self.base + (H - 1 - (CAR_Y + CAR_H - 1))
        hi = self.base + (H - 1 - CAR_Y)
        for n, x in self.obs:
            if n <= hi and n + OB_H - 1 >= lo and abs(x - self.car_x) <= CAR_W - 1:
                self.done, self.why = True, 'crash'
        if not self.done and abs(self.car_x - self.center_at(self.base + 12)) > HALF - 1:
            self.done, self.why = True, 'offroad'

    def render(self):
        wr = self.base + H - 1 - np.arange(H)
        c = (self.cx[wr & 63] + 8) >> 4
        xs = np.arange(W)
        d = np.abs(xs[None, :] - c[:, None])
        dash = ((wr >> 2) & 1) == 0
        center = (xs[None, :] == c[:, None]) & dash[:, None]
        img = np.where(d < HALF, np.where(center, 4, 2), np.where(d == HALF, 3, np.where(tuft_rows(wr), 1, 0))).astype(np.uint8)
        for dd in range(1, 15):
            r, hw = CAR_Y - dd, 1 + dd // 3
            x0, x1 = max(0, self.car_x - hw), min(W - 1, self.car_x + hw)
            seg = img[r, x0:x1 + 1]
            seg[seg == 2] = 9
            seg[(seg == 0) | (seg == 1)] = 10
        for n, ox in self.obs:
            for k in range(OB_H):
                r = self.base + (H - 1) - (n + OB_H - 1) + k
                if r < 0 or r >= H:
                    continue
                for j in range(CAR_W):
                    x, p = ox - 2 + j, OB[k, j]
                    if p >= 0 and 0 <= x < W:
                        img[r, x] = p
        for k in range(CAR_H):
            for j in range(CAR_W):
                x, p = self.car_x - 2 + j, CAR[k, j]
                if p >= 0 and 0 <= x < W:
                    img[CAR_Y + k, x] = p
        return img

    def state(self):
        """训练之后给潜变量各维找含义用：几个可以直接读出来的游戏状态"""
        ahead = [self.center_at(self.base + k) for k in (12, 32, 52)]
        near = [o for o in self.obs if o[0] - (self.base + 8) < 56]
        ob = min(near, key=lambda o: o[0]) if near else None
        return {
            'car_x': self.car_x,
            'road_near': ahead[0], 'road_mid': ahead[1], 'road_far': ahead[2],
            'bend': ahead[2] - ahead[0],
            'car_off': self.car_x - ahead[0],
            'n_obs': len(near),
            'ob_y': (self.base + 63 - (ob[0] + OB_H - 1)) if ob else -1,
            'ob_x': ob[1] if ob else -1,
            'dash': ((self.base >> 1) & 3),
        }


PLAN, BAD = 20, 1000000


def heuristic(g):
    """和 JS 的 heuristic() 一致：往前看 20 步，动态规划挑不撞车、尽量贴着路中间的走法，返回第一步的动作"""
    xs = np.arange(2, W - 2)
    nxt = np.zeros(W, dtype=np.int64)
    for k in range(PLAN, 0, -1):
        b = g.base + SPEED * k
        lo, hi, c = b + 8, b + 15, g.center_at(b + 12)
        ok = np.abs(xs - c) <= HALF - 1
        for n, ox in g.obs:
            if n <= hi and n + OB_H - 1 >= lo:
                ok &= np.abs(ox - xs) > CAR_W - 1
        if k < PLAN:
            best = np.maximum.reduce([nxt[np.clip(xs + d, 2, W - 3)] for d in (0, -STEER, STEER)])
        else:
            best = np.zeros_like(xs)
        cur = np.zeros(W, dtype=np.int64)
        cur[xs] = np.where(ok, best - np.abs(xs - c), -BAD * (PLAN - k + 1))
        nxt = cur
    a, best = 0, -BAD * PLAN * 4
    for act, d in ((0, 0), (1, -STEER), (2, STEER)):
        v = nxt[max(2, min(W - 3, g.car_x + d))]
        if v > best:
            best, a = v, act
    return a
