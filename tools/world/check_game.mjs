// 核对 JS 版游戏和 Python 版逐像素一致：重放 make_game_check.py 记下的种子和动作，逐帧比哈希。
//   node tools/world/check_game.mjs /mnt/d/cjc/world-model/game_check.json
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { Game, heuristic } from '../../public/world/js/game.js';

const cases = JSON.parse(readFileSync(process.argv[2] || '/mnt/d/cjc/world-model/game_check.json', 'utf8'));
const md5 = (u8) => createHash('md5').update(u8).digest('hex');
let frames = 0, bad = 0, badHeur = 0, badEnd = 0;
for (const c of cases) {
  const g = new Game(c.seed);
  const out = new Uint8Array(64 * 64);
  if (md5(g.render(out)) !== c.hashes[0]) bad++;
  frames++;
  c.acts.forEach((a, i) => {
    if (heuristic(g) !== c.heur[i]) badHeur++;
    g.step(a);
    if (md5(g.render(out)) !== c.hashes[i + 1]) { if (!bad) console.log(`第一处不一致：种子 ${c.seed} 第 ${i + 1} 步`); bad++; }
    frames++;
  });
  if (g.why !== c.why || g.t !== c.T) badEnd++;
}
console.log(`${cases.length} 局 · ${frames} 帧：画面不一致 ${bad} 帧，启发式司机不一致 ${badHeur} 次，结局不一致 ${badEnd} 局`);
process.exit(bad || badHeur || badEnd ? 1 : 0);
