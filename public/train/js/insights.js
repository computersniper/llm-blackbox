// 知识碎片：在训练现场走到特定的地方才会解锁。en = 英文的 [位置, 标题, 正文]。
import { isEn } from './lang.js';

const RAW = [
  { id: 'format', en: ["Pretraining run · first few hundred steps", "Format before content", "The first thing the tiny model learns isn’t poetry but punctuation: a few dozen steps in, it already predicts more than 90% of the commas and full stops in “On the Stork Tower” correctly, while it can barely guess the characters themselves. The most frequent, most regular patterns are learned first."], where: '预热全程 · 前几百步', title: '格式先于内容',
    text: '小模型最先学会的不是诗意，而是标点：几十步之后，《登鹳雀楼》里逗号、句号出现的位置已经能猜中九成以上，正文的字却还几乎猜不中。最常见、最有规律的东西最先被学会。' },
  { id: 'warmup', en: ["Pretraining run · learning rate", "Learning-rate warmup", "Right after initialization the gradient directions are chaotic and Adam’s second-moment estimate hasn’t built up yet. Ramping the learning rate linearly from 0 to its peak over the first 200 steps keeps the weights from being shoved somewhere strange at the very start."], where: '预热全程 · 学习率', title: '学习率预热',
    text: '刚初始化时梯度方向很乱，Adam 的二阶动量也还没攒起来。前 200 步让学习率从 0 线性升到峰值，可以避免一开始就把权重推到奇怪的地方。' },
  { id: 'cosine', en: ["Pretraining run · the end", "Cosine annealing", "After warmup the learning rate slowly follows a cosine curve down to a tenth of its peak. Smaller steps late in training let the loss settle deeper into the valley — the training loss is still falling at the end of the curve partly thanks to annealing."], where: '预热全程 · 末尾', title: '余弦退火',
    text: '预热之后，学习率沿余弦曲线慢慢降到峰值的十分之一。后期步子变小，损失能更细地落进谷底——曲线末端训练损失还在下降，就有退火的功劳。' },
  { id: 'teacher', en: ["One step · forward", "Teacher forcing", "During training every position is given the real preceding text, not characters the model wrote itself, so all positions in a row can be computed at once. Generation has to go one token at a time — training is far faster than generating."], where: '一步之内 · 前向', title: '教师强制',
    text: '训练时每个位置拿到的都是真实的上文，而不是模型自己写出来的字，所以一行里所有位置可以同时算。推理时却只能一个一个地生成——训练比生成快得多。' },
  { id: 'lens', en: ["Forward · layer by layer", "A logit lens on training", "Wire a middle layer’s residual stream straight into the output head and you can read “what it would guess right now”. The higher the layer, the more positions it gets right: the prediction is built up layer by layer."], where: '前向 · 逐层', title: '逻辑透镜看训练',
    text: '把中间层的残差流直接接到输出头上，能读出“此刻它会猜什么”。越往上，猜中的位置越多：预测是一层层加工出来的。' },
  { id: 'ce', en: ["Loss · one position", "Cross-entropy", "The loss only looks at the probability p of the correct answer and takes −ln p. Confident and right gives almost 0; a blind guess gives ln(vocabulary size). Average over all positions and you get that falling loss curve."], where: '损失 · 一个位置', title: '交叉熵',
    text: '损失只看正确答案的概率 p，取 −ln p。猜对且很有把握时接近 0；完全瞎猜时等于 ln(词表大小)。所有位置平均起来，就是那条往下走的损失曲线。' },
  { id: 'backprop', en: ["One step · backward", "Backpropagation", "Gradients start at the output head and flow back layer by layer, the opposite direction of the forward pass. Residual connections act like a highway that carries gradients to the bottom without passing through any matrix — that is what makes networks dozens of layers deep trainable."], where: '一步之内 · 反向', title: '反向传播',
    text: '梯度从输出头出发，沿着和前向相反的方向一层层流回去。残差连接像一条高速公路，让梯度不经过任何矩阵也能直达底层，几十层的网络才训得动。' },
  { id: 'outer', en: ["Backward · inside one layer", "Gradients are outer products", "A matrix’s gradient = the upstream gradient flowing into it × its input (an outer product), summed over all positions. So weights change most along directions where both the input and the upstream gradient are large."], where: '反向 · 一层之内', title: '梯度是外积',
    text: '一个矩阵的梯度 = 流进它的上游梯度 × 它的输入（外积），再对所有位置求和。所以输入越大、上游梯度越大的方向，权重改得越多。' },
  { id: 'clip', en: ["Update · clipping", "Gradient clipping", "When the combined length of all gradients exceeds 1.0, everything is scaled down to 1.0. The direction stays the same; only the step is capped — so one odd batch can’t kick the model off course. On Qwen3-0.6B that step’s gradient norm was 254, so it was shrunk more than 200-fold."], where: '更新 · 裁剪', title: '梯度裁剪',
    text: '所有梯度合起来的长度超过 1.0 时，整体按比例缩短到 1.0。方向不变，只限制步子——防止偶然的一批怪数据把模型“踹”飞。Qwen3-0.6B 那一步的梯度长度是 254，被缩小了两百多倍。' },
  { id: 'sign', en: ["Update · step 1", "Adam’s first step only sees the sign", "At step 1, m̂ equals g and v̂ equals g² exactly, so the update becomes lr × g/|g|: every weight moves exactly lr against its gradient, however large the gradient is. The loss on that Qwen3-0.6B conversation drops from 5.0 to 1.4 in one step."], where: '更新 · 第 1 步', title: 'Adam 的第一步只看符号',
    text: '第 1 步时 m̂ 正好等于 g，v̂ 正好等于 g²，更新量变成 lr × g/|g|：每个权重都朝梯度的反方向挪 lr 那么远，不管梯度本身多大。Qwen3-0.6B 那条对话的损失一步就从 5.0 掉到 1.4。' },
  { id: 'bias', en: ["Update · bias correction", "Bias correction", "m and v are moving averages that start at 0, so for the first few steps they are dragged too small. Dividing by (1 − βᵗ) scales them back up; as t grows, the factor approaches 1."], where: '更新 · 偏差校正', title: '偏差校正',
    text: 'm 和 v 从 0 开始滑动平均，前几步会被 0 拖累得偏小。除以 (1 − βᵗ) 把它们放大回来；t 越大，这个系数越接近 1。' },
  { id: 'decay', en: ["Update · Δw", "Decoupled weight decay", "AdamW applies weight decay on its own: every step multiplies the weight by (1 − lr·λ), bypassing m and v. It keeps weights from growing without bound. By convention RMSNorm scales are not decayed."], where: '更新 · Δw', title: '解耦的权重衰减',
    text: 'AdamW 把权重衰减单独算：每步把权重乘以 (1 − lr·λ)，不经过 m 和 v。它让权重不会无限长大。RMSNorm 的缩放系数按惯例不做衰减。' },
  { id: 'fp32', en: ["Update · bits", "Why fp32 master weights", "One update changes a weight by only a few parts in 100,000. bf16 has just 7 mantissa bits, so a change that small is swallowed by rounding. After Qwen3-0.6B’s 3 steps, about 85% of the weights would snap back to their original values if stored in bf16 — which is why training keeps a separate fp32 copy of the master weights."], where: '更新 · 比特', title: '为什么要 fp32 主权重',
    text: '一次更新只改动权重的十万分之几。bf16 只有 7 位尾数，这么小的改动在舍入时会被直接吞掉。Qwen3-0.6B 走完 3 步后，如果把权重存回 bf16，有约 85% 会变回原值。所以训练时要另存一份 fp32 的主权重。' },
  { id: 'template', en: ["SFT · batch", "Chat template", "A question and its answer are stitched into one token sequence with special markers: <|im_start|>user … <|im_end|> <|im_start|>assistant …. The model is still only learning to “continue the text” — the format of the continuation is simply fixed."], where: 'SFT · 批次', title: '聊天模板',
    text: '一问一答被拼成一串带特殊标记的词元：<|im_start|>user … <|im_end|> <|im_start|>assistant …。模型学的始终是“续写”，只是续写的格式被固定了下来。' },
  { id: 'sftmask', en: ["SFT · mask", "Loss only on the answer", "SFT uses exactly the same loss function as pretraining; the only difference is that the prompt’s labels are set to −100. The model doesn’t have to learn to “ask” — only “how to answer when it’s my turn”."], where: 'SFT · 遮罩', title: '只对回答算损失',
    text: 'SFT 和预训练的损失函数一模一样，区别只是把提示部分的标签设成 −100。模型不必学会“提问”，只学“轮到我时怎么答”。' },
  { id: 'memorize', en: ["SFT, 3 steps · the end", "Memorized in three steps", "Train on the same conversation 3 steps in a row and the answer’s loss falls from 5.0 to 0.01 — the model has memorized the sentence almost word for word. Real SFT uses thousands of different conversations and a tiny learning rate precisely so that it learns a style rather than rote answers."], where: 'SFT 三步 · 末尾', title: '三步就背下来了',
    text: '同一条对话连着训 3 步，回答的损失从 5.0 掉到 0.01，模型几乎一字不差地背下了这句话。真实的 SFT 用成千上万条不同的对话、很小的学习率，就是为了学会“风格”而不是死记硬背。' },
  { id: 'onestep', en: ["One step · forward again", "How much one step changes", "Early in training a single update visibly lowers the loss on the same batch; late in training that loss moves by only a few thousandths. The model is settling down, and each step is just a small adjustment."], where: '一步之内 · 重新前向', title: '一步能改变多少',
    text: '训练初期，一步更新能让同一批数据的损失明显下降；到了后期，同一批数据的损失只变化千分之几。模型越来越“定型”，每一步都只是微调。' },
  { id: 'head', en: ["Full run · attention heads", "Heads divide up the work", "At random initialization every head “looks evenly” at everything. During training each one finds a job: some watch the previous character, some the beginning, and some look at the same position in the previous line — exactly where the parallel couplets line up."], where: '训练全程 · 注意力头', title: '注意力头的分工',
    text: '随机初始化时每个头都在“平均地看”。训练中它们各自找到差事：有的专看前一个字，有的专看开头，还有的会去看上一句同一位置的字——那正是对仗的地方。' },
  { id: 'dpo', en: ["Pipeline · preference alignment", "DPO", "DPO needs no reward model: given a good and a bad answer to the same question, it nudges the current model, relative to a reference model, toward the good one. Loss = −log σ(β·[(gain on good) − (gain on bad)])."], where: '流水线 · 偏好对齐', title: 'DPO',
    text: 'DPO 不需要奖励模型：对同一个问题的一好一坏两个回答，让当前模型相对参考模型更偏向好的那个。损失 = −log σ(β·[(好的提升) − (坏的提升)])。' },
];

export const INSIGHTS = RAW.map(({ en, ...x }) => (isEn ? { ...x, where: en[0], title: en[1], text: en[2] } : x));

export const INSIGHT_BY_ID = new Map(INSIGHTS.map((x) => [x.id, x]));
