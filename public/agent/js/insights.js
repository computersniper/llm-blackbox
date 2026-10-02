// 知识碎片：在 agent 里走到特定的地方才会解锁。
// 英文版的例子和数字取自英文录制（data/en/）：工具定义 474 个词元；犯错再改正的例子都出自英文轨迹。
import { isEn } from '../../js/i18n.js';

const ZH = [
  { id: 'loop', where: '循环', title: 'agent = 模型 + 工具 + 循环',
    text: '所谓 agent，就是一个 while 循环：把对话喂给模型 → 模型写下要调用的工具 → 程序去执行 → 把结果接回对话 → 再喂给模型。Claude Code 这类编程 agent 也是这个模式。' },
  { id: 'harness', where: '循环 · 解析', title: '模型只会写字',
    text: '模型从头到尾只是在输出文字。<tool_call> 只是它写下的一段 JSON；真正去执行命令、读写文件的，是外面那个程序（harness）。' },
  { id: 'sandbox', where: '循环 · 执行', title: '沙箱与权限',
    text: '模型写出的命令会被真的执行，所以要关进笼子：这里的沙箱让系统目录只读、只能写 /work、断网、10 秒超时。Claude Code 这类工具还会在执行有风险的命令前先问你。' },
  { id: 'stop', where: '循环 · 结束', title: '什么时候停',
    text: '没有人告诉循环“任务完成了”。只要模型的一次输出里不再有 <tool_call>，harness 就把它当成最终回答，循环结束。' },
  { id: 'tooldefs', where: '上下文 · 第一圈', title: '工具也是提示词',
    text: '模型怎么知道有哪些工具？工具的名字、说明、参数格式被写成 JSON，塞进系统提示里。这里 4 个工具连同调用格式的说明，一共四百多个词元，而且每一圈都要重新读一遍。' },
  { id: 'refeed', where: '上下文', title: '每一圈都从头读',
    text: '模型没有记忆。每一圈，harness 都把整段历史——系统提示、你的任务、它之前说过的每句话、每一次工具结果——重新拼成一串喂进去。' },
  { id: 'growth', where: '上下文 · 阶梯', title: '上下文越滚越长',
    text: '每调用一次工具，结果就永远留在上下文里。读一个文件、跑一次测试，几百个词元就进来了。长任务的上下文会一直涨，直到撞上模型的上限。' },
  { id: 'kvreuse', where: '上下文 · KV 缓存', title: '复用 KV 缓存',
    text: '虽然每一圈都要“从头读”，但前面的部分和上一圈一模一样，它们在每一层的 K、V 可以直接复用。真正要新算的，只有刚接上去的那一小段。' },
  { id: 'prefill', where: '上下文 · KV 缓存', title: '读得快，写得慢',
    text: '预填充时几百个词元可以并行算，几十毫秒就完；生成时却只能一个词元一个词元地来，每个都要把整个模型跑一遍。所以 agent 的时间大多花在“写”上。' },
  { id: 'toolresp', where: '上下文 · 工具结果', title: '工具结果以 user 的身份回来',
    text: '在 Qwen3 的聊天模板里，工具结果被包进 <tool_response>，放在一条 user 消息里。对模型来说，命令的输出和用户说的话一样，都只是上下文里的文字。' },
  { id: 'decision', where: '生成 · 关键词元', title: '决策就是一次抽签',
    text: '“先读文件还是先跑测试”“用 python 还是 python3”，在模型内部都只是某个位置上几个候选词元的概率。agent 的每个决定，都是这样一个词元一个词元选出来的。' },
  { id: 'json', where: '生成 · 工具调用', title: '工具调用是一个词元一个词元写出来的',
    text: '工具名、命令、文件路径、要替换的原文，都是模型逐个词元写出的 JSON。写错一个引号，harness 就解析失败；写错一个字，edit_file 就找不到原文。' },
  { id: 'selfcorrect', where: '屏幕 · 报错之后', title: '犯错，然后自己改',
    text: '命令报错、找不到文件、缺依赖……错误信息原样回到上下文里，模型下一圈读到它，就可能换个办法。这里的几条轨迹里都能看到：python 不存在就换 python3，装不了 pandas 就改用 csv 模块。' },
  { id: 'forward', where: '模型内部', title: '每个词元都是一次完整的计算',
    text: '写出一个词元，就要把它穿过全部 36 层、动用约 40 亿个参数。一趟任务几百个词元，就是几百次这样的前向计算，外加每一圈的预填充。' },
];

const EN = [
  { id: 'loop', where: 'Loop', title: 'agent = model + tools + loop',
    text: 'An “agent” is just a while loop: feed the conversation to the model → the model writes down the tool it wants to call → a program runs it → the result is appended to the conversation → feed it to the model again. Coding agents like Claude Code follow the same pattern.' },
  { id: 'harness', where: 'Loop · Parse', title: 'The model only writes text',
    text: 'From start to finish the model only outputs text. <tool_call> is just a piece of JSON it writes; what actually runs commands and reads or writes files is the program outside (the harness).' },
  { id: 'sandbox', where: 'Loop · Execute', title: 'Sandbox and permissions',
    text: 'The commands the model writes really get executed, so they have to be caged: this sandbox makes system directories read-only, allows writes only to /work, cuts the network and times out after 10 seconds. Tools like Claude Code also ask you first before running risky commands.' },
  { id: 'stop', where: 'Loop · Done', title: 'When does it stop?',
    text: 'Nobody tells the loop “the task is done”. As soon as one of the model’s outputs no longer contains a <tool_call>, the harness treats it as the final answer and the loop ends.' },
  { id: 'tooldefs', where: 'Context · first turn', title: 'Tools are part of the prompt too',
    text: 'How does the model know which tools exist? Their names, descriptions and parameter formats are written as JSON into the system prompt. Here the 4 tools plus the calling instructions take 474 tokens — and they are read again on every turn.' },
  { id: 'refeed', where: 'Context', title: 'Every turn starts from the top',
    text: 'The model has no memory. Every turn, the harness reassembles the whole history — system prompt, your task, everything it said before, every tool result — into one string and feeds it in.' },
  { id: 'growth', where: 'Context · staircase', title: 'The context keeps growing',
    text: 'Every tool call leaves its result in the context for good. Read a file or run the tests and a few hundred tokens come in. In a long task the context keeps growing until it hits the model’s limit.' },
  { id: 'kvreuse', where: 'Context · KV cache', title: 'Reusing the KV cache',
    text: 'Even though every turn “reads from the top”, the beginning is identical to last turn, so its K and V in every layer can be reused as-is. Only the small piece just appended needs computing.' },
  { id: 'prefill', where: 'Context · KV cache', title: 'Fast to read, slow to write',
    text: 'In prefill, hundreds of tokens are computed in parallel and done in tens of milliseconds; in generation it’s one token at a time, each a full pass through the model. So an agent spends most of its time “writing”.' },
  { id: 'toolresp', where: 'Context · tool results', title: 'Tool results come back as the user',
    text: 'In Qwen3’s chat template, a tool result is wrapped in <tool_response> inside a user message. To the model, a command’s output is just like what the user says: text in the context.' },
  { id: 'decision', where: 'Generation · key tokens', title: 'A decision is a draw',
    text: '“Read the file first or run the tests first?”, “python or python3?” — inside the model these are just the probabilities of a few candidate tokens at one position. Every decision an agent makes is picked token by token like this.' },
  { id: 'json', where: 'Generation · tool calls', title: 'Tool calls are written token by token',
    text: 'The tool name, the command, the file path, the text to replace — all are JSON written by the model token by token. Get one quote wrong and the harness can’t parse it; get one character wrong and edit_file can’t find the text.' },
  { id: 'selfcorrect', where: 'Screen · after an error', title: 'Make a mistake, then fix it',
    text: 'A command fails, a file is missing, a dependency can’t be installed… the error goes back into the context verbatim, and next turn the model may try another way. You can see it in these runs: “python: command not found” → it switches to python3; apt-get can’t install anything without a network → it stops trying; a tool call that isn’t valid JSON → it rewrites the call.' },
  { id: 'forward', where: 'Inside the model', title: 'Every token is a full computation',
    text: 'Writing one token means sending it through all 36 layers and using about 4 billion parameters. A task of a few hundred tokens is a few hundred such forward passes, plus the prefill of every turn.' },
];

export const INSIGHTS = isEn ? EN : ZH;

export const INSIGHT_BY_ID = new Map(INSIGHTS.map((x) => [x.id, x]));
