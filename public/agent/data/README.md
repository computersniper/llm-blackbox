# 智能体录制数据

`tools/record_agent.py` 使用本机 Qwen3-1.7B 在临时任务目录中运行小型编程任务。模型逐词元生成思考、回答和工具调用；录制器按命令白名单执行工具并将输出送回模型。页面只播放这些记录，不在浏览器里执行命令。

## 文件

- `manifest.json`：模型、采样参数、工具定义与真实录制任务列表。
- `<task-id>.json`：任务录制，包含每轮生成的词元概率、工具参数、终端结果、文件 diff、起止文件快照和独立复核结果。
- `<task-id>.json.gz`：同一内容的压缩副本。

目录中可能还保留旧版模拟文件；只有 `manifest.json` 列出的真实录制任务会出现在页面中。重新录制相应任务后，模拟文件会被覆盖。

## 重新录制

在安装了 `torch`、`transformers` 的 `xnewenv` 环境中运行：

```powershell
conda run --no-capture-output -n xnewenv python tools/record_agent.py --model D:/models/Qwen3-1.7B
```

也可加 `--only fix-bug`、`--only add-feature`、`--only refactor` 或 `--only analyze-data`。脚本会重置 `--sandbox` 下对应任务的目录，默认位置是 `D:/models/agent_sandbox`。不要将 `--sandbox` 指向需要保留的目录。

`verified` 同时要求任务自带测试和录制器持有的独立用例通过。加功能与重构任务还会在临时副本中故意改错目标函数，要求任务测试确实报错，防止测试只定义而未执行。独立用例不保存在模型可修改的任务目录中；如果模型提前回答而复核失败，录制器会把错误反馈给模型，让它继续修复。每次这样的检查记录在对应轮次的 `autoCheck` 中。最终失败的录制保留在磁盘供诊断，不进入 `manifest.json`，页面不会把它展示为完成任务。

执行器只允许查看当前目录中的文件以及运行指定的测试脚本，并给子进程最小环境和 30 秒超时。这是本机临时目录中的受限执行，不提供操作系统级隔离；不要将不可信代码或文件放入任务模板。
