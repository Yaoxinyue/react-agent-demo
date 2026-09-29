# react-agent-demo

用 Python 实现的最小 ReAct Agent：接收问题 → LLM 思考 → 调用工具（计算/搜索）→ 返回结果。

## 架构说明（10 行）

1. `agent.py` 是主循环，只做三件事：问模型、解析输出、把工具结果回灌。
2. 每一轮模型必须输出 `Thought` / `Action` / `Action Input` 三段，或者用 `Final Answer` 收尾。
3. `tools.py` 里每个工具都是普通函数——字符串进、字符串出，完全不感知 LLM 的存在。
4. 循环把工具返回值拼成 `Observation: ...` 追加进对话历史，再交给模型进入下一轮。
5. 终止条件有两个：解析到 `Final Answer`，或达到 `MAX_STEPS`（默认 6）防止死循环。
6. 模型调用统一走 `OpenAIChat`，兼容任何 OpenAI 格式端点，换模型只改 `.env` 不动代码。
7. 提示词硬约束输出格式 + 正则解析，是这套方案能跑通的关键；格式错就把报错回灌给模型自纠。
8. `test_offline.py` 用脚本化假模型覆盖了正常循环、工具名幻觉、超步数三种情况，零成本可复现。
9. `run_agent()` 额外返回完整的 trace（每步的 action / input / observation），用于复盘和写博客。
10. 加新工具只需两步：在 `tools.py` 写函数并注册到 `TOOLS`，在 `TOOL_DESCRIPTIONS` 补一行描述。

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python test_offline.py
python agent.py "帮我算一下 (1234*5678)/2，再查一下 SWE-bench 是什么"
```

> Windows 上的激活命令是 `.venv\Scripts\activate`。

`cp` 之后打开 `.env` 填入自己的 API key —— 示例文件里列了 OpenAI / DeepSeek / 智谱 / 通义 / 本地 Ollama 五种端点，换模型只改这里三行，不动代码。`test_offline.py` 不联网、不花额度，用来先确认代码本身没问题。

## 实测结果：工具到底值不值得用

`bench_arithmetic.py` 是一组对照实验：同一批题分别"直接问模型"和"走 Agent 调 calculator"，两种模式各 140 题，`temperature=0`，除法题由乘法构造以保证结果为精确整数。

| 参与运算的位数 | 直接问模型（乘法 / 除法） | 走 Agent（乘法 / 除法） |
| ------- | ------------- | --------------- |
| 2–3 位   | 100% / 100%   | 100% / 100%     |
| 4 位     | 60% / 90%     | 100% / 100%     |
| 5 位     | 50% / 60%     | 100% / 100%     |
| 6–8 位   | 0% / 0%       | 100% / 100%     |

**结论**：模型自算的能力边界落在 **4～6 位之间**；超过 6 位它不是在计算，而是在"续写"一个看起来像答案的数字串（典型例子：把 `699019262795` 答成 `699000000000`）。所以 calculator 的价值不是"让模型更聪明"，而是**把这一步从模型手里拿走、交给确定性的 Python**——代价是每题多一次 LLM 调用。原始数据与判分脚本都在 `bench/` 下，可复现。
