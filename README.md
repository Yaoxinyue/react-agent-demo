# react-agent-demo

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![dependency](https://img.shields.io/badge/dependency-openai%20only-orange)
![framework](https://img.shields.io/badge/framework-none%20%28hand--written%20loop%29-success)

> 一个**不借助任何 Agent 框架、从零手写**的最小 ReAct Agent：约 300 行 Python、一个循环、两个工具。
>
> 此项目方向是大模型 + 软件工程（LLM for SE / Agent / SWE-bench）。仓库不是为了做一个能用的产品，而是**为了把一个被各种框架封装起来的概念拆开看清楚**：所谓"AI Agent"，剥掉外壳之后到底剩下什么。

---

## 目录

- [0. 这个项目在干什么](#0-先说人话这个项目在干什么)
- [1. 为什么要手写一个（而不是直接上 LangChain）](#1-为什么要手写一个而不是直接上-langchain)
- [2. 灵感来源：ReAct 论文](#2-灵感来源react-论文)
- [3. 一次真实运行长什么样](#3-一次真实运行长什么样)
- [4. 架构说明（10 行）](#4-架构说明10-行)
- [5. 快速开始](#5-快速开始)
- [6. 实测结果：工具到底值不值得用](#6-实测结果工具到底值不值得用)
- [7. 我踩过的坑与几个反直觉的发现](#7-我踩过的坑与几个反直觉的发现)
- [8. 这个项目**不是**什么（能力边界）](#8-这个项目不是什么能力边界)
- [9. 后续计划](#9-后续计划)
- [10. 参考资料](#10-参考资料)

---

## 0. 这个项目在干什么

如果你也是刚开始接触大模型，可以先这样理解：

**普通用法**（就是你平时在对话框里问问题）：

```
你 → 问题 → 模型 → 答案
```

模型只能靠"脑子里记过的东西"回答。你问它"今天北京几度"、"帮我算 1234×5678"，它要么说不知道，要么**编一个看起来很合理的答案**（这叫**幻觉 / hallucination**）。

**这个项目做的事**，是给模型配上一双手：

```
你 → 问题 → 模型想一下 → 决定用一个工具 → 工具真的去执行 → 结果喂回给模型 → 模型再想 → … → 答案
```

模型不再自己硬答，而是可以**先想、再去查/去算、看完结果接着想**，直到它有把握了才作答。这个"想一步 → 做一步 → 看结果"的循环，就叫 **ReAct**（Reasoning + Acting）。

打个比方：**没有工具的模型像一个记性很好但关在屋子里的人**，你问什么它都凭记忆答，记错了也照样理直气壮；**有工具的模型像一个能用搜索引擎和计算器的实习生**，它会说"这个我查一下"、"这个我算一下"，然后再回答你。

**几个名词（给同样是新手的人）**

| 名词 | 一句话解释 |
| --- | --- |
| LLM | 大语言模型，本项目里就是那个"负责思考和说话的脑子" |
| Prompt | 你喂给模型的文本，包含让它扮演什么角色、必须遵守什么格式 |
| Tool | 一个普通函数，字符串进、字符串出。本项目有 `calculator` 和 `web_search` 两个 |
| Agent | 模型 + 工具 + 让它俩串起来的那个循环 |
| Loop / Harness | 我自己写的那层"驱动程序"：发请求、解析输出、调工具、把结果塞回对话 |
| Trace | 一次运行的完整轨迹（每一步的 Thought / Action / Observation），复盘和写博客主要靠它 |
| 幻觉 | 模型编造出格式正确但事实错误的内容，比如把 `699019262795` 答成 `699000000000` |

---

## 1. 为什么要手写一个（而不是直接上 LangChain）

LangChain、LlamaIndex 这类框架里，一行 `initialize_agent(...)` 就能得到一个 Agent，非常方便。但对刚学习此领域的我来说，那样只是**换了一个更高级的黑盒**：出问题时我根本不知道是提示词写错了、解析逻辑有 bug、还是工具本身返回了空。

所以我选择不用框架，自己把那层循环写出来。写完最大的收获是——**Agent 的神秘感消失了**，它其实就是：

> 一个 `for` 循环 + 一个字符串模板 + 几个正则 + 几个普通函数。

框架帮你做的主要是"工程化"（重试、流式、记忆、多工具选择、可观测性），而**核心机制简单到可以一个下午写完**。先手写一遍，再去用框架，才知道框架到底在替你做什么。

---

## 2. 灵感来源：ReAct 论文

这个项目的思路不是我想出来的，来自下面这篇论文——它也是现在几乎所有 Agent 框架的源头。

| 项 | 内容 |
| --- | --- |
| 标题 | **ReAct: Synergizing Reasoning and Acting in Language Models** |
| 作者 | Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao |
| 机构 | Princeton University + Google Research (Brain team) |
| 时间 | 2022 年 10 月首次上传 |
| arXiv | [2210.03629](https://arxiv.org/abs/2210.03629) |
| 会议 | ICLR 2023（Oral） |
| 项目主页 | https://react-lm.github.io |

**论文的核心一句话**：不要让模型"只推理"（Chain-of-Thought）或"只行动"（只调工具），而是让 **Thought（推理）和 Action（行动）交替出现**——推理帮助模型规划下一步干什么、发现上一步为什么失败；行动让模型能够接触外部信息（知识库、搜索引擎、环境），从而修正自己脑子里错误的记忆。

**论文里长这样**（原文示例，思路不是代码）：

```
Question: 除了 Apple Remote，还有哪些设备可以控制 Apple Remote 原本设计的程序？
Thought:  我需要先搜索 Apple Remote，看看它原本是给什么程序设计的。
Action:   Search[Apple Remote]
Observation: Apple Remote 是为了控制 Front Row 媒体中心设计的……
Thought:  现在我需要查一下还有哪些设备支持 Front Row。
Action:   Search[Front Row]
Observation: 支持 Front Row 的只有带红外接收器的 Mac……
Thought:  我已经有足够信息了。
Action:   Finish[只有部分老款 Mac 支持 Front Row，因此没有其他设备可以控制它。]
```

**论文报告的效果**（我只记结论，具体数值请以原文为准）：在 HotpotQA 这类需要多跳事实检索的任务上，ReAct 相比纯 CoT **大幅降低了幻觉率**——因为答案不再全凭记忆，而是有检索结果兜底；在 ALFWorld / WebShop 这两个交互式决策任务上，论文摘要称其比模仿学习和强化学习基线分别高出 **34% / 10% 的绝对成功率**，而且只用了一两个 few-shot 例子，**没有微调任何参数**。

**这个仓库和论文的关系**：论文是"提出范式并用 PaLM-540B / GPT-3 在若干基准上做系统验证"；我这里是**用最小代价把这个范式复刻成一个能跑通的骨架**——换成了任何 OpenAI 兼容接口（我用的是 DeepSeek），工具从"Wikipedia API"缩成了两个玩具工具，也没有做原论文那么多实验。但 `Thought → Action → Observation` 那个循环，是一模一样的。

**顺着这篇往下看的论文**（我自己列的路线，还没读完）：

- **Chain-of-Thought Prompting**（Wei et al., 2022, [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)）— ReAct 的 Thought 部分直接继承自它：让模型"一步步想"。但它只能靠内部知识，所以会一本正经地编。
- **Toolformer**（Schick et al., 2023, [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)）— 换了个方向：不靠提示词，而是让模型**自己学会**什么时候该调用哪个 API。
- **Reflexion**（Shinn et al., 2023, [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)）— 给 Agent 加上"用语言反思自己上一次为什么失败"的机制。
- **Tree of Thoughts**（Yao et al., 2023, [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)）— 把线性的推理链扩展成搜索树，多了"回溯"的能力。
- **SWE-bench**（Jimenez et al., 2023, [arXiv:2310.06770](https://arxiv.org/abs/2310.06770)）— 让模型去修真实 GitHub 仓库的 issue，是我这个方向最相关的评测集（也是我下一步想碰的东西）。

另外值得一提的是：LangChain 里的 `ZeroShotAgent`、AutoGPT 的主循环、以及现在各家 API 的 tool calling，本质上跑的都是这个三元组。

---

## 3. 一次真实运行长什么样

> 下面这段是**格式示意**。跑一遍 `python agent.py "..."`，把你自己的完整输出替换到这里——自己跑出来的输出比任何示例都有说服力。

```text
User: 帮我算一下 (1234*5678)/2，再查一下 ReAct 是什么
模型: deepseek-chat @ https://api.deepseek.com/v1

--- step 1 ---
Thought: 需要分两步：先算数，再查资料。
Action: calculator
Action Input: (1234*5678)/2
Observation: 3503626.0

--- step 2 ---
Thought: 算完了，接下来查 ReAct。
Action: web_search
Action Input: ReAct 论文
Observation: - ReAct: Synergizing Reasoning and Acting ...：ReAct 让模型交替产生推理轨迹和行动……

--- step 3 ---
Thought: 信息已足够
Final Answer: (1234*5678)/2 = 3503626；ReAct 是一种让大模型交替"推理"和"行动"的框架……

调用工具次数: 2
```

一次完整运行里，模型被调用了 **3 次**（step 1、2、3），工具被执行了 **2 次**。这个"LLM 调用次数 ≈ 工具调用次数 + 1"的关系，正是第 6 节里讨论成本问题的依据。

---

## 4. 架构说明（10 行）

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

**目录结构**

```
react-agent-demo/
├── agent.py              # 主循环：提示词、解析、驱动、配置自检
├── tools.py              # 工具：calculator（ast 安全求值）+ web_search（Bing→Wikipedia 兜底）
├── test_offline.py       # 离线自测，不联网不花额度
├── bench_arithmetic.py   # 对照实验脚本：直接问 vs 走 Agent
├── bench/                # 实验产出（accuracy.csv / summary.csv / raw_results.jsonl）
├── .env.example          # 配置模板（五种端点示例）
└── requirements.txt
```

---

## 5. 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python test_offline.py
python agent.py "帮我算一下 (1234*5678)/2，再查一下 SWE-bench 是什么"
```

> Windows 上的激活命令是 `.venv\Scripts\activate`。虚拟环境的名字随你起（`.venv` / `venv` / `.venv_xxx` 都行），只要 `venv` 和 `source` 两条命令里的名字一致即可。

`cp` 之后打开 `.env` 填入自己的 API key —— 示例文件里列了 OpenAI / DeepSeek / 智谱 / 通义 / 本地 Ollama 五种端点，换模型只改这里三行，不动代码。`test_offline.py` 不联网、不花额度，用来先确认代码本身没问题。

---

## 6. 实测结果：工具到底值不值得用

`bench_arithmetic.py` 是一组对照实验：同一批题分别"直接问模型"和"走 Agent 调 calculator"，两种模式各 140 题，`temperature=0`，除法题由乘法构造以保证结果为精确整数。

| 参与运算的位数 | 直接问模型（乘法 / 除法） | 走 Agent（乘法 / 除法） |
| ------- | ------------- | --------------- |
| 2–3 位   | 100% / 100%   | 100% / 100%     |
| 4 位     | 60% / 90%     | 100% / 100%     |
| 5 位     | 50% / 60%     | 100% / 100%     |
| 6–8 位   | 0% / 0%       | 100% / 100%     |

**结论**：模型自算的能力边界落在 **4～6 位之间**；超过 6 位它不是在计算，而是在"续写"一个看起来像答案的数字串（典型例子：把 `699019262795` 答成 `699000000000`）。所以 calculator 的价值不是"让模型更聪明"，而是**把这一步从模型手里拿走、交给确定性的 Python**——代价是每题多一次 LLM 调用（实测约 2 倍成本，不是想当然的 3 倍）。

原始数据与判分脚本都在 `bench/` 下，可复现。

---

## 7. 我踩过的坑与几个反直觉的发现

这一节是我觉得比代码本身更有价值的部分。

**① `.env` 不是环境变量。**
`os.getenv()` 读的是系统环境变量，它**不会**自己去读 `.env` 文件。所以必须先解析 `.env` 并把键值对灌进 `os.environ`，后面才读得到。真实项目里用 `python-dotenv` 一行搞定，但原理就是这个——我为了不引额外依赖，在 `agent.py` 里手写了 12 行。

**② 提示词不是权限边界。**
我在系统提示词里写了"涉及数字计算一律用 calculator，不要自己心算"。结果在对照实验里，模型**照样自己心算**了。教训：**提示词是"软约束"，代码才是"硬边界"**。如果你真的不允许某个行为，就不能指望用一句"请不要"实现，而要让它**做不到**（比如工具只暴露一个经过校验的接口，或者干脆把那条路径从代码里删掉）。这一点在做 Agent 安全时是个很关键的心智模型。

**③ 工具的可用性是一等公民。**
我一开始的 `web_search` 只用 Wikipedia API，结果在国内网络下直接连不通（HTTP 000）。修法不是"换个问法"，而是给检索加**引擎兜底链**（Bing → Wikipedia），并且对返回的 HTML 做实体还原（`&ensp;` `&#0183;` 这类）。**工具挂掉时要返回一句"我查不到"，而不是抛异常**——因为异常会让整个循环崩掉，而一句"查不到"会作为 Observation 回灌给模型，它自己会想办法。

**④ 报错也是一种 Observation。**
我故意让模型去调一个不存在的工具（比如 `google`），它收到的不是崩溃，而是 `错误：不存在工具 google，可用工具为 [...]`，然后它**自己换成了 `web_search`**。这就是 a) 里那个"格式错就把报错回灌给模型自纠"机制的真正价值：**让错误变成环境反馈的一部分**，而不是流程的终点。`test_offline.py` 里专门有一个用例覆盖这种情况。

**⑤ 我的一次预测被实验打脸了。**
我原以为"不调工具的模型做两位数乘法一定会错"。实测发现，对当前这一代模型来说，2–3 位乘法它 100% 正确。**是我的先验错了，不是模型不行**。所以对模型能力的判断不能靠"印象"，要做对照实验；而且**单条样本不构成证据**（我一开始只测了一道题就下结论，是被指出来的）。

---

## 8. 这个项目**不是**什么（能力边界）

写清楚边界，比吹嘘功能更诚实：

- **不是框架，也不是产品**：它只处理单轮问答，没有持久化、没有并发、没有 UI。
- **没有记忆**：每次运行都是从零开始的对话，上一轮的历史不会被保留。
- **没有真正的多步规划**：`MAX_STEPS=6` 是一根保险丝，不是 Planner。复杂任务它会直接撞上限。
- **只有两个工具，且都很弱**：搜索是抓 Bing 结果页 HTML（正则匹配），不稳定、可能被改版打断；计算器只支持四则 / 取模 / 乘方。
- **不做权限控制**：`calculator` 用了 `ast` 白名单求值而不是 `eval()`，能拦住 `1+1; import os` 这种表达式，但这只是"防呆"，不是沙箱。
- **没有复现原论文的实验**：论文在 HotpotQA / Fever / ALFWorld / WebShop 上的评测，我这里一件都没做。

---

## 9. 后续计划

- [ ] 把"一次真实运行"替换成本机我自己跑出来的 trace
- [ ] 工具调用失败时的重试策略（现在是一次不成就换引擎）
- [ ] 加一个 `python_repl` 工具，体验一下"工具变强之后 Agent 的能力边界在哪里移动"
- [ ] 想清楚 **ReAct 循环在真实软件工程任务（SWE-bench 那类 issue 修复）上会卡在哪一步**
- [ ] 回头看框架：同样的事在 LangChain 里怎么写，它替我做了什么、又隐藏了什么

---

## 10. 参考资料

**论文**

- Yao et al. **ReAct: Synergizing Reasoning and Acting in Language Models.** ICLR 2023. [arXiv:2210.03629](https://arxiv.org/abs/2210.03629) · [项目主页](https://react-lm.github.io)
- Wei et al. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models.** NeurIPS 2022. [arXiv:2201.11903](https://arxiv.org/abs/2201.11903)
- Schick et al. **Toolformer: Language Models Can Teach Themselves to Use Tools.** 2023. [arXiv:2302.04761](https://arxiv.org/abs/2302.04761)
- Shinn et al. **Reflexion: Language Agents with Verbal Reinforcement Learning.** 2023. [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)
- Yao et al. **Tree of Thoughts: Deliberate Problem Solving with Large Language Models.** 2023. [arXiv:2305.10601](https://arxiv.org/abs/2305.10601)
- Jimenez et al. **SWE-bench: Can Language Models Resolve Real-World GitHub Issues?** 2023. [arXiv:2310.06770](https://arxiv.org/abs/2310.06770)

**代码**

- 原论文官方实现：[github.com/ysymyth/ReAct](https://github.com/ysymyth/ReAct)
- LangChain 里的对应实现：`langchain.agents.ZeroShotAgent`

---

*这是一个学习性质的项目，代码以可读性优先，欢迎指出任何概念上的错误。*
