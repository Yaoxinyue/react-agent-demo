"""最小可用的 ReAct Agent：Thought -> Action -> Observation 循环。

兼容任何 OpenAI 格式接口：OpenAI、DeepSeek、智谱 GLM、通义千问、
Moonshot、本地 Ollama，以及各类 Claude 网关。换模型只改 .env，不动代码。

用法：
    python agent.py "帮我算一下 (1234*5678)/2，再查一下什么是 ReAct"
"""
import os
import re
import sys

from tools import TOOLS, TOOL_DESCRIPTIONS


def load_dotenv(path=None):
    """极简 .env 读取器（只用标准库，不额外装包）。

    os.getenv() 读的是「系统环境变量」，它并不会自动去读 .env 文件——
    所以必须先把 .env 里的键值对灌进 os.environ，后面的 os.getenv 才拿得到。
    真实项目里通常用 python-dotenv 库（`load_dotenv()` 一行），原理与此相同。
    """
    env_path = path or os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.exists(env_path):
        return False
    with open(env_path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip().removeprefix("export ").strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:  # 已存在的环境变量优先，不被文件覆盖
                os.environ[key] = value
    return True


load_dotenv()  # 必须在下面读取任何配置之前调用

MAX_STEPS = int(os.getenv("MAX_STEPS", "6"))

SYSTEM_PROMPT = """你是一个 ReAct 风格的智能体：面对问题时交替进行「思考」和「行动」。

可用工具：
{tools}

输出格式（严格遵守，每次只输出一步）：

Thought: 你的推理过程（中文，简短）
Action: 工具名，只能是 [{tool_names}]
Action Input: 传给工具的参数，纯文本，不要写解释

当你已经可以回答时，改用：

Thought: 信息已足够
Final Answer: 最终答案（中文）

硬性规则：
1. 一次只调用一个工具，必须等到 Observation 之后才能进入下一步。
2. Action Input 里只写参数本身，例如 `(1234*5678)/2` 或 `ReAct 论文`。
3. 涉及数字计算一律用 calculator，不要自己心算。
4. 不要在 Thought 里编造工具返回的内容。
"""

FINAL_RE = re.compile(r"Final\s*Answer\s*[:：]\s*(.+)", re.S)
ACTION_RE = re.compile(r"Action\s*[:：]\s*([A-Za-z_][A-Za-z0-9_]*)")
INPUT_RE = re.compile(
    r"Action\s*Input\s*[:：]\s*(.*?)(?=\n\s*(?:Thought|Action|Observation|Final\s*Answer)\s*[:：]|\Z)",
    re.S,
)


class OpenAIChat:
    """真实模型调用：任何 OpenAI 兼容端点都能用。"""

    def __init__(self, model=None, base_url=None, api_key=None, temperature=0.0):
        from openai import OpenAI  # 延迟导入，方便离线测试时不需要装依赖

        self.model = model or os.getenv("MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=(base_url or os.getenv("OPENAI_BASE_URL") or None),
        )

    def chat(self, messages):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content


def build_system_prompt():
    desc = "\n".join("- %s：%s" % (name, text) for name, text in TOOL_DESCRIPTIONS)
    return SYSTEM_PROMPT.format(tools=desc, tool_names=", ".join(TOOLS))


def run_agent(question, client, max_steps=MAX_STEPS, verbose=True):
    """核心循环：LLM 输出 -> 解析 -> 执行工具 -> 回灌 Observation -> 再问 LLM。

    返回 (最终答案, 轨迹列表)；轨迹用来复盘，也是写博客的一手素材。
    """
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": question},
    ]
    trace = []

    for step in range(1, max_steps + 1):
        reply = client.chat(messages)
        messages.append({"role": "assistant", "content": reply})
        if verbose:
            print("\n--- step %d ---\n%s" % (step, reply.strip()))

        final = FINAL_RE.search(reply)
        if final:
            answer = final.group(1).strip()
            if verbose:
                print("\n=== Final Answer ===\n%s" % answer)
            return answer, trace

        action, action_input = ACTION_RE.search(reply), INPUT_RE.search(reply)
        if action and action_input:
            name = action.group(1).strip()
            arg = action_input.group(1).strip().strip("`\"' “”")
            if name in TOOLS:
                observation = TOOLS[name](arg)
            else:
                observation = "错误：不存在工具 %s，可用工具为 %s" % (name, list(TOOLS))
        else:
            name, arg = None, None
            observation = "格式错误：必须同时给出 Action 和 Action Input，或直接给出 Final Answer。"

        if verbose:
            print("Observation: %s" % observation)
        trace.append({"step": step, "action": name, "input": arg, "observation": observation})
        messages.append({"role": "user", "content": "Observation: %s" % observation})

    return "达到最大步数（%d）仍未得到 Final Answer，请检查工具是否返回了有效信息。" % max_steps, trace


def check_config():
    """跑真实模型前的自检：把最常见的两种环境问题提前说清楚。"""
    if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
        print("找不到 .env 文件。请先执行：cp .env.example .env，然后在里面填 OPENAI_API_KEY。")
        raise SystemExit(1)
    key = os.getenv("OPENAI_API_KEY") or ""
    if not key or key.startswith("sk-your"):
        print(".env 里的 OPENAI_API_KEY 还是占位值，请填入真实 key。")
        raise SystemExit(1)


def main():
    if len(sys.argv) < 2:
        print('用法: python agent.py "你的问题"')
        raise SystemExit(1)
    question = " ".join(sys.argv[1:])
    check_config()
    print("User: %s" % question)
    print("模型: %s @ %s" % (os.getenv("MODEL", "gpt-4o-mini"), os.getenv("OPENAI_BASE_URL", "(默认)")))
    answer, trace = run_agent(question, OpenAIChat())
    print("\n调用工具次数: %d" % len(trace))


if __name__ == "__main__":
    main()
