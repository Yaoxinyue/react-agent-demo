"""离线自测：用脚本化的假 LLM 验证 ReAct 循环，不花 API 额度、不需要网络。

    python test_offline.py
"""
from agent import run_agent
from tools import calculator, web_search


class ScriptedChat:
    """按剧本返回内容的假客户端，接口和 OpenAIChat 一致。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def chat(self, messages):
        self.calls.append([dict(m) for m in messages])  # 快照，messages 后续会被原地修改
        # 剧本用完后重复最后一句，模拟模型一直输出不合格式的内容
        return self.script.pop(0) if len(self.script) > 1 else self.script[0]


def test_calculator():
    assert calculator("(3+5)*7") == "56"
    assert calculator("2^10") == "1024"
    assert calculator("1+1; import os").startswith("计算失败")
    print("[ok] calculator 正常且拒绝非法表达式")


def test_single_tool_loop():
    script = [
        "Thought: 需要计算\nAction: calculator\nAction Input: (3+5)*7",
        "Thought: 已得到结果\nFinal Answer: 56",
    ]
    fake = ScriptedChat(script)
    answer, trace = run_agent("(3+5)*7 等于多少？", fake, verbose=False)
    assert answer == "56", answer
    assert len(trace) == 1 and trace[0]["observation"] == "56"
    assert "Observation: 56" in fake.calls[1][-1]["content"]
    print("[ok] 单工具循环：Action 被解析、工具被调用、Observation 回灌")


def test_unknown_tool_and_recovery():
    script = [
        "Thought: 查一下\nAction: google\nAction Input: ReAct",
        "Thought: 换个工具\nAction: web_search\nAction Input: ReAct 论文",
        "Thought: 够了\nFinal Answer: ReAct 是 Reasoning + Acting 的循环框架。",
    ]
    answer, trace = run_agent("什么是 ReAct？", ScriptedChat(script), verbose=False)
    assert "ReAct" in answer
    assert "不存在工具 google" in trace[0]["observation"]
    print("[ok] 工具名幻觉 -> 报错回灌 -> 模型自行纠错")


def test_max_steps_guard():
    answer, trace = run_agent("问题", ScriptedChat(["Thought: 卡住了"]), max_steps=2, verbose=False)
    assert "最大步数" in answer
    assert len(trace) == 2
    print("[ok] 死循环保护：超步数后正常退出")


if __name__ == "__main__":
    test_calculator()
    test_single_tool_loop()
    test_unknown_tool_and_recovery()
    test_max_steps_guard()
    print("\n全部通过。接下来配好 .env 跑真实模型：python agent.py \"你的问题\"")
