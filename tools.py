"""ReAct Agent 的工具集：计算器 + 搜索。

设计要点：工具只是一个「字符串进、字符串出」的普通 Python 函数，
Agent 循环完全不关心它内部怎么实现。加新工具 = 写一个函数 + 注册到 TOOLS。
"""
import ast
import html
import json
import operator
import re
import urllib.parse
import urllib.request

_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("不支持的表达式片段：%s" % type(node).__name__)


def calculator(expression: str) -> str:
    """安全计算器：用 ast 解析，只允许四则运算/取模/乘方，绝不调用 eval()。"""
    expr = (
        expression.strip()
        .replace("^", "**")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("，", ",")
        .replace("=", "")
    )
    try:
        return str(_eval_node(ast.parse(expr, mode="eval").body))
    except Exception as exc:
        return "计算失败：%s（请只输入纯算式，例如 (3+5)*7）" % exc


_SEARCH_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def _clean(html_text: str) -> str:
    """去标签 + 还原 HTML 实体（&ensp; &#0183; 之类）+ 压缩空白。"""
    text = re.sub(r"<[^>]+>", "", html_text)
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _search_bing(query: str, top_k: int) -> str:
    """Bing 网页检索：国内可直连，抓结果页 HTML 里的标题 + 摘要。"""
    url = "https://cn.bing.com/search?q=%s&setlang=zh-CN" % urllib.parse.quote(query)
    req = urllib.request.Request(
        url, headers={"User-Agent": _SEARCH_UA, "Accept-Language": "zh-CN,zh;q=0.9"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        html = resp.read().decode("utf-8", "ignore")

    lines = []
    for block in re.findall(r'<li class="b_algo".*?</li>', html, re.S)[:top_k]:
        head = re.search(r'<h2[^>]*>\s*<a[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not head:
            continue
        title = _clean(head.group(2))
        para = re.search(r"<p[^>]*>(.*?)</p>", block, re.S)
        snippet = _clean(para.group(1)) if para else ""
        if not snippet:
            continue
        lines.append("- %s：%s" % (title[:60], snippet[:150]))
    return "\n".join(lines)


def _search_wikipedia(query: str, top_k: int) -> str:
    """Wikipedia 检索：海外网络下可用，作为备用引擎。"""
    for lang in ("zh", "en"):
        try:
            url = (
                "https://%s.wikipedia.org/w/api.php?action=query&list=search"
                "&format=json&srlimit=%d&srsearch=%s"
                % (lang, top_k, urllib.parse.quote(query))
            )
            req = urllib.request.Request(url, headers={"User-Agent": "react-agent-demo/1.0"})
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.load(resp)
            hits = data.get("query", {}).get("search", [])
            if hits:
                return "\n".join(
                    "- %s：%s" % (h["title"], re.sub(r"<[^>]+>", "", h["snippet"]))
                    for h in hits
                )
        except Exception:
            continue
    return ""


def web_search(query: str, top_k: int = 3) -> str:
    """联网检索：依次尝试 Bing → Wikipedia，谁先返回结果就用谁。

    想换成 Tavily / SerpAPI / Bing 官方 API，只要保持 `函数名(query) -> str` 的签名即可。
    """
    for engine in (_search_bing, _search_wikipedia):
        try:
            result = engine(query, top_k)
        except Exception:
            continue
        if result:
            return result[:1200]
    return "搜索无结果（网络不可达或该词条不存在），可以换关键词或直接用已有信息作答。"



TOOLS = {
    "calculator": calculator,
    "web_search": web_search,
}

TOOL_DESCRIPTIONS = [
    ("calculator", "数学计算。输入纯算式，例如 (1234*5678)/2"),
    ("web_search", "检索事实性知识。输入一个搜索关键词短句"),
]
