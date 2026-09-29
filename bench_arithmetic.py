"""算术能力边界实验：把「直接问模型」和「带工具的 ReAct Agent」放到同一批题上对比。

为什么需要它
------------
你手头那两组对照只跑了 1 道题，n=1 既不能证明"工具没用"，也不能证明"模型会算"。
这个脚本把样本量拉到几十上百题，把"我猜"变成"我测出来"。

设计上刻意保证公平与可复现，四条：

1. 两种模式用**完全相同的题目、相同的后缀约束、相同的 temperature=0**，
   唯一的变量就是「有没有工具」。
2. 除法题全部由乘法构造（先算 a*b，再问 (a*b)/b），**商保证是精确整数**。
   这样判分是 100% 客观的数值比较，不会陷入"3.33 算不算对"的争议。
3. 每道题的正确答案由 Python 自己算（Decimal 精确比较），不靠任何模型。
4. 结果实时写入 JSONL 缓存，中断后重跑会**自动跳过已完成的题**，不重复花钱。
   想重测某一批就删掉 outdir 里的 raw_results.jsonl。

用法
----
    python bench_arithmetic.py --dry-run                        # 只出题，不花额度
    python bench_arithmetic.py --digits 2,3 --per-digit 3        # 小样本试跑
    python bench_arithmetic.py                                   # 正式跑（约 140 题）
    python bench_arithmetic.py --digits 2,3,4,5 --modes agent    # 只跑带工具那组

跑完看两个地方：
    bench/summary.csv      每道题的原始记录（含模型原话），用来人工复核判分
    bench/accuracy.csv     按位数档汇总的准确率，直接导进 Excel 画曲线
"""
import argparse
import csv
import decimal
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agent import OpenAIChat, check_config, run_agent  # noqa: E402

DIRECT_SYSTEM = (
    "你是一个数学助手。只输出计算结果本身，"
    "不要解释、不要写计算步骤、不要复述题目。"
)
SUFFIX = "（只输出最终数字，不要写过程）"

# 允许千分位逗号与科学计数法；Decimal 能精确比较 3503326.0 和 3503326
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:[eE][+-]?\d+)?")


# --------------------------------------------------------------------------
# 出题
# --------------------------------------------------------------------------
def generate_items(digits_list, per_digit, seed):
    """每个位数档生成 per_digit 道乘法 + per_digit 道整除题。

    operand_digits 是**参与运算的两个数的位数**：
      mul  -> a(d 位) × b(d 位)，答案约 2d 位
      div  -> (a*b)(2d 位) ÷ b(d 位)，答案恰好 d 位
    两条曲线按 operand_digits 对齐，但答案长度不同，所以 CSV 里两个都记，
    你想按哪个轴画都行。
    """
    rnd = random.Random(seed)
    items = []
    for d in digits_list:
        lo, hi = 10 ** (d - 1), 10 ** d - 1
        for i in range(1, per_digit + 1):
            a = rnd.randint(lo, hi)
            b = rnd.randint(lo, hi)
            items.append({
                "id": "mul-d%d-%02d" % (d, i),
                "kind": "mul",
                "operand_digits": d,
                "answer_digits": len(str(a * b)),
                "question": "%d × %d 等于多少？" % (a, b),
                "answer": str(a * b),
            })
            items.append({
                "id": "div-d%d-%02d" % (d, i),
                "kind": "div",
                "operand_digits": d,
                "answer_digits": len(str(a)),
                "question": "%d ÷ %d 等于多少？" % (a * b, b),
                "answer": str(a),
            })
    return items


# --------------------------------------------------------------------------
# 判分
# --------------------------------------------------------------------------
def numbers_in(text):
    out = []
    for m in NUM_RE.finditer(text or ""):
        try:
            out.append(decimal.Decimal(m.group(0).replace(",", "")))
        except decimal.InvalidOperation:
            continue
    return out


def judge(text, expected):
    """返回 (是否正确, 说明)。只有「最后一个数字 == 正确答案」才算对。

    取最后一个数字是因为模型习惯把结论放末尾；同时会给"答案出现在中间但
    最后说错"的情况单独打标，方便你复盘时区分"算错"和"抄过程写错结论"。
    """
    nums = numbers_in(text)
    if not nums:
        return False, "未找到任何数字"
    exp = decimal.Decimal(expected)
    if nums[-1] == exp:
        return True, ""
    if any(n == exp for n in nums):
        return False, "答案出现在中间但末位不是它"
    return False, "数值不等"


# --------------------------------------------------------------------------
# 两种模式
# --------------------------------------------------------------------------
class DirectChat:
    """对照组：同一张卷子，但只有一次模型调用、没有任何工具。"""

    def __init__(self, temperature=0.0):
        from openai import OpenAI

        self.model = os.getenv("MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=(os.getenv("OPENAI_BASE_URL") or None),
        )

    def ask(self, question):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": DIRECT_SYSTEM},
                {"role": "user", "content": question},
            ],
            temperature=self.temperature,
        )
        return resp.choices[0].message.content or ""


def run_one(item, mode, direct, retries):
    """跑一道题，返回一条记录。异常不会中断整轮，只记进记录里。"""
    question = item["question"] + SUFFIX
    last_err = ""
    for attempt in range(retries + 1):
        try:
            t0 = time.time()
            if mode == "direct":
                raw = direct.ask(question)
                tool_calls, llm_calls, calls_note = 0, 1, ""
            else:
                raw, trace = run_agent(question, OpenAIChat(), verbose=False)
                tool_calls = sum(1 for t in trace if t["action"])
                llm_calls = len(trace) + 1
                calls_note = " ; ".join(
                    "%s(%s)" % (t["action"], str(t["input"])[:30])
                    for t in trace if t["action"]
                )
            seconds = round(time.time() - t0, 1)
            ok, why = judge(raw, item["answer"])
            return {
                "key": "%s|%s" % (item["id"], mode),
                "id": item["id"],
                "kind": item["kind"],
                "operand_digits": item["operand_digits"],
                "answer_digits": item["answer_digits"],
                "mode": mode,
                "question": item["question"],
                "expected": item["answer"],
                "raw_answer": (raw or "").replace("\n", " ").strip()[:200],
                "ok": ok,
                "why": why,
                "tool_calls": tool_calls,
                "llm_calls": llm_calls,
                "tools_used": calls_note,
                "seconds": seconds,
                "error": "",
            }
        except Exception as exc:  # 网络抖动、限流、余额不足都走这里
            last_err = "%s: %s" % (type(exc).__name__, exc)
            if attempt < retries:
                time.sleep(2 * (attempt + 1))
    return {
        "key": "%s|%s" % (item["id"], mode),
        "id": item["id"],
        "kind": item["kind"],
        "operand_digits": item["operand_digits"],
        "answer_digits": item["answer_digits"],
        "mode": mode,
        "question": item["question"],
        "expected": item["answer"],
        "raw_answer": "",
        "ok": None,
        "why": "调用失败",
        "tool_calls": 0,
        "llm_calls": 0,
        "tools_used": "",
        "seconds": 0.0,
        "error": last_err[:200],
    }


# --------------------------------------------------------------------------
# 缓存与输出
# --------------------------------------------------------------------------
FIELDS = [
    "id", "kind", "operand_digits", "answer_digits", "mode", "question",
    "expected", "raw_answer", "ok", "why", "tool_calls", "llm_calls",
    "tools_used", "seconds", "error",
]


def load_cache(path):
    cache = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("key"):
                    cache[rec["key"]] = rec
    return cache


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(records):
    """按 (位数档, 题型) 汇总两模式的准确率与耗时。"""
    buckets = {}
    for rec in records:
        key = (rec["operand_digits"], rec["kind"])
        b = buckets.setdefault(key, {})
        m = b.setdefault(rec["mode"], {"n": 0, "ok": 0, "sec": 0.0, "fail": 0})
        m["n"] += 1
        if rec["ok"] is True:
            m["ok"] += 1
        elif rec["ok"] is None:
            m["fail"] += 1
        m["sec"] += rec["seconds"] or 0.0
    return buckets


def print_report(buckets, modes):
    print("\n" + "=" * 78)
    print("准确率对照表（按参与运算的两个数的位数分档）")
    print("=" * 78)
    header = "%-8s %-5s %-5s" % ("档位", "题型", "题数")
    for m in modes:
        header += "  %-14s" % ("%s 正确" % m)
    print(header)
    print("-" * 78)
    rows = []
    for (digits, kind) in sorted(buckets):
        b = buckets[(digits, kind)]
        n = max(v["n"] for v in b.values())
        line = "%-8s %-5s %-5d" % ("d=%d" % digits, kind, n)
        row = {"operand_digits": digits, "kind": kind, "n": n}
        for m in modes:
            v = b.get(m, {"ok": 0, "n": 1, "sec": 0.0, "fail": 0})
            pct = 100.0 * v["ok"] / max(v["n"], 1)
            line += "  %-14s" % ("%d/%d (%.0f%%)" % (v["ok"], v["n"], pct))
            row["%s_ok" % m] = v["ok"]
            row["%s_n" % m] = v["n"]
            row["%s_acc" % m] = round(pct, 1)
            row["%s_avg_sec" % m] = round(v["sec"] / max(v["n"], 1), 1)
            row["%s_error" % m] = v["fail"]
        print(line)
        rows.append(row)

    print("-" * 78)
    totals = {}
    for m in modes:
        n = sum(r["%s_n" % m] for r in rows)
        ok = sum(r["%s_ok" % m] for r in rows)
        err = sum(r["%s_error" % m] for r in rows)
        totals[m] = (ok, n, err)
    line = "%-8s %-5s %-5s" % ("总计", "", "")
    for m in modes:
        ok, n, err = totals[m]
        line += "  %-14s" % ("%d/%d (%.0f%%)" % (ok, n, 100.0 * ok / max(n, 1)))
    print(line)
    for m in modes:
        ok, n, err = totals[m]
        if err:
            print("  ! %s 有 %d 次调用失败，未计入正确/错误" % (m, err))
    return rows


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="对比「直接问模型」与「ReAct Agent」的算术准确率",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--digits", default="2,3,4,5,6,7,8",
                    help="参与运算的两个数的位数，逗号分隔")
    ap.add_argument("--per-digit", type=int, default=10,
                    help="每个档位、每种题型的题数")
    ap.add_argument("--modes", default="direct,agent",
                    help="要跑的模式：direct / agent / 两个用逗号连起来")
    ap.add_argument("--seed", type=int, default=20260929, help="随机种子，固定可复现")
    ap.add_argument("--limit", type=int, default=0, help="最多跑多少道题（试跑用，0 = 不限）")
    ap.add_argument("--max-steps", type=int, default=6, help="agent 模式的最大步数")
    ap.add_argument("--retries", type=int, default=1, help="单题失败后的重试次数")
    ap.add_argument("--out", default="", help="结果目录，默认脚本同级的 bench/")
    ap.add_argument("--dry-run", action="store_true", help="只出题打印，不调用模型")
    args = ap.parse_args()

    digits = [int(x) for x in args.digits.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in ("direct", "agent"):
            print("未知模式 %r，只能是 direct 或 agent" % m)
            raise SystemExit(1)

    items = generate_items(digits, args.per_digit, args.seed)
    if args.limit:
        items = items[:args.limit]

    if args.dry_run:
        print("模式: %s   种子: %d   题数: %d" % (",".join(modes), args.seed, len(items)))
        print("-" * 78)
        for it in items:
            print("%-14s %-4s %-34s = %s" % (
                it["id"], it["kind"], it["question"], it["answer"]))
        print("-" * 78)
        print("dry-run 结束，没有调用任何模型。")
        return

    check_config()
    print("模型: %s @ %s" % (os.getenv("MODEL", "?"), os.getenv("OPENAI_BASE_URL", "(默认)")))
    est_calls = len(items) * (3 if "agent" in modes else 0) + len(items) * (1 if "direct" in modes else 0)
    print("模式: %s   题数: %d   预计模型调用: 约 %d 次" % (",".join(modes), len(items), est_calls))

    outdir = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench")
    os.makedirs(outdir, exist_ok=True)
    cache_path = os.path.join(outdir, "raw_results.jsonl")
    cache = load_cache(cache_path)
    if cache:
        print("已加载 %d 条历史记录，命中的题会跳过" % len(cache))

    direct = DirectChat() if "direct" in modes else None
    records = []
    for idx, item in enumerate(items, 1):
        for mode in modes:
            key = "%s|%s" % (item["id"], mode)
            if key in cache:
                rec = cache[key]
                tag = "缓存"
            else:
                rec = run_one(item, mode, direct, args.retries)
                if rec["ok"] is not None:
                    cache[key] = rec
                    with open(cache_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fh.flush()
                tag = "新跑"
            records.append(rec)
            mark = {True: "对", False: "错", None: "失败"}[rec["ok"]]
            print("[%3d/%3d] %-14s %-4s %-9s %-4s 期望=%-12s 得到=%-20s %s" % (
                idx, len(items), rec["id"], rec["kind"], mode, mark,
                rec["expected"], rec["raw_answer"][:20] or "(空)", tag))

    write_csv(os.path.join(outdir, "summary.csv"), records, FIELDS)
    rows = print_report(summarize(records), modes)
    acc_fields = ["operand_digits", "kind", "n"]
    for m in modes:
        acc_fields += ["%s_ok" % m, "%s_n" % m, "%s_acc" % m, "%s_avg_sec" % m, "%s_error" % m]
    write_csv(os.path.join(outdir, "accuracy.csv"), rows, acc_fields)

    print("\n明细: %s" % os.path.join(outdir, "summary.csv"))
    print("汇总: %s" % os.path.join(outdir, "accuracy.csv"))
    print("说明: summary.csv 里保存了模型原话，判分有疑问时人工复核这一列。")


if __name__ == "__main__":
    main()
