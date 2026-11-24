"""
问答基准测试：用真实 OpenAI + 本地 secrets 里（测试）Sheet 的 qa_bank 跑一组典型问题，
打印每题的回答、耗时、会写入哪个 tab。写 Sheet 和后台爬虫都被替换成假的，不产生任何记录。

用法（项目根目录）：
  .venv/bin/python scripts/bench_answers.py                 # 测当前代码
  .venv/bin/python scripts/bench_answers.py <代码目录> <标签>  # 例如对比 main 的 git worktree
结果另存为 scripts/bench_<标签>.json（已 gitignore）。会消耗少量 OpenAI 额度。
"""
import os, sys, types, time, tomllib, json
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REPO = sys.argv[1] if len(sys.argv) > 1 else str(ROOT)
LABEL = sys.argv[2] if len(sys.argv) > 2 else "current"
sys.path.insert(0, REPO)
os.chdir(REPO)

secrets = tomllib.load(open(ROOT / ".streamlit" / "secrets.toml", "rb"))

import sheets as real_sheets
logs = []
real_sheets.append_pending_row = lambda *a, **k: logs.append("pending")
real_sheets.append_unanswered_row = lambda *a, **k: logs.append("unanswered:" + str(a[-1])[:40])
real_sheets.append_signup_row = lambda *a, **k: (True, "")
scr = types.ModuleType("scraper"); scr.rebuild_index = lambda *a, **k: False
sys.modules["scraper"] = scr

from streamlit.testing.v1 import AppTest

CASES = [
    ("中文(繁)", "什麼是財務證明", "应回答：银行存款证明"),
    ("中文(简)", "神学硕士（ThM）项目的申请费和学费是多少？", "应回答 ThM 学费"),
    ("中文(简)", "道学硕士（MDiv）项目的申请截止时间是什么时候？", "不能用 ThM 的截止时间"),
    ("中文(简)", "本地学生可以申请新生奖学金吗", "不能用国际学生的答案"),
    ("中文(简)", "宿舍可以养宠物吗", "库里没有，不能乱答"),
    ("English", "How much is the tuition per credit?", "应回答每学分学费"),
    ("中文(简)", "学校的地址在哪里", "应回答地址"),
    ("中文(简)", "图书馆周末几点开门", "大概率答不了"),
]

def ask(lang, q):
    at = AppTest.from_file(f"{REPO}/app.py", default_timeout=180)
    for k, v in secrets.items(): at.secrets[k] = v
    for k, v in dict(verse_displayed=True, signed_up=True, user_name="Bench User",
                     user_email="bench@example.com", user_phone="", user_program="MDiv").items():
        at.session_state[k] = v
    at.run()
    at.sidebar.radio[0].set_value(lang).run()
    logs.clear(); t0 = time.time()
    if len(at.chat_input):
        at.chat_input[0].set_value(q).run()
    else:
        at.text_input[0].input(q); at.button[0].click().run()
    dt = time.time() - t0
    ans = [e.value for e in at.success]
    info = [e.value for e in at.info]
    exc = [str(e.value)[:200] for e in at.exception]
    return dt, (ans[0] if ans else ("[答不了] " if info else "[无输出]")), list(logs), exc

ask("中文(简)", "预热：申请费可以退吗？")  # 预热缓存，不计入
out = []
for lang, q, expect in CASES:
    dt, ans, lg, exc = ask(lang, q)
    out.append(dict(q=q, lang=lang, expect=expect, secs=round(dt, 1), answer=ans, log=lg, exc=exc))
    print(f"\n[{LABEL}] ({dt:.1f}s) {q}\n  期望: {expect}\n  回答: {ans[:220]}\n  记录: {lg} {exc or ''}", flush=True)
json.dump(out, open(HERE / f"bench_{LABEL}.json", "w"), ensure_ascii=False, indent=1)
print(f"\n[{LABEL}] 平均耗时 {sum(o['secs'] for o in out)/len(out):.1f}s")
