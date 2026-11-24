import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sheets  # noqa: E402


class FakeWS:
    def __init__(self, first_row):
        self.title = "fake"
        self.first_row = first_row
        self.appended = []
        self.updated = None

    def row_values(self, i):
        return list(self.first_row)

    def update(self, values, rng):
        self.updated = (values, rng)

    def append_row(self, row, **kw):
        self.appended.append(row)


VALUES = {"conversation_id": "C1", "email": "e@x.com", "question": "Q"}
DEFAULT = ["conversation_id", "email", "question"]


def test_append_follows_existing_header_order():
    ws = FakeWS(["question", "conversation_id", "PT - 太平洋时间", "email"])
    sheets._append_by_header(ws, DEFAULT, VALUES)
    assert ws.appended == [["Q", "C1", "", "e@x.com"]]


def test_append_writes_default_header_on_empty_sheet():
    ws = FakeWS([])
    sheets._append_by_header(ws, DEFAULT, VALUES)
    assert ws.updated == ([DEFAULT], "A1")
    assert ws.appended == [["C1", "e@x.com", "Q"]]


def test_append_falls_back_to_default_order_when_row1_is_not_a_header():
    # 第一行是数据而不是表头：不能写出空行丢数据
    ws = FakeWS(["C0", "old@x.com", "old question"])
    sheets._append_by_header(ws, DEFAULT, VALUES)
    assert ws.appended == [["C1", "e@x.com", "Q"]]
