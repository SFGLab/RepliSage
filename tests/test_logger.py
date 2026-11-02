"""Verbosity gating of the logger."""
import io
from src.logger import RepliLogger

def make(v):
    s = io.StringIO(); return RepliLogger(verbosity=v, color=False, stream=s), s

def test_always_shown():
    l, s = make(0)
    l.info("a"); l.success("b"); l.alert("c"); l.error("d")
    assert all(x in s.getvalue() for x in "abcd")

def test_levels():
    for v, shown in [(0, set()), (2, {"warn", "diag"}), (3, {"warn", "diag", "trace"})]:
        l, s = make(v)
        l.warning("warn"); l.diag("diag"); l.trace("trace")
        assert {w for w in ("warn", "diag", "trace") if w in s.getvalue()} == shown

def test_table_gated_by_level():
    l, s = make(0); l.table("T", [("k", 1)]); assert "k" not in s.getvalue()
    l, s = make(1); l.table("T", [("k", 1)]); assert "k" in s.getvalue()
    l, s = make(0); l.table("T", [("k", 1)], level=0); assert "k" in s.getvalue()

def test_no_ansi_without_color():
    l, s = make(3); l.info("x"); l.warning("y"); assert "\033[" not in s.getvalue()

def test_timestamp_present():
    l, s = make(0); l.info("x"); assert s.getvalue().startswith("[20")
