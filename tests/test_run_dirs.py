"""Run folder naming and location."""
from types import SimpleNamespace as NS
from pathlib import Path
from src.utils.run_dirs import region_tag, mode_tag, run_name, make_run_dir, resolve_out_root

def A(**k):
    d = dict(CHROM="chr14", REGION_START=1000, REGION_END=2000, RUN_NAME="", RUN_TAG="")
    d.update(k); return NS(**d)

def test_region_tags():
    assert region_tag("chr14", 1000, 2000) == "chr14_1000-2000"
    assert region_tag("chr14", None, None) == "chr14_whole-chromosome"

def test_mode_tags():
    assert [mode_tag(r, m) for r, m in [(0, 0), (1, 0), (0, 1), (1, 1)]] == ["LE", "LE-RT", "LE-MD", "LE-RT-MD"]

def test_name_with_tag_and_timestamp():
    assert run_name(A(RUN_TAG="t1"), True, True) == "chr14_1000-2000_LE-RT-MD_t1"
    assert run_name(A(), False, False).startswith("chr14_1000-2000_LE_")
    assert "whole-chromosome" in run_name(A(REGION_START=None, REGION_END=None), False, False)

def test_run_name_overrides_everything():
    assert run_name(A(RUN_NAME="mine", RUN_TAG="x"), True, True) == "mine"

def test_make_run_dir_creates_subfolders(tmp_path):
    d = make_run_dir(tmp_path, "r")
    for s in ("plots", "metadata", "ensemble", "plots/MCMC_diagnostics"): assert (Path(d) / s).is_dir()
    make_run_dir(tmp_path, "r")                              # reuse is harmless

def test_out_root_resolution(tmp_path):
    assert resolve_out_root(str(tmp_path)) == tmp_path
    assert resolve_out_root("results").name == "results" and resolve_out_root("results").is_absolute()
