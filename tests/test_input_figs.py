"""Anchor-level analysis and CTCF input figures for BEDPE / BED / narrowPeak."""
import numpy as np
import pytest
from types import SimpleNamespace as NS
from src.preproc.readers import read_binding
from src.analysis.anchors import anchors_of
from src.plots.style import make_geo, apply_style, FIGS
from src.plots.input_figs import fig_input, fig_anchor_map, fig_anchor_stats
from tests.test_readers import write, REG, N, CH

def ctx_of(path, fmt):
    b = read_binding(path, REG, CH, N); return NS(binding=b, region=REG, input_fmt=fmt, n_beads=N, chrom=CH)

@pytest.fixture
def files(tmp_path):
    bedpe = write(tmp_path, "l.bedpe", [(CH, 1_100_000, 1_102_000, CH, 1_400_000, 1_402_000, 2.0, 0, 1), (CH, 1_800_000, 1_802_000, CH, 1_500_000, 1_502_000, 1.0, 0, 1), (CH, 1_300_000, 1_302_000, CH, 1_600_000, 1_602_000, 1.0, -1, -1)])
    bed = write(tmp_path, "p.bed", [(CH, 1_100_000, 1_100_400, "a", 5, "+"), (CH, 1_300_000, 1_300_400, "b", 3, "-"), (CH, 1_500_000, 1_500_400, "c", 4, ".")])
    return bedpe, bed

def test_loop_configurations(files):
    a = anchors_of(ctx_of(files[0], "bedpe").binding)
    assert a.has_loops and list(a.config) == ["convergent", "divergent", "uncertain"]     # left forward + right reverse = convergent
    assert (a.left_bp < a.right_bp).all() and len(a.bp) == 6                               # reversed loop is re-ordered

def test_peak_anchors(files):
    a = anchors_of(ctx_of(files[1], "bed").binding)
    assert not a.has_loops and list(a.orient) == ["forward", "reverse", "unknown"] and a.has_orientation

@pytest.mark.parametrize("k,fmt", [(0, "bedpe"), (1, "bed")])
def test_figures_written(files, tmp_path, k, fmt):
    apply_style(); ctx = ctx_of(files[k], fmt); geo = make_geo(ctx); p = NS(L=ctx.binding.L, R=ctx.binding.R); FIGS.clear()
    fig_input(ctx, p, geo, str(tmp_path)); fig_anchor_map(ctx, geo, str(tmp_path)); fig_anchor_stats(ctx, str(tmp_path))
    assert {"loops_input", "anchor_map", "anchor_stats"} <= set(FIGS)
