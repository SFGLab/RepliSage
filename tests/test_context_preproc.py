"""Stage selection / validation and BEDPE preprocessing."""
import numpy as np
import pytest
from src.config.loader import get_config
from src.pipeline.context import build_context
from src.preproc.bedpe import preprocessing
from tests.conftest import REGION

def ctx(cli, **over):
    a, _ = get_config(cli(**over)); return build_context(a, make_dirs=False)

def test_bedpe_only_is_loop_extrusion_one_chain(cli):
    c = ctx(cli)
    assert not c.has_rep and not c.has_md and c.region_given and c.n_beads == 100

def test_rept_path_enables_replication(cli, tmp_path):
    f = tmp_path / "timing.txt"; f.write_text("chr14\t80835000\t80836000\t0.5\n")      # an existing file (a missing one falls back to the built-in data)
    c = ctx(cli, rept_path=str(f))
    assert c.has_rep and not c.is_sc

def test_sc_flag_enables_replication(cli):
    c = ctx(cli, use_sc_rept="True")
    assert c.has_rep and c.is_sc

def test_md_only_with_simulation_type(cli):
    assert ctx(cli, simulation_type="MD").has_md and ctx(cli, simulation_type="EM").has_md

def test_whole_chromosome(cli):
    a, _ = get_config(cli(region_start="", region_end="", n_beads=3000))
    c = build_context(a, make_dirs=False)
    assert not c.region_given and c.region[0] == 0

def test_errors(cli):
    with pytest.raises(ValueError): ctx(cli, bedpe_path="")
    with pytest.raises(ValueError): ctx(cli, chrom="")
    with pytest.raises(ValueError): ctx(cli, rept_path="x.csv")
    with pytest.raises(ValueError): ctx(cli, simulation_type="XX")

def test_preprocessing_vectors(bedpe_file):
    L, R, n = preprocessing(bedpe_file, list(REGION), "chr14", 100)
    assert L.shape == R.shape == (100,) and n > 0 and (L >= 0).all() and (R >= 0).all()
    assert n == max(np.count_nonzero(L), np.count_nonzero(R))

def test_empty_region_raises(bedpe_file):
    with pytest.raises(Exception): preprocessing(bedpe_file, [1, 1000], "chr14", 50)
