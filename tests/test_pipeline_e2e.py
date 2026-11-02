"""End-to-end: BEDPE-only run through run.main()."""
import numpy as np
from pathlib import Path
from src.run import main

def test_le_only_run(cli, tmp_path):
    main(cli(run_tag="t"))
    runs = list((tmp_path / "results").iterdir())
    assert len(runs) == 1
    r = runs[0]
    assert r.name.startswith("chr14_") and r.name.endswith("_LE_t") and "-" in r.name
    assert (r / "config_used.ini").exists()
    html = (r / "report.html").read_text()          # report: model, command and all parameters
    assert "Command and parameters" in html and "--bedpe_path" in html or "bedpe_path" in html.lower()
    assert any((r / "metadata").rglob("*"))

def test_run_name_override(cli, tmp_path):
    main(cli(run_name="custom"))
    assert (tmp_path / "results" / "custom").is_dir()

def test_replicas_in_subfolders(cli, tmp_path):
    main(cli(run_name="rr", n_replicas=2, n_threads=2))
    assert {p.name for p in (tmp_path / "results" / "rr").glob("replica_*")} == {"replica_00", "replica_01"}

def test_same_seed_same_result(cli, tmp_path):
    main(cli(run_name="a")); main(cli(run_name="b"))
    fa = sorted((tmp_path / "results" / "a").rglob("*.npy")); fb = sorted((tmp_path / "results" / "b").rglob("*.npy"))
    assert fa and [x.name for x in fa] == [x.name for x in fb]
    for x, y in zip(fa, fb): assert np.array_equal(np.load(x), np.load(y))

def test_bad_input_exits_nonzero(cli):
    import pytest
    with pytest.raises(SystemExit) as e: main(cli(bedpe_path="/no/such/file.bedpe"))
    assert e.value.code != 0
