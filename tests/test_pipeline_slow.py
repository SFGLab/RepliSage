"""Full pipelines with replication + OpenMM on CPU (slow)."""
import pytest
from pathlib import Path
from src.run import main
from src.config.loader import get_config

pytestmark = pytest.mark.slow
pytest.importorskip("openmm")
SC = str(Path(__file__).resolve().parents[1] / "src" / "data" / "replication_timing_data.parquet")

def test_le_md_one_chain(cli, tmp_path):
    main(cli(run_name="md1", simulation_type="MD", sim_step=5, platform="CPU", save_plots="False"))
    out = tmp_path / "results" / "md1"
    cifs = [p.name for p in out.rglob("*.cif")]
    assert cifs and any("_LE" in c for c in cifs)

def test_replication_two_chains(cli, tmp_path):
    main(cli(run_name="rt", use_sc_rept="True", sc_rept_path=SC, rep_sim_t=40, rep_time_duration=400, rep_start_time=500,
             n_steps=1500, mc_step=50, burnin=100, n_lef2=3, simulation_type="MD", sim_step=5, platform="CPU"))
    cifs = [p.name for p in (tmp_path / "results" / "rt").rglob("*.cif")]
    assert any(c.endswith(("_BR.cif", "_R.cif", "_AR.cif")) for c in cifs)
    out = tmp_path / "results" / "rt"
    html = (out / "viewer_3d.html").read_text(); rep = (out / "report.html").read_text()   # 3D viewer and report
    assert "RepliSage" in html and "G2/M" in html and "3D ensemble" in rep
    assert (out / "ensemble" / "view.cxc").exists() and "covale" in next(out.rglob("ensemble_1_*.cif")).read_text()
