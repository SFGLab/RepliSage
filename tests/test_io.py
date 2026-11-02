"""Structure file round trips."""
import numpy as np
from src.io.structure_io import write_mmcif, get_coordinates_cif, generate_psf

def test_mmcif_round_trip(tmp_path):
    V = np.random.default_rng(0).normal(size=(25, 3)); f = str(tmp_path / "s.cif")
    write_mmcif(V, cif_file_name=f)
    assert np.allclose(get_coordinates_cif(f), V, atol=1e-2)

def test_mmcif_two_chains_has_both(tmp_path):
    V = np.random.default_rng(1).normal(size=(10, 3)); f = str(tmp_path / "s2.cif")
    write_mmcif(V, V + 1, cif_file_name=f)
    assert len(get_coordinates_cif(f)) == 20

def test_psf_written(tmp_path):
    f = str(tmp_path / "a.psf"); generate_psf(12, f)
    assert "12 !NATOM" in open(f).read()
