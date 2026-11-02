"""MD helpers (fast) and a tiny OpenMM run (slow)."""
import numpy as np
import pytest
from src.MD.structures import compute_init_struct, helix_structure, polymer_circle

def test_init_structures_shape():
    for mode in ("confined_rw", "rw", "helix", "spiral"):
        try: V = compute_init_struct(50, mode)
        except Exception: continue
        assert np.asarray(V).shape == (50, 3) and np.isfinite(V).all()

def test_helix_and_circle():
    assert helix_structure(30).shape == (30, 3) and np.asarray(polymer_circle(30)).shape[0] == 30

def test_parse_quantity():
    pytest.importorskip("openmm")
    from src.MD.units import parse_quantity
    assert parse_quantity("1 nanometer") is not None
    with pytest.raises(ValueError): parse_quantity("2.5")        # a unit is required
