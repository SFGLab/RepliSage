"""Energy terms of the new kernel must equal the published (legacy, dense-J) formulas."""
import numpy as np
import pytest
from src.MC import potts_energy as P
from src.MC import loops_energy as LE
from src.MC import replication_energy as RE
from tests.reference import legacy_kernel as old

N = 60

def random_state(rng, n_lef=15, n_unbound=3):
    ms = rng.integers(0, N - 1, n_lef).astype(np.int64)
    ns = np.minimum(ms + rng.integers(0, 12, n_lef), N - 1).astype(np.int64)   # includes n == m (unbound-like) and n == m+1
    ms[:n_unbound], ns[:n_unbound] = -5, -5                                    # unbound LEFs
    spins = rng.integers(-2, 3, N).astype(np.int64)
    return ms, ns, spins

@pytest.mark.parametrize("seed", range(8))
def test_J_pair_matches_dense_matrix(seed):
    rng = np.random.default_rng(seed)
    ms, ns, _ = random_state(rng)
    J = old.initialize_J(N, ms, ns)
    for _ in range(300):
        m = int(rng.integers(0, N)); n = int(rng.integers(m, N))
        assert P.J_pair(ms, ns, m, n) == J[m, n]

@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("rfo", [True, False])
def test_E_potts_matches_legacy(seed, rfo):
    rng = np.random.default_rng(seed)
    ms, ns, spins = random_state(rng)
    J = old.initialize_J(N, ms, ns)
    h, ht = rng.normal(size=N), rng.normal(size=N)
    for t in (0, 5):
        a = old.E_potts(spins, J, h, ht, -1.3, 0.7, t, rfo)
        b = P.E_potts(spins, ms, ns, h, ht, -1.3, 0.7, t, rfo)
        assert b == pytest.approx(a, rel=1e-12, abs=1e-9)

@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize("rfo", [True, False])
def test_dE_node_equals_energy_difference(seed, rfo):
    """dE of a spin flip == E(after) - E(before): an independent check, no legacy code involved."""
    rng = np.random.default_rng(seed)
    ms, ns, spins = random_state(rng)
    h, ht = rng.normal(size=N), rng.normal(size=N)
    for t in (0, 7):
        for _ in range(40):
            i = int(rng.integers(N)); s_new = int(rng.choice([s for s in range(-2, 3) if s != spins[i]]))
            dE = P.get_dE_node(spins, i, s_new, ms, ns, h, ht, -1.1, 0.9, t, rfo)
            sp2 = spins.copy(); sp2[i] = s_new
            ref = P.E_potts(sp2, ms, ns, h, ht, -1.1, 0.9, t, rfo) - P.E_potts(spins, ms, ns, h, ht, -1.1, 0.9, t, rfo)
            assert dE == pytest.approx(ref, abs=1e-9)

@pytest.mark.parametrize("seed", range(6))
def test_dE_node_matches_legacy_when_ht_old_aliases_ht(seed):
    """The legacy code passes the same array as ht and ht_old (aliasing); the new O(1) field term must agree."""
    rng = np.random.default_rng(seed)
    ms, ns, spins = random_state(rng)
    J = old.initialize_J(N, ms, ns)
    h, ht = rng.normal(size=N), rng.normal(size=N)
    for _ in range(60):
        i = int(rng.integers(N)); s_new = int(rng.choice([s for s in range(-2, 3) if s != spins[i]]))
        a = old.get_dE_node(spins, i, s_new, J, h, ht, ht, -1.0, 1.0, 3, True)
        b = P.get_dE_node(spins, i, s_new, ms, ns, h, ht, -1.0, 1.0, 3, True)
        assert b == pytest.approx(a, abs=1e-9)

@pytest.mark.parametrize("seed", range(8))
def test_dE_potts_link_matches_legacy(seed):
    rng = np.random.default_rng(seed)
    ms, ns, spins = random_state(rng)
    J = old.initialize_J(N, ms, ns)
    for _ in range(200):
        k = int(rng.integers(len(ms)))
        m_new = int(rng.integers(0, N - 1)); n_new = int(rng.integers(m_new, N))
        a = old.get_dE_potts_link(spins, J, m_new, n_new, ms[k], ns[k], 1.0)
        b = P.get_dE_potts_link(spins, ms, ns, m_new, n_new, ms[k], ns[k], 1.0)
        assert b == pytest.approx(a, abs=1e-12)

@pytest.mark.parametrize("seed", range(6))
def test_loop_energy_terms_match_legacy(seed):
    rng = np.random.default_rng(seed)
    ms, ns, _ = random_state(rng)
    L, R = np.abs(rng.normal(size=N)), np.abs(rng.normal(size=N))
    for cbc in (False, True):
        assert LE.E_cross(ms, ns, 10, 5, 3.0, cbc) == old.E_cross(ms, ns, 10, 5, 3.0, cbc)
    assert LE.E_bind(L, R, ms, ns, -0.5) == pytest.approx(old.E_bind(L, R, ms, ns, -0.5))
    assert LE.E_fold(ms, ns, -0.4) == pytest.approx(old.E_fold(ms, ns, -0.4))
    for idx in (0, 4, 12):
        m_new = int(rng.integers(0, N - 1)); n_new = int(rng.integers(m_new, N))
        for cbc in (False, True):
            assert LE.get_dE_cross(ms, ns, m_new, n_new, idx, 10, 5, 2.0, cbc) == old.get_dE_cross(ms, ns, m_new, n_new, idx, 10, 5, 2.0, cbc)
        assert LE.get_dE_bind(L, R, -0.5, ms, ns, m_new, n_new, idx) == pytest.approx(old.get_dE_bind(L, R, -0.5, ms, ns, m_new, n_new, idx))

def test_dE_cross_equals_energy_difference():
    """Independent check: dE_cross of moving one LEF == E_cross(after) - E_cross(before)."""
    rng = np.random.default_rng(0)
    ms, ns, _ = random_state(rng)
    for idx in range(15):
        m_new = int(rng.integers(0, N - 1)); n_new = int(rng.integers(m_new, N))
        ms2, ns2 = ms.copy(), ns.copy(); ms2[idx], ns2[idx] = m_new, n_new
        d = LE.get_dE_cross(ms, ns, m_new, n_new, idx, 10, 5, 1.0, False)
        assert d == LE.E_cross(ms2, ns2, 10, 5, 1.0, False) - LE.E_cross(ms, ns, 10, 5, 1.0, False)

@pytest.mark.parametrize("seed", range(10))
def test_replication_penalty_matches_legacy(seed):
    rng = np.random.default_rng(seed)
    f = (rng.random(N) < 0.5).astype(np.int8)
    for _ in range(200):
        m = int(rng.integers(0, N - 1)); n = int(rng.integers(m, N))
        assert RE.Rep_Penalty(m, n, f) == pytest.approx(old.Rep_Penalty(m, n, f.astype(np.int64)))
        assert RE.violation_check(m, n, f) == old.violation_check(m, n, f.astype(np.int64))

def test_replication_penalty_semantics():
    f = np.array([0, 0, 1, 1, 1, 0, 0], dtype=np.int8)
    assert RE.Rep_Penalty(0, 1, f) == 0.0          # both ends unreplicated, nothing between
    assert RE.Rep_Penalty(1, 3, f) == 1.0          # straddles a fork
    assert RE.Rep_Penalty(2, 4, f) == 0.0          # entirely inside replicated DNA
    assert RE.Rep_Penalty(-5, -5, f) == 0.0        # unbound LEF
    assert RE.Rep_Penalty(0, 6, f) == pytest.approx(3 / 6)   # unreplicated ends, 3 replicated beads inside

def test_kappa_crossing_cases():
    K = LE.Kappa
    assert K(0, 5, 3, 8) == 1      # i < j < n_i < n_j  (crossing)
    assert K(3, 8, 0, 5) == 1      # reversed roles
    assert K(0, 10, 2, 6) == 0     # nested loops do not cross
    assert K(0, 3, 5, 9) == 0      # disjoint
    assert K(0, 5, 5, 9) == 1      # touching legs
    assert K(-5, -5, 0, 4) == 0    # unbound LEFs never cross
