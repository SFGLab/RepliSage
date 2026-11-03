"""The phase bar of the movie must show the real simulation moment; fork markers on the contact map."""
import numpy as np
from src.viz3d.data import Traj, replication_eyes
from src.viz3d.video import bar_image, playhead, heat_image, Contacts, _rgb
from src.viz3d.palette import FORK
from src.plots.style import PHASE_COL

F, N, W = 300, 40, 1200
PH = ["G1"] * 100 + ["S"] * 100 + ["G2/M"] * 100          # phases of the recorded frames
STEP = np.arange(F) * 150                                  # MC step of every frame

def traj(rep=None):
    V = np.cumsum(np.random.default_rng(0).normal(size=(F, N, 3)), axis=1)
    rep = np.zeros((F, N), np.uint8) if rep is None else rep
    return Traj(V, 1, N, PH, STEP, np.zeros((F, N), np.int8), rep, [np.zeros((0, 2), int)] * F, True)

def playhead_x(img):
    """Pixel column of the playhead (the darkest vertical bar)."""
    dark = np.all(img[30:40] < 40, axis=2).sum(0); return np.flatnonzero(dark == dark.max()).mean()

def test_playhead_covers_full_trajectory_when_movie_is_subsampled():
    idx = np.unique(np.linspace(0, F - 1, 120).astype(int)); K = len(idx)       # the movie shows 120 of the 300 frames
    pos = [playhead(idx, a + t / 6) for a in range(K - 1) for t in range(6)]
    assert pos[0] == 0 and abs(pos[-1] - idx[-2]) < 3 and abs(playhead(idx, K - 1) - (F - 1)) < 1e-9   # starts at the first frame, ends at the last
    assert np.all(np.diff(pos) >= 0)

def test_bar_position_matches_step_and_phase():
    tr = traj(); bw = W / F
    for f in (0, 99, 150, 250, F - 1):
        img = bar_image(tr, f, W); x = playhead_x(img)
        assert abs(x - (f + .5) * bw) <= 3, f"playhead at {x:.0f}px for frame {f} (expected {(f + .5) * bw:.0f})"
        cell = img[28, int((f + .5) * bw) + (8 if f == 0 else -8)]   # bar colour next to the playhead (same phase)
        assert tuple(cell) == _rgb(PHASE_COL[PH[f]]), f"bar colour under frame {f} is not the colour of phase {PH[f]}"

def test_movie_position_maps_to_step():
    idx = np.unique(np.linspace(0, F - 1, 120).astype(int)); tr = traj()
    for a in (0, 30, 60, 119):
        f = playhead(idx, a); assert STEP[int(round(f))] == STEP[idx[a]]          # the displayed step is the one of the shown frame

def test_replication_eyes():
    rp = np.zeros(20, int); rp[3:8] = 1; rp[12:] = 1
    assert replication_eyes(rp) == [(3, 7), (12, 19)] and replication_eyes(np.ones(20, int)) == [] and replication_eyes(np.zeros(20, int)) == []

def test_forks_are_drawn_on_the_heatmap():
    fk = np.array([int(FORK[i:i + 2], 16) for i in (1, 3, 5)])
    def n_fork_px(rep):
        tr = traj(rep); img = heat_image(tr, Contacts(tr, 3), 10, 640)
        return int((np.abs(img.astype(int) - fk).sum(2) < 30).sum())
    none = np.zeros((F, N), np.uint8); eye = none.copy(); eye[:, 10:25] = 1; full = np.ones((F, N), np.uint8)
    base = n_fork_px(none)                                                          # the colour map itself may contain teal pixels
    assert n_fork_px(eye) > base + 100 and abs(n_fork_px(full) - base) < 5          # forks only where a replication eye ends

def test_phase_bar_independent_of_mc_step():
    """The S window must start/end at the same place of the bar (and of the movie) whatever the sampling interval."""
    from src.viz3d.data import phase_at
    t_rep, dur, n_steps, burnin = 20000, 20000, 60000, 3000
    for ms in (50, 100, 150, 250):
        steps = np.arange(burnin, n_steps, ms); Fm = len(steps); ph = [phase_at(s, t_rep, dur) for s in steps]
        tr = Traj(np.zeros((Fm, N, 3)), 1, N, ph, steps, np.zeros((Fm, N), np.int8), np.zeros((Fm, N), np.uint8), [np.zeros((0, 2), int)] * Fm, True)
        img = bar_image(tr, 0, 1200); row = img[28]
        s_cols = np.flatnonzero(np.all(row == _rgb(PHASE_COL["S"]), axis=1)); x0, x1 = s_cols.min() / 1200, s_cols.max() / 1200
        exp0, exp1 = (t_rep - burnin) / (n_steps - burnin), (t_rep + dur - burnin) / (n_steps - burnin)
        assert abs(x0 - exp0) < .02 and abs(x1 - exp1) < .02, f"MC_step={ms}: S spans {x0:.2f}-{x1:.2f}, expected {exp0:.2f}-{exp1:.2f}"
        idx = np.unique(np.linspace(0, Fm - 1, min(120, Fm)).astype(int)); k = int(np.argmin(np.abs(steps[idx] - t_rep)))   # the movie frame where S starts
        assert abs(playhead(idx, k) / Fm - exp0) < .02, f"MC_step={ms}: playhead at S start is off"
