"""MC convergence diagnostics: autocorrelation, ESS, Geweke z, split R-hat."""
import numpy as np

def autocorr(x):
    """Normalised autocorrelation function (FFT)."""
    x = np.asarray(x, float); n = len(x); x = x - x.mean()
    f = np.fft.rfft(x, 2 * n); ac = np.fft.irfft(f * np.conj(f))[:n]
    return ac / ac[0] if ac[0] > 0 else np.r_[1.0, np.zeros(n - 1)]

def iat(x):
    """Integrated autocorrelation time (Geyer initial positive sequence)."""
    rho = autocorr(x); m = (len(rho) - 1) // 2
    if m < 1: return 1.0
    g = rho[0:2 * m:2] + rho[1:2 * m + 1:2]
    neg = np.flatnonzero(g <= 0)
    g = np.minimum.accumulate(g[:neg[0] if neg.size else m])
    return max(1.0, -1.0 + 2.0 * g.sum()) if g.size else 1.0

def ess(x):
    """Effective sample size; NaN for a constant series."""
    x = np.asarray(x, float)
    return np.nan if len(x) < 4 or x.std() == 0 else len(x) / iat(x)

def geweke(x, a=0.1, b=0.5):
    """Geweke z: first 10 % vs last 50 % of the series (|z| < 2 is fine)."""
    x = np.asarray(x, float); n = len(x)
    xa, xb = x[:int(a * n)], x[int((1 - b) * n):]
    if len(xa) < 4 or len(xb) < 4 or x.std() == 0: return np.nan
    va, vb = xa.var(ddof=1) / ess(xa), xb.var(ddof=1) / ess(xb)
    return float((xa.mean() - xb.mean()) / np.sqrt(va + vb)) if va + vb > 0 else np.nan

def split_rhat(chains):
    """Gelman-Rubin R-hat on split chains (1 chain -> two halves)."""
    n = min(len(c) for c in chains) // 2
    if n < 4: return np.nan
    parts = np.array([c[i * n:(i + 1) * n] for c in chains for i in (0, 1)], float)
    w = parts.var(axis=1, ddof=1).mean()
    if w == 0: return np.nan
    b = n * parts.mean(axis=1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * w + b / n) / w))

def diagnose(series, seg):
    """List of dicts (name, n, mean, sd, ess, iat, mcse, geweke, rhat) on slice seg of every series."""
    out = []
    for name, y in series.items():
        v = np.asarray(y, float)[seg]
        if len(v) < 8: continue
        e = ess(v)
        out.append(dict(name=name, n=len(v), mean=v.mean(), sd=v.std(ddof=1), ess=e, iat=len(v) / e if e == e else np.nan,
                        mcse=v.std(ddof=1) / np.sqrt(e) if e == e else np.nan, geweke=geweke(v), rhat=split_rhat([v])))
    return out

def diag_segment(ph, min_len=20):
    """Stationary part used for diagnostics: G1 after burn-in with replication, else everything after burn-in."""
    if ph.has_rep and ph.s0 - ph.g1 >= min_len: return slice(ph.g1, ph.s0), "G1 after burn-in"
    return slice(ph.g1, ph.n), "after burn-in"
