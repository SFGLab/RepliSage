"""Tiny offline renderer of inline maths ($...$) to HTML: sub/superscripts, fractions, Greek letters (no external scripts)."""
import html
import re

SYM = {"alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "Delta": "Δ", "varepsilon": "ε", "epsilon": "ε", "kappa": "κ", "lambda": "λ", "mu": "μ", "rho": "ρ",
       "sigma": "σ", "tau": "τ", "theta": "θ", "phi": "φ", "Sigma": "Σ", "sum": "∑", "cdot": "·", "times": "×", "le": "≤", "ge": "≥", "in": "∈", "approx": "≈",
       "pm": "±", "to": "→", "ldots": "…", "infty": "∞", "neq": "≠", "sim": "∼", "partial": "∂", "pi": "π", "ell": "ℓ", "dots": "…", "left": "", "right": "", "quad": "  "}
FUNC = {"exp", "log", "max", "min", "ln", "mod", "sgn"}
SPACE = {",": "\u2009", ";": "\u2005", "!": "", " ": " "}

def _group(s, i):
    """Return (content, next index) of a {...} group or a single token at s[i]."""
    if i < len(s) and s[i] == "{":
        d, j = 1, i + 1
        while j < len(s) and d: d += (s[j] == "{") - (s[j] == "}"); j += 1
        return s[i + 1:j - 1], j
    if i < len(s) and s[i] == "\\":
        m = re.match(r"\\[A-Za-z]+", s[i:]); n = len(m.group(0)) if m else 2; return s[i:i + n], i + n
    return s[i:i + 1], i + 1

def render(s):
    """LaTeX-like string -> HTML."""
    out, i = [], 0
    while i < len(s):
        c = s[i]
        if c == "\\":
            m = re.match(r"\\([A-Za-z]+)", s[i:])
            if not m: ch = s[i + 1:i + 2]; out.append(SPACE.get(ch, html.escape(ch))); i += 2; continue
            name = m.group(1); i += len(m.group(0))
            if name == "frac":
                a, i = _group(s, i); b, i = _group(s, i); out.append(f'<span class="fr"><span>{render(a)}</span><span>{render(b)}</span></span>')
            elif name == "text": a, i = _group(s, i); out.append(f'<span class="fn">{html.escape(a)}</span>')
            elif name == "bar": a, i = _group(s, i); out.append(f'<span class="ov">{render(a)}</span>')
            elif name in FUNC: out.append(f'<span class="fn">{name}</span>')
            else: out.append(SYM.get(name, html.escape(name)))
        elif c in "_^":
            a, i = _group(s, i + 1); out.append(f'<{"sub" if c == "_" else "sup"}>{render(a)}</{"sub" if c == "_" else "sup"}>')
        elif c == "{": a, i = _group(s, i); out.append(render(a))
        elif c.isalpha(): out.append(f"<i>{c}</i>"); i += 1
        else: out.append(html.escape(c)); i += 1
    return "".join(out)

def rich(text):
    """Escape plain text and render every $...$ part as maths."""
    parts = re.split(r"\$(.+?)\$", str(text)); return "".join(html.escape(p) if k % 2 == 0 else f'<span class="m">{render(p)}</span>' for k, p in enumerate(parts))
