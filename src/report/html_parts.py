"""Small HTML builders for the report."""
import base64
import html
from pathlib import Path
from .text import cap_for, how_for
from .mathml import rich

def esc(x): return html.escape(str(x))

def img_uri(path):
    mime = "gif" if str(path).endswith(".gif") else "png"
    return f"data:image/{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()

def figure(fid, path, title=None):
    t, shows, expect = cap_for(fid)
    how = how_for(fid)
    body = (f'<div class="cap"><div><span>Shows</span>{rich(shows)}</div><div><span>Expect</span>{rich(expect)}</div>'
            + (f'<details class="how"><summary>How to read it</summary><p>{rich(how)}</p></details>' if how else "") + "</div>") if shows else ""
    return f'<figure id="fig-{esc(fid)}"><img alt="{esc(title or t)}" src="{img_uri(path)}"><figcaption><h4>{esc(title or t)}</h4>{body}</figcaption></figure>'

def video(fid, rel):
    """Inline mp4 player (the file sits next to report.html) with the caption of fid."""
    t, shows, expect = cap_for(fid)
    return (f'<figure id="fig-{esc(fid)}"><video src="{esc(rel)}" controls loop muted playsinline preload="metadata" style="width:100%"></video>'
            f'<figcaption><h4>{esc(t)}</h4><div class="cap"><div><span>Shows</span>{rich(shows)}</div><div><span>Expect</span>{rich(expect)}</div></div></figcaption></figure>')

def _fmt(c):
    """Thousands separators for plain large numbers; other cells unchanged."""
    try: v = float(str(c).replace(",", ""))
    except ValueError: return c
    return f"{int(v):,}" if str(c).lstrip("-").isdigit() and abs(v) >= 10000 else c

def table(t):
    head = "".join(f"<th>{esc(h)}</th>" for h in t.header)
    rows = "".join("<tr>" + "".join(f'<td class="num">{esc(_fmt(c))}</td>' if i else f"<td>{esc(c)}</td>" for i, c in enumerate(r)) + "</tr>" for r in t.rows)
    note = f'<div class="note">{esc(t.note)}</div>' if t.note else ""
    return f'<div class="tbl"><h4>{esc(t.title)}</h4><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>{note}</div>'

def findings(items):
    """Key-finding cards: status colour, label, big value, one-sentence detail."""
    names = {"ok": "good", "note": "note", "warn": "check"}
    cards = "".join(f'<div class="fc {f["status"]}"><span class="st">{names[f["status"]]}</span><div class="lb">{esc(f["label"])}</div><div class="v">{esc(f["value"])}</div><div class="dt">{rich(f["detail"])}</div></div>' for f in items)
    return f'<div class="find">{cards}</div><p class="hint">Rule-based reading aids, not statistical tests. Details are in the sections below.</p>'

def group(title, body, sub="", open_=False):
    """Collapsible block (used for diagnostics)."""
    return f'<details class="grp"{" open" if open_ else ""}><summary>{esc(title)}<small>{esc(sub)}</small></summary>{body}</details>'

def section(sid, title, body, lead=""):
    return f'<section id="{sid}"><h2>{esc(title)}</h2>' + (f'<p class="lead">{esc(lead)}</p>' if lead else "") + body + "</section>"

def chip(text, kind=""): return f'<span class="chip {kind}">{esc(text)}</span>'

def params_table(args):
    """All parameters: section, name, value, default, description; changed values in red."""
    rows = []
    for a in args:
        v, d = "" if a.val is None else a.val, "" if a.default in (None, "") else a.default
        chg = str(v) != str(d)
        rows.append(f'<tr class="{"chg" if chg else ""}"><td>{esc(a.section)}</td><td><b>{esc(a.name)}</b></td><td>{esc(v)}</td><td>{esc(d)}</td><td>{esc(a.help)}</td></tr>')
    return ('<input id="pf" class="f" placeholder="Filter parameters…"><div class="tbl"><table id="pt"><thead><tr><th>Section</th><th>Parameter</th><th>Value</th><th>Default</th><th>Description</th></tr></thead><tbody>'
            + "".join(rows) + '</tbody></table></div><div class="note">Values that differ from the default are shown in red.</div>')

def page(title, nav, main, css, js):
    links = "".join(f'<a class="l" href="#{i}">{esc(t)}</a>' for i, t in nav)
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title><style>{css}</style></head>'
            f'<body><div class="layout"><nav><a class="brand" href="#top">Repli<b>Sage</b></a>{links}</nav><main id="top">{main}</main></div><div id="lb"><img alt=""></div><script>{js}</script></body></html>')
