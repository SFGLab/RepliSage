"""Black + royal red theme of report.html (white page, light mode)."""
CSS = """
:root{--ink:#141414;--red:#A50F2D;--red2:#7E0B22;--mute:#5E5E5E;--line:#E6E6E6;--soft:#F7F7F7;--pink:#FBEDEF}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:#fff;color:var(--ink);font:16px/1.65 system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
a{color:var(--red);text-decoration:none}a:hover{text-decoration:underline}
.layout{display:grid;grid-template-columns:230px minmax(0,1fr);gap:0;max-width:1320px;margin:0 auto}
nav{position:sticky;top:0;align-self:start;height:100vh;overflow:auto;padding:34px 22px;border-right:1px solid var(--line)}
nav .brand{font-weight:800;font-size:20px;letter-spacing:-.02em;margin-bottom:18px;display:block;color:var(--ink)}
nav .brand b{color:var(--red)}nav a{display:block;padding:5px 0 5px 12px;color:var(--ink);font-size:14px;border-left:2px solid var(--line)}
nav a:hover{color:var(--red);border-left-color:var(--red);text-decoration:none}
main{padding:44px 54px 80px;min-width:0}
header.hero{border-bottom:4px solid var(--red);padding-bottom:26px;margin-bottom:34px}
.hero h1{font-size:68px;line-height:1;margin:0;letter-spacing:-.04em;font-weight:900}.hero h1 b{color:var(--red)}
.hero .tag{font-size:20px;color:var(--mute);margin:10px 0 18px}
.meta{display:flex;flex-wrap:wrap;gap:8px}.chip{border:1px solid var(--ink);border-radius:99px;padding:2px 12px;font-size:13px;font-weight:600}
.chip.on{background:var(--ink);color:#fff}.chip.red{background:var(--red);border-color:var(--red);color:#fff}.chip.off{border-color:var(--line);color:#9a9a9a}
h2{font-size:30px;color:var(--red);margin:62px 0 6px;letter-spacing:-.02em;border-bottom:1px solid var(--line);padding-bottom:8px}
h3{font-size:18px;margin:30px 0 8px}p.lead{color:var(--mute);margin:6px 0 18px}
.eq{background:var(--pink);border-left:4px solid var(--red);padding:12px 18px;margin:14px 0;font-family:Georgia,"Times New Roman",serif;font-size:18px}
.eq small{display:block;font:13px system-ui;color:var(--mute);margin-top:4px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;margin:16px 0}
.card{border:1px solid var(--line);border-radius:10px;padding:14px 16px}.card b{color:var(--red)}
figure{margin:26px 0;border:1px solid var(--line);border-radius:12px;overflow:hidden;background:#fff}
figure img{display:block;max-width:100%;height:auto;margin:0 auto;cursor:zoom-in}
figcaption{padding:14px 18px 16px;border-top:3px solid var(--red);background:var(--soft)}
figcaption h4{margin:0 0 6px;font-size:16px}
.cap{display:grid;grid-template-columns:1fr 1fr;gap:16px;font-size:14.5px}.cap .how{grid-column:1/-1;border-left:3px solid var(--red);padding-left:12px;background:#fbf4f5;border-radius:0 6px 6px 0;padding-top:6px;padding-bottom:6px}@media(max-width:900px){.cap{grid-template-columns:1fr}}
.cap span{display:block;color:var(--red);font-weight:700;font-size:11.5px;text-transform:uppercase;letter-spacing:.08em}
.tbl{margin:18px 0;overflow-x:auto}.tbl h4{margin:0 0 6px;font-size:16px}
table{border-collapse:collapse;width:100%;font-size:14px}
th{text-align:left;font-size:12px;text-transform:uppercase;letter-spacing:.06em;border-bottom:2px solid var(--red);padding:7px 10px;white-space:nowrap}
td{padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top}tr:nth-child(even) td{background:var(--soft)}
td.num{font-variant-numeric:tabular-nums}.note{font-size:13px;color:var(--mute);margin-top:6px}
pre{background:var(--soft);border-left:4px solid var(--red);padding:14px 16px;overflow:auto;font:13px/1.5 ui-monospace,Menlo,Consolas,monospace;border-radius:0 8px 8px 0;white-space:pre-wrap;word-break:break-word}
tr.chg td:nth-child(3){color:var(--red);font-weight:700}input.f{width:100%;padding:9px 12px;border:1px solid var(--ink);border-radius:8px;margin:8px 0;font-size:14px}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--mute);margin:10px 0}.sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
#lb{position:fixed;inset:0;background:rgba(255,255,255,.97);display:none;align-items:center;justify-content:center;z-index:9;cursor:zoom-out}#lb img{max-width:96vw;max-height:96vh}
footer{margin-top:70px;border-top:1px solid var(--line);padding-top:16px;color:var(--mute);font-size:13px}
@media(max-width:1000px){.layout{grid-template-columns:1fr}nav{position:static;height:auto;border:0}main{padding:24px 18px}.hero h1{font-size:44px}}
@media print{nav{display:none}.layout{display:block}figure{break-inside:avoid}}
"""
JS = """
document.querySelectorAll('figure img').forEach(i=>i.onclick=()=>{const l=document.getElementById('lb');l.firstChild.src=i.src;l.style.display='flex'});
document.getElementById('lb').onclick=e=>e.currentTarget.style.display='none';
const f=document.getElementById('pf');if(f)f.oninput=()=>{const q=f.value.toLowerCase();document.querySelectorAll('#pt tbody tr').forEach(r=>r.style.display=r.textContent.toLowerCase().includes(q)?'':'none')};
"""
