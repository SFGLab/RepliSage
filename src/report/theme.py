"""Theme of report.html: white page, ink text, royal-red accent (light mode)."""
CSS = """
:root{--ink:#16181D;--red:#A50F2D;--mute:#5F6670;--line:#E7E9EE;--soft:#F6F7F9;--pink:#FBEDEF;--ok:#1B9E8F;--note:#E3A008;--warn:#A50F2D}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:16px}
body{margin:0;background:#fff;color:var(--ink);font:16px/1.65 system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
a{color:var(--red);text-decoration:none}a:hover{text-decoration:underline}
.layout{display:grid;grid-template-columns:220px minmax(0,1fr);max-width:1340px;margin:0 auto}
nav{position:sticky;top:0;align-self:start;height:100vh;overflow:auto;padding:34px 20px;border-right:1px solid var(--line)}
nav .brand{font-weight:800;font-size:20px;letter-spacing:-.02em;margin-bottom:18px;display:block;color:var(--ink)}nav .brand b{color:var(--red)}
nav a.l{display:block;padding:5px 0 5px 12px;color:var(--mute);font-size:14px;border-left:2px solid var(--line)}
nav a.l:hover,nav a.l.on{color:var(--red);border-left-color:var(--red);text-decoration:none}nav a.l.on{font-weight:600}
main{padding:40px 54px 80px;min-width:0;max-width:1060px}
header.hero{padding-bottom:22px;margin-bottom:10px;border-bottom:1px solid var(--line)}
.hero h1{font-size:56px;line-height:1;margin:0;letter-spacing:-.04em;font-weight:900}.hero h1 b{color:var(--red)}
.hero .tag{font-size:18px;color:var(--mute);margin:8px 0 16px}
.meta{display:flex;flex-wrap:wrap;gap:8px}.chip{background:var(--soft);border-radius:99px;padding:2px 12px;font-size:13px;font-weight:600;color:var(--ink)}
.chip.on{background:var(--pink);color:var(--red)}.chip.red{background:var(--red);color:#fff}.chip.off{color:#9aa1ab;text-decoration:line-through}
h2{font-size:26px;margin:56px 0 4px;letter-spacing:-.02em}h2:before{content:"";display:inline-block;width:6px;height:22px;background:var(--red);border-radius:3px;margin-right:12px;vertical-align:-2px}
h3{font-size:17px;margin:28px 0 8px}p.lead{color:var(--mute);margin:2px 0 18px}
.find{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:14px;margin:18px 0}
.fc{border:1px solid var(--line);border-top:4px solid var(--c);border-radius:12px;padding:14px 16px 12px;background:#fff;--c:var(--note)}
.fc.ok{--c:var(--ok)}.fc.note{--c:var(--note)}.fc.warn{--c:var(--warn)}
.fc .lb{font-size:11.5px;text-transform:uppercase;letter-spacing:.08em;color:var(--mute);font-weight:700}
.fc .v{font-size:27px;font-weight:800;letter-spacing:-.02em;line-height:1.2;margin:2px 0 4px}
.fc .dt{font-size:13.5px;color:var(--mute);line-height:1.5}.fc .st{float:right;font-size:11px;font-weight:700;color:var(--c);text-transform:uppercase;letter-spacing:.06em}
.hint{font-size:13px;color:var(--mute)}
.eq{background:var(--soft);border-left:4px solid var(--red);padding:10px 16px;margin:12px 0;font-family:Georgia,"Times New Roman",serif;font-size:18px;border-radius:0 8px 8px 0}
.eq small{display:block;font:13px system-ui;color:var(--mute);margin-top:4px}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:12px;margin:14px 0}
.card{border:1px solid var(--line);border-radius:10px;padding:12px 14px;font-size:14.5px}.card b{color:var(--red)}
figure{margin:22px 0;border:1px solid var(--line);border-radius:14px;overflow:hidden;background:#fff}
figure img{display:block;max-width:100%;height:auto;margin:0 auto;cursor:zoom-in;padding:6px}
figcaption{padding:12px 18px 14px;border-top:1px solid var(--line);background:var(--soft)}
figcaption h4{margin:0 0 6px;font-size:15.5px}
.cap{display:grid;grid-template-columns:1fr 1fr;gap:6px 22px;font-size:14px;color:#333a44}
.cap span{display:block;color:var(--red);font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.08em}
details.how{grid-column:1/-1;margin-top:2px}details.how summary{cursor:pointer;color:var(--red);font-weight:600;font-size:13px}
details.how p{margin:6px 0 0;border-left:3px solid var(--red);padding-left:12px}
@media(max-width:900px){.cap{grid-template-columns:1fr}}
details.grp{border:1px solid var(--line);border-radius:12px;margin:18px 0;padding:0 16px;background:#fff}
details.grp>summary{cursor:pointer;padding:12px 0;font-weight:700;font-size:16px}details.grp>summary small{font-weight:400;color:var(--mute);margin-left:8px}
.tbl{margin:16px 0;overflow-x:auto}.tbl h4{margin:0 0 6px;font-size:15.5px}
table{border-collapse:collapse;width:100%;font-size:14px}
th{text-align:left;font-size:11.5px;text-transform:uppercase;letter-spacing:.06em;color:var(--mute);border-bottom:2px solid var(--line);padding:7px 10px;white-space:nowrap}
td{padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top}td.num{font-variant-numeric:tabular-nums;white-space:nowrap}
.note{font-size:13px;color:var(--mute);margin-top:6px}
pre{background:var(--soft);padding:14px 16px;overflow:auto;font:13px/1.5 ui-monospace,Menlo,Consolas,monospace;border-radius:8px;white-space:pre-wrap;word-break:break-word}
tr.chg td:nth-child(3){color:var(--red);font-weight:700}input.f{width:100%;padding:9px 12px;border:1px solid var(--line);border-radius:8px;margin:8px 0;font-size:14px}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;color:var(--mute);margin:10px 0}.sw{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.links{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0}.links a{border:1px solid var(--line);border-radius:99px;padding:4px 14px;font-size:14px;font-weight:600}
#lb{position:fixed;inset:0;background:rgba(255,255,255,.97);display:none;align-items:center;justify-content:center;z-index:9;cursor:zoom-out}#lb img{max-width:96vw;max-height:96vh}
footer{margin-top:70px;border-top:1px solid var(--line);padding-top:16px;color:var(--mute);font-size:13px}
@media(max-width:1000px){.layout{grid-template-columns:1fr}nav{position:static;height:auto;border:0}main{padding:24px 18px}.hero h1{font-size:40px}}
@media print{nav{display:none}.layout{display:block}figure,.fc{break-inside:avoid}}
"""
JS = """
document.querySelectorAll('figure img').forEach(i=>i.onclick=()=>{const l=document.getElementById('lb');l.firstChild.src=i.src;l.style.display='flex'});
document.getElementById('lb').onclick=e=>e.currentTarget.style.display='none';
const f=document.getElementById('pf');if(f)f.oninput=()=>{const q=f.value.toLowerCase();document.querySelectorAll('#pt tbody tr').forEach(r=>r.style.display=r.textContent.toLowerCase().includes(q)?'':'none')};
const L=[...document.querySelectorAll('nav a.l')];const io=new IntersectionObserver(es=>es.forEach(e=>{if(e.isIntersecting)L.forEach(a=>a.classList.toggle('on',a.getAttribute('href')==='#'+e.target.id))}),{rootMargin:'-20% 0px -70% 0px'});
document.querySelectorAll('main section').forEach(s=>io.observe(s));
window.onbeforeprint=()=>document.querySelectorAll('details').forEach(d=>d.open=true);
"""
