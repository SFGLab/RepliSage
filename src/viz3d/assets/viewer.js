(()=>{
const D=JSON.parse(document.getElementById('data').textContent);
const b64=(s,T)=>{const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new T(u.buffer)};
const X=b64(D.xyz,Int16Array),SP=b64(D.spin,Int8Array),RP=b64(D.rep,Uint8Array);
const F=D.F,C=D.C,N=D.N,SC=D.scale,$=id=>document.getElementById(id);
const PCOL={'G1':'#2B6CB0','S':'#A50F2D','G2/M':'#1B9E8F','All':'#7B4FA3'};
const MODES=C==2?['Sister chromatids','Compartments','Rainbow','Replication']:['Single chain','Compartments','Rainbow','Replication'];
let mode=0,tpos=0,playing=false,last=0;
// ---- three.js scene
const view=$('view'),W=()=>view.clientWidth,H=()=>view.clientHeight;
const R=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});R.setPixelRatio(window.devicePixelRatio);R.setSize(W(),H());R.setClearColor(0xffffff);view.appendChild(R.domElement);
const S=new THREE.Scene(),cam=new THREE.PerspectiveCamera(40,W()/H(),1,5000);
S.add(new THREE.AmbientLight(0xffffff,.75));const dl=new THREE.DirectionalLight(0xffffff,.8);dl.position.set(1,1.3,1.6);S.add(dl);
const dl2=new THREE.DirectionalLight(0xffffff,.35);dl2.position.set(-1,-.6,-1);S.add(dl2);
const ext=D.ext;
const home=()=>{cam.position.set(0,0,ext*1.8);ctl.target.set(0,0,0);ctl.update()};
const ctl=new THREE.OrbitControls(cam,R.domElement);ctl.enableDamping=true;home();
const group=new THREE.Group();S.add(group);
const P=new Float32Array(C*N*3);
function pos(t){const f0=Math.min(F-1,Math.floor(t)),f1=Math.min(F-1,f0+1),a=t-f0,n3=C*N*3;
 for(let k=0;k<n3;k++)P[k]=SC*((1-a)*X[f0*n3+k]+a*X[f1*n3+k])}
const col=new THREE.Color(),RB=i=>col.setHSL(.78*(1-i/(N-1)),.8,.5).clone();
function beadColor(f,c,i){
 if(mode==0)return col.set(C==2?(c?'#E3A008':'#E8702A'):'#E8702A').clone();
 if(mode==1){const s=SP[f*N+i];return col.set(s>0?'#C8213F':(s<0?'#2B6CB0':'#BDBDBD')).clone()}
 if(mode==2)return RB(i);
 return col.set(RP[f*N+i]?'#A50F2D':'#9AA3AE').clone()}
let meshes=[],tubeR=6*ext/240;
function build(){meshes.forEach(m=>{group.remove(m);m.geometry.dispose()});meshes=[];const f=Math.min(F-1,Math.round(tpos));
 for(let c=0;c<C;c++){const pts=[];for(let i=0;i<N;i++){const k=(c*N+i)*3;pts.push(new THREE.Vector3(P[k],P[k+1],P[k+2]))}
  const seg=N*3,g=new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts),seg,tubeR,10,false),ring=g.attributes.position.count/(seg+1),cs=new Float32Array(g.attributes.position.count*3);
  for(let r=0;r<=seg;r++){const b=Math.min(N-1,Math.round(r/seg*(N-1))),cc=beadColor(f,c,b);for(let j=0;j<ring;j++){const v=(r*ring+j)*3;cs[v]=cc.r;cs[v+1]=cc.g;cs[v+2]=cc.b}}
  g.setAttribute('color',new THREE.BufferAttribute(cs,3));
  const m=new THREE.Mesh(g,new THREE.MeshStandardMaterial({vertexColors:true,roughness:.45,metalness:.05}));group.add(m);meshes.push(m)}}
// markers: LEF anchors, loop chords, fork beads
const sph=new THREE.SphereGeometry(1,14,10),MAXL=Math.max(1,...D.lefs.map(l=>l.length/2));
const aMat=new THREE.MeshStandardMaterial({color:0x141414,roughness:.3}),fMat=new THREE.MeshStandardMaterial({color:0x1B9E8F,roughness:.3,transparent:true,opacity:.9});
const anchors=new THREE.InstancedMesh(sph,aMat,MAXL*2*C),forks=new THREE.InstancedMesh(sph,fMat,N*C);group.add(anchors,forks);
const chord=new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0xA50F2D}));group.add(chord);
const M4=new THREE.Matrix4(),ZERO=new THREE.Matrix4().makeScale(0,0,0);
function markers(f){let ai=0,fi=0;const L=D.lefs[f],cp=[];const rs=tubeR*1.25;
 if($('tLef').checked)for(let c=0;c<C;c++)for(let k=0;k<L.length;k+=2){const a=(c*N+L[k])*3,b=(c*N+L[k+1])*3;
  for(const q of [a,b]){M4.makeScale(rs,rs,rs).setPosition(P[q],P[q+1],P[q+2]);anchors.setMatrixAt(ai++,M4)}cp.push(P[a],P[a+1],P[a+2],P[b],P[b+1],P[b+2])}
 for(let k=ai;k<anchors.count;k++)anchors.setMatrixAt(k,ZERO);anchors.instanceMatrix.needsUpdate=true;
 chord.geometry.setAttribute('position',new THREE.Float32BufferAttribute(cp,3));
 if($('tFork').checked&&D.hasRep)for(let c=0;c<C;c++)for(let i=0;i<N-1;i++)if(RP[f*N+i]!=RP[f*N+i+1]){const q=(c*N+i)*3,s=tubeR*1.8;M4.makeScale(s,s,s).setPosition(P[q],P[q+1],P[q+2]);forks.setMatrixAt(fi++,M4)}
 for(let k=fi;k<forks.count;k++)forks.setMatrixAt(k,ZERO);forks.instanceMatrix.needsUpdate=true}
// ---- contact map
const hm=$('hm'),hx=hm.getContext('2d'),off=document.createElement('canvas');off.width=off.height=N;const ox=off.getContext('2d');
let cache=new Map(),rcMul=1.5;
function d0(){let s=[];for(let i=0;i<N-1;i++){const a=i*3,b=a+3;s.push(Math.hypot(X[a]-X[b],X[a+1]-X[b+1],X[a+2]-X[b+2])*SC)}s.sort((a,b)=>a-b);return s[N>>1]}
const bond=d0();
function contact(f){if(cache.has(f))return cache.get(f);const rc=bond*rcMul,r2=rc*rc,m=new Float32Array(N*N),base=f*C*N*3;
 for(let c1=0;c1<C;c1++)for(let c2=0;c2<C;c2++)for(let i=0;i<N;i++){const a=base+(c1*N+i)*3;for(let j=0;j<N;j++){const b=base+(c2*N+j)*3,dx=(X[a]-X[b])*SC,dy=(X[a+1]-X[b+1])*SC,dz=(X[a+2]-X[b+2])*SC;
  if(dx*dx+dy*dy+dz*dz<r2)m[i*N+j]+=1/(C*C)}}
 cache.set(f,m);if(cache.size>400)cache.delete(cache.keys().next().value);return m}
const LUT=(()=>{const st=[[255,255,255],[255,240,179],[232,112,42],[165,15,45],[58,15,63]],o=[];for(let k=0;k<256;k++){const x=k/255*4,i=Math.min(3,Math.floor(x)),a=x-i;o.push(st[i].map((v,q)=>v+(st[i+1][q]-v)*a))}return o})();
const hex=s=>s;
function heat(f){const w=+$('win').value,avg=new Float32Array(N*N),a0=Math.max(0,f-w+1);
 for(let g=a0;g<=f;g++){const m=contact(g);for(let k=0;k<N*N;k++)avg[k]+=m[k]/(f-a0+1)}
 const v=[];for(let i=0;i<N;i++)for(let j=0;j<N;j++)if(Math.abs(i-j)>3)v.push(avg[i*N+j]);v.sort((a,b)=>a-b);const vmax=Math.max(.05,v[Math.floor(v.length*.995)]||1);
 const im=ox.createImageData(N,N);for(let k=0;k<N*N;k++){const t=Math.pow(Math.min(1,avg[k]/vmax),.6),c=LUT[Math.round(t*255)];im.data.set([c[0],c[1],c[2],255],k*4)}
 ox.putImageData(im,0,0);const T=hm.width,m0=14,sz=T-m0;hx.fillStyle='#fff';hx.fillRect(0,0,T,T);hx.imageSmoothingEnabled=false;hx.drawImage(off,m0,m0,sz,sz);
 const u=sz/N;for(let i=0;i<N;i++){const s=SP[f*N+i];hx.fillStyle=s>0?'#C8213F':(s<0?'#2B6CB0':'#DDD');hx.fillRect(m0+i*u,2,u+.5,9);hx.fillStyle=RP[f*N+i]?'#A50F2D':'#E3E3E3';hx.fillRect(2,m0+i*u,9,u+.5)}
 if($('tFork').checked&&D.hasRep){hx.strokeStyle='#1B9E8F';hx.lineWidth=1.4;hx.setLineDash([5,4]);for(let i=0;i<N-1;i++)if(RP[f*N+i]!=RP[f*N+i+1]){const p=m0+(i+.5)*u;hx.beginPath();hx.moveTo(p,m0);hx.lineTo(p,T);hx.moveTo(m0,p);hx.lineTo(T,p);hx.stroke()}hx.setLineDash([])}
 if($('tLef').checked){const L=D.lefs[f];hx.fillStyle='#141414';hx.strokeStyle='#fff';hx.lineWidth=1.2;for(let k=0;k<L.length;k+=2)for(const [a,b] of [[L[k],L[k+1]],[L[k+1],L[k]]]){hx.beginPath();hx.arc(m0+(a+.5)*u,m0+(b+.5)*u,3.2,0,6.283);hx.fill();hx.stroke()}}}
// ---- timeline
const tl=$('tl'),tx=tl.getContext('2d');
function drawTL(){const dpr=window.devicePixelRatio||1,w=tl.clientWidth,h=tl.clientHeight;tl.width=w*dpr;tl.height=h*dpr;tx.setTransform(dpr,0,0,dpr,0,0);tx.clearRect(0,0,w,h);
 const bw=w/F,top=14,hh=h-top-4;let s=0;
 for(let f=0;f<=F;f++)if(f==F||D.phase[f]!=D.phase[s]){tx.fillStyle=PCOL[D.phase[s]]||'#777';tx.globalAlpha=.9;tx.fillRect(s*bw,top,(f-s)*bw+.5,hh);tx.globalAlpha=1;
  tx.fillStyle='#141414';tx.font='bold 11px sans-serif';tx.textAlign='center';tx.fillText(D.phase[s],(s+f)/2*bw,10);s=f}
 if(D.hasRep){tx.strokeStyle='#fff';tx.lineWidth=2;tx.beginPath();for(let f=0;f<F;f++){let m=0;for(let i=0;i<N;i++)m+=RP[f*N+i];const y=top+hh-(m/N)*(hh-4)-2;f?tx.lineTo((f+.5)*bw,y):tx.moveTo((f+.5)*bw,y)}tx.stroke()}
 const x=(tpos+.5)*bw;tx.fillStyle='#141414';tx.fillRect(x-1.5,top-4,3,hh+6);tx.beginPath();tx.moveTo(x-6,top-6);tx.lineTo(x+6,top-6);tx.lineTo(x,top+2);tx.fill()}
function seek(e){const r=tl.getBoundingClientRect();tpos=Math.max(0,Math.min(F-1,(e.clientX-r.left)/r.width*F-.5));render()}
let drag=false;tl.onmousedown=e=>{drag=true;seek(e)};window.onmouseup=()=>drag=false;window.onmousemove=e=>{if(drag)seek(e)};
tl.ontouchstart=tl.ontouchmove=e=>{seek(e.touches[0]);e.preventDefault()};
// ---- UI
function legend(){const L={0:C==2?[['#E8702A','Sister A'],['#E3A008','Sister B']]:[['#E8702A','Chromatin']],1:[['#C8213F','A compartment'],['#2B6CB0','B compartment'],['#BDBDBD','Neutral']],2:[['#6a3df0','Start → end of region']],3:[['#A50F2D','Replicated'],['#9AA3AE','Unreplicated']]}[mode];
 $('legend3').innerHTML=L.map(([c,t])=>`<span><i class="sw" style="background:${c}"></i>${t}</span>`).join('')+'<span><i class="sw" style="background:#141414"></i>LEF anchor</span>'+(D.hasRep?'<span><i class="sw" style="background:#1B9E8F"></i>Fork</span>':'')}
$('modes').innerHTML=MODES.map((m,i)=>`<span class="chip${i==mode?' on':''}" data-i="${i}">${m}</span>`).join('');
$('modes').onclick=e=>{if(e.target.dataset.i!=null){mode=+e.target.dataset.i;[...$('modes').children].forEach((c,i)=>c.classList.toggle('on',i==mode));legend();render()}};
function info(f){const ph=D.phase[f];const b=$('badge');b.textContent=ph==='All'?'Simulation':({'G1':'G1 · before replication','S':'S phase · replicating','G2/M':'G2/M · replicated'})[ph]||ph;b.style.background=PCOL[ph]||'#777';
 let m=0;for(let i=0;i<N;i++)m+=RP[f*N+i];$('stats').innerHTML=`MC step <b>${D.step[f]}</b><br>`+(D.hasRep?`Replicated <b>${(100*m/N).toFixed(0)} %</b><br>`:'')+`Bound LEFs <b>${D.lefs[f].length/2}</b>`;
 $('pos').textContent=`Frame ${f+1} / ${F}`}
function render(){const f=Math.min(F-1,Math.round(tpos));pos(tpos);build();markers(f);heat(f);drawTL();info(f)}
$('play').onclick=()=>{playing=!playing;$('play').innerHTML=playing?'&#10074;&#10074; Pause':'&#9654; Play';last=performance.now();if(playing&&tpos>=F-1)tpos=0};
['tLef','tFork'].forEach(i=>$(i).onchange=render);
$('win').oninput=()=>{$('winv').textContent=$('win').value;render()};
$('rc').oninput=()=>{rcMul=$('rc').value/20;cache.clear();render()};
$('rad').oninput=()=>{tubeR=$("rad").value*ext/240;render()};$('reset').onclick=home;
function loop(t){requestAnimationFrame(loop);if(playing){tpos+=(t-last)/1000*3*(+$('speed').value);last=t;if(tpos>=F-1){tpos=F-1;playing=false;$('play').innerHTML='&#9654; Play'}render()}ctl.update();R.render(S,cam)}
window.onresize=()=>{R.setSize(W(),H());cam.aspect=W()/H();cam.updateProjectionMatrix();drawTL()};
rcMul=$('rc').value/20;legend();render();requestAnimationFrame(loop);
})();
