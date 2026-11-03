(()=>{
const D=JSON.parse(document.getElementById('data').textContent);
const b64=(s,T)=>{const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new T(u.buffer)};
const X=b64(D.xyz,Int16Array),SP=b64(D.spin,Int8Array),RP=b64(D.rep,Uint8Array);
const F=D.F,C=D.C,N=D.N,SC=D.scale,$=id=>document.getElementById(id);
const PL=D.pal,SIS=PL.sis,LEFC=PL.lef,CTC=PL.ctcf,SPC=PL.spin;   // palette shared with the PyVista movie (viz3d/palette.py)
const PCOL={'G1':'#2B6CB0','S':'#A50F2D','G2/M':'#1B9E8F','All':'#7B4FA3'};
const MODES=C==2?['Sister chromatids','Compartments','Rainbow','Replication']:['Single chain','Compartments','Rainbow','Replication'];
let mode=0,tpos=0,playing=false,last=0;
// ---- three.js scene
const view=$('view'),W=()=>view.clientWidth,H=()=>view.clientHeight;
const R=new THREE.WebGLRenderer({antialias:true,preserveDrawingBuffer:true});R.setPixelRatio(window.devicePixelRatio);R.setSize(W(),H());R.setClearColor(0xeef2f8);view.appendChild(R.domElement);
const S=new THREE.Scene();S.fog=new THREE.Fog(0xeef2f8,D.ext*2.7,D.ext*9.7);const cam=new THREE.PerspectiveCamera(40,W()/H(),1,5000);
S.add(new THREE.HemisphereLight(0xffffff,0xc9d2e0,.85));const dl=new THREE.DirectionalLight(0xffffff,.8);dl.position.set(1,1.3,1.6);S.add(dl);
const dl2=new THREE.DirectionalLight(0xffffff,.35);dl2.position.set(-1,-.6,-1);S.add(dl2);
const ext=D.ext;
const home=()=>{cam.position.set(0,0,ext*2.7);ctl.target.set(0,0,0);ctl.update()};
const ctl=new THREE.OrbitControls(cam,R.domElement);ctl.enableDamping=true;home();
const group=new THREE.Group();S.add(group);
const P=new Float32Array(C*N*3);
function pos(t){const f0=Math.min(F-1,Math.floor(t)),f1=Math.min(F-1,f0+1),a=t-f0,n3=C*N*3;
 for(let k=0;k<n3;k++)P[k]=SC*((1-a)*X[f0*n3+k]+a*X[f1*n3+k])}
const col=new THREE.Color(),RB=i=>col.setHSL(.78*(1-i/(N-1)),.8,.5).clone();
function beadColor(f,c,i){
 if(mode==0)return col.set(SIS[c%2]).clone();
 if(mode==1)return col.set(SPC[Math.max(-2,Math.min(2,SP[f*N+i]))]).clone();
 if(mode==2)return RB(i);
 return col.set(RP[f*N+i]?PL.repl:PL.unrepl).clone()}
let meshes=[],tubeR=4*ext/240;
function build(){meshes.forEach(m=>{group.remove(m);m.geometry.dispose()});meshes=[];const f=Math.min(F-1,Math.round(tpos));
 for(let c=0;c<C;c++){const pts=[];for(let i=0;i<N;i++){const k=(c*N+i)*3;pts.push(new THREE.Vector3(P[k],P[k+1],P[k+2]))}
  const seg=N*3,g=new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts),seg,tubeR,10,false),ring=g.attributes.position.count/(seg+1),cs=new Float32Array(g.attributes.position.count*3);
  for(let r=0;r<=seg;r++){const b=Math.min(N-1,Math.round(r/seg*(N-1))),cc=beadColor(f,c,b);for(let j=0;j<ring;j++){const v=(r*ring+j)*3;cs[v]=cc.r;cs[v+1]=cc.g;cs[v+2]=cc.b}}
  g.setAttribute('color',new THREE.BufferAttribute(cs,3));
  const m=new THREE.Mesh(g,new THREE.MeshStandardMaterial({vertexColors:true,roughness:.32,metalness:.1}));group.add(m);meshes.push(m)}}
// markers: LEF anchors, loop chords, fork beads
const sph=new THREE.SphereGeometry(1,14,10),MAXL=Math.max(1,...D.lefs.map(l=>l.length/2));
const mk=c=>new THREE.MeshStandardMaterial({color:c,roughness:.3}),fMat=new THREE.MeshStandardMaterial({color:PL.fork,roughness:.3});
const anchors=[0,1].map(k=>new THREE.InstancedMesh(sph,mk(LEFC[k]),MAXL*2*C)),forks=new THREE.InstancedMesh(sph,fMat,N*C),chords=[0,1].map(k=>new THREE.LineSegments(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:LEFC[k]})));
const NCT=D.ctcf[0].length,cone=new THREE.ConeGeometry(.5,1.4,14),arrows=new THREE.InstancedMesh(cone,mk(CTC),Math.max(1,NCT*C));
group.add(...anchors,forks,...chords,arrows);
const M4=new THREE.Matrix4(),ZERO=new THREE.Matrix4().makeScale(0,0,0),UP=new THREE.Vector3(0,1,0),V3=new THREE.Vector3(),Q=new THREE.Quaternion();
function markers(f){const ai=[0,0],cp=[[],[]];let fi=0,ni=0;const L=D.lefs[f],K=D.lefk[f],rs=tubeR*1.25,P3=(c,i)=>(c*N+Math.max(0,Math.min(N-1,i)))*3;
 if($('tLef').checked)for(let c=0;c<C;c++)for(let k=0;k<L.length;k+=2){const q=K[k/2],a=P3(c,L[k]),b=P3(c,L[k+1]);
  for(const p of [a,b]){M4.makeScale(rs,rs,rs).setPosition(P[p],P[p+1],P[p+2]);anchors[q].setMatrixAt(ai[q]++,M4)}cp[q].push(P[a],P[a+1],P[a+2],P[b],P[b+1],P[b+2])}
 for(let q=0;q<2;q++){for(let k=ai[q];k<anchors[q].count;k++)anchors[q].setMatrixAt(k,ZERO);anchors[q].instanceMatrix.needsUpdate=true;chords[q].geometry.setAttribute('position',new THREE.Float32BufferAttribute(cp[q],3))}
 if($('tCtcf').checked)for(let c=0;c<C;c++)for(let m=0;m<NCT;m++){const i=D.ctcf[0][m],s=D.ctcf[1][m],a=P3(c,i-1),b=P3(c,i+1),p=P3(c,i);   // arrow points along (+1) / against (-1) the chain
  V3.set(P[b]-P[a],P[b+1]-P[a+1],P[b+2]-P[a+2]).normalize().multiplyScalar(s);Q.setFromUnitVectors(UP,V3);
  const o=new THREE.Vector3(P[p],P[p+1],P[p+2]),out=o.clone().normalize().multiplyScalar(tubeR*3);o.add(out);const sz=tubeR*3;
  M4.compose(o,Q,new THREE.Vector3(sz,sz,sz));arrows.setMatrixAt(ni++,M4)}
 for(let k=ni;k<arrows.count;k++)arrows.setMatrixAt(k,ZERO);arrows.instanceMatrix.needsUpdate=true;
 if($('tFork').checked&&D.hasRep)for(let c=0;c<C;c++)for(let i=0;i<N-1;i++)if(RP[f*N+i]!=RP[f*N+i+1]){const q=(c*N+i)*3,s=tubeR*1.8;M4.makeScale(s,s,s).setPosition(P[q],P[q+1],P[q+2]);forks.setMatrixAt(fi++,M4)}
 for(let k=fi;k<forks.count;k++)forks.setMatrixAt(k,ZERO);forks.instanceMatrix.needsUpdate=true}
// ---- contact map
const hm=$('hm'),hx=hm.getContext('2d'),off=document.createElement('canvas');off.width=off.height=N;const ox=off.getContext('2d');
let cache=new Map(),rcMul=1.5;
function d0(){let s=[];for(let i=0;i<N-1;i++){const a=i*3,b=a+3;s.push(Math.hypot(X[a]-X[b],X[a+1]-X[b+1],X[a+2]-X[b+2])*SC)}s.sort((a,b)=>a-b);return s[N>>1]}
const bond=d0();
function contact(f){if(cache.has(f))return cache.get(f);const rc=bond*rcMul,r2=rc*rc,m=new Float32Array(N*N),base=f*C*N*3;
 for(let c1=0;c1<C;c1++)for(let c2=0;c2<C;c2++)for(let i=0;i<N;i++){const a=base+(c1*N+i)*3;for(let j=0;j<N;j++){const b=base+(c2*N+j)*3,dx=(X[a]-X[b])*SC,dy=(X[a+1]-X[b+1])*SC,dz=(X[a+2]-X[b+2])*SC;
  {const q=(dx*dx+dy*dy+dz*dz)/r2;m[i*N+j]+=1/((1+q*q)*C*C)}}}
 cache.set(f,m);if(cache.size>400)cache.delete(cache.keys().next().value);return m}
const LUT=PL.lut||(()=>{const st=[[255,255,255],[255,240,179],[232,112,42],[165,15,45],[58,15,63]],o=[];for(let k=0;k<256;k++){const x=k/255*4,i=Math.min(3,Math.floor(x)),a=x-i;o.push(st[i].map((v,q)=>v+(st[i+1][q]-v)*a))}return o})();
const hex=s=>s;
// fork markers: triangle pointing the way the fork moves (v=0 horizontal, 1 vertical) or a diamond on the diagonal (v=2)
function forkMark(x,y,sd,v){hx.fillStyle=PL.fork;hx.strokeStyle='#fff';hx.lineWidth=1;hx.beginPath();
 if(v==2){hx.moveTo(x,y-5);hx.lineTo(x+5,y);hx.lineTo(x,y+5);hx.lineTo(x-5,y)}else if(v==0){hx.moveTo(x+4.5*sd,y);hx.lineTo(x-3*sd,y-4);hx.lineTo(x-3*sd,y+4)}else{hx.moveTo(x,y+4.5*sd);hx.lineTo(x-4,y-3*sd);hx.lineTo(x+4,y-3*sd)}
 hx.closePath();hx.fill();hx.stroke()}
function heat(f){const w=+$('win').value,avg=new Float32Array(N*N),a0=Math.max(0,f-w+1);
 for(let g=a0;g<=f;g++){const m=contact(g);for(let k=0;k<N*N;k++)avg[k]+=m[k]/(f-a0+1)}
 const v=[];for(let i=0;i<N;i++)for(let j=0;j<N;j++)if(Math.abs(i-j)>3)v.push(avg[i*N+j]);v.sort((a,b)=>a-b);const vmax=Math.max(.05,v[Math.floor(v.length*.995)]||1);
 const im=ox.createImageData(N,N);for(let k=0;k<N*N;k++){const t=Math.sqrt(Math.min(1,avg[k]/vmax)),c=LUT[Math.round(t*255)];im.data.set([c[0],c[1],c[2],255],k*4)}
 ox.putImageData(im,0,0);const T=hm.width,m0=14,sz=T-m0;hx.fillStyle='#fff';hx.fillRect(0,0,T,T);hx.imageSmoothingEnabled=false;hx.drawImage(off,m0,m0,sz,sz);
 const u=sz/N;for(let i=0;i<N;i++){const s=SP[f*N+i];hx.fillStyle=SPC[Math.max(-2,Math.min(2,s))];hx.fillRect(m0+i*u,2,u+.5,9);hx.fillStyle=RP[f*N+i]?'#A50F2D':'#E3E3E3';hx.fillRect(2,m0+i*u,9,u+.5)}
 if($('tFork').checked&&D.hasRep){let i=0;while(i<N){if(!RP[f*N+i]){i++;continue}const a=i;while(i<N&&RP[f*N+i])i++;const b=i-1;if(a==0&&b==N-1)continue;
  hx.strokeStyle=PL.fork;hx.globalAlpha=.85;hx.lineWidth=4;hx.beginPath();hx.moveTo(m0+a*u,m0+a*u);hx.lineTo(m0+(b+1)*u,m0+(b+1)*u);hx.stroke();hx.globalAlpha=1;
  for(const [pos,sd] of [[a,-1],[b+1,1]]){if((sd<0&&a==0)||(sd>0&&b+1>=N))continue;const p=m0+pos*u;forkMark(p,6.5,sd,0);forkMark(6.5,p,sd,1);forkMark(p,p,0,2)}}}
 if($('tLef').checked){const L=D.lefs[f];hx.strokeStyle='#fff';hx.lineWidth=1.2;for(let k=0;k<L.length;k+=2)for(const [a,b] of [[L[k],L[k+1]],[L[k+1],L[k]]]){hx.fillStyle=LEFC[D.lefk[f][k/2]];hx.beginPath();hx.arc(m0+(a+.5)*u,m0+(b+.5)*u,3.2,0,6.283);hx.fill();hx.stroke()}}}
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
function legend(){const L={0:C==2?[[SIS[0],'Sister 1'],[SIS[1],'Sister 2']]:[[SIS[0],'Chromatin']],1:[[SPC[2],'A: positive states'],[SPC[-2],'B: negative states']],2:[['#6a3df0','Start → end of region']],3:[[PL.repl,'Replicated'],[PL.unrepl,'Unreplicated']]}[mode],hasK=k=>D.lefk.some(a=>a.includes(k));
 $('legend3').innerHTML=L.map(([c,t])=>`<span><i class="sw" style="background:${c}"></i>${t}</span>`).join('')+[0,1].filter(hasK).map(k=>`<span><i class="sw" style="background:${LEFC[k]}"></i>${PL.lef_name[k]}</span>`).join('')+(NCT?`<span><i class="sw" style="background:${CTC}"></i>CTCF (arrow = orientation)</span>`:'')+(D.hasRep?`<span><i class="sw" style="background:${PL.fork}"></i>Fork</span>`:'')}
$('modes').innerHTML=MODES.map((m,i)=>`<span class="chip${i==mode?' on':''}" data-i="${i}">${m}</span>`).join('');
$('modes').onclick=e=>{if(e.target.dataset.i!=null){mode=+e.target.dataset.i;[...$('modes').children].forEach((c,i)=>c.classList.toggle('on',i==mode));legend();render()}};
function info(f){const ph=D.phase[f];const b=$('badge');b.textContent=ph==='All'?'Simulation':({'G1':'G1 · before replication','S':'S phase · replicating','G2/M':'G2/M · replicated'})[ph]||ph;b.style.background=PCOL[ph]||'#777';
 let m=0;for(let i=0;i<N;i++)m+=RP[f*N+i];$('stats').innerHTML=`MC step <b>${D.step[f]}</b><br>`+(D.hasRep?`Replicated <b>${(100*m/N).toFixed(0)} %</b><br>`:'')+`Bound LEFs <b>${D.lefs[f].length/2}</b>`;
 $('pos').textContent=`Frame ${f+1} / ${F}`}
function render(){const f=Math.min(F-1,Math.round(tpos));pos(tpos);build();markers(f);heat(f);drawTL();info(f)}
$('play').onclick=()=>{playing=!playing;$('play').innerHTML=playing?'&#10074;&#10074; Pause':'&#9654; Play';last=performance.now();if(playing&&tpos>=F-1)tpos=0};
['tLef','tFork','tCtcf'].forEach(i=>$(i).onchange=render);
$('win').oninput=()=>{$('winv').textContent=$('win').value;render()};
$('rc').oninput=()=>{rcMul=$('rc').value/20;cache.clear();render()};
$('rad').oninput=()=>{tubeR=$("rad").value*ext/240;render()};$('reset').onclick=home;$('spin').onchange=()=>{ctl.autoRotate=$('spin').checked;ctl.autoRotateSpeed=1.5};$('shot').onclick=()=>{fog();R.render(S,cam);const a=document.createElement('a');a.download='replisage_3d.png';a.href=R.domElement.toDataURL('image/png');a.click()};
function fog(){const d=cam.position.distanceTo(ctl.target);S.fog.near=d;S.fog.far=d+ext*7}   // gentle depth cue, only behind the centre
function loop(t){requestAnimationFrame(loop);if(playing){tpos+=(t-last)/1000*3*(+$('speed').value);last=t;if(tpos>=F-1){tpos=F-1;playing=false;$('play').innerHTML='&#9654; Play'}render()}ctl.update();fog();R.render(S,cam)}
window.onresize=()=>{R.setSize(W(),H());cam.aspect=W()/H();cam.updateProjectionMatrix();drawTL()};
rcMul=$('rc').value/20;legend();render();requestAnimationFrame(loop);
})();
