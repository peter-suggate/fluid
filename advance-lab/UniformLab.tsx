"use client";
import {useEffect,useRef,useState} from "react";
import {SCENE_CATALOG,findSceneDefinition} from "../lib/core/scenes";
import {sceneDocument} from "../lib/core/scene-definition";
import {UNIFORM_VOLUME_PIPELINE} from "../lib/methods/uniform/uniform-volume-pipeline";
import {UNIFORM_GEOMETRIC_PARAMS} from "../lib/methods/uniform/uniform-geometric-parameters";
import {UNIFORM_PAPER_DT_S} from "../lib/methods/uniform/uniform-paper";
import {UniformLabController,UNIFORM_LAB_VALUES,uniformLabSceneLimitation,type UniformView} from "../lib/physics-wasm/uniform-controller";
import {advanceRdfTriangles,clippedScalarTriangle} from "../lib/physics-wasm/advance-view";
import {DEFAULT_LAB_SCENE_ID,labSceneValue} from "./lab-scenes";
import base from "./AdvanceLab.module.css";
import css from "./UniformLab.module.css";

type Lens="surface"|"volume"|"pressure"|"velocity"|"tiles"|"release";
const lenses:readonly [Lens,string][]=[["surface","Surface · phi = 0"],["volume","Conserved volume"],["pressure","Pressure"],["velocity","Velocity"],["tiles","Work tiles"],["release","Released faces"]];
const catalog=SCENE_CATALOG.map(def=>({id:def.id,name:def.name,limitation:uniformLabSceneLimitation(sceneDocument(def))}));
const stages=UNIFORM_VOLUME_PIPELINE.stages.filter(stage=>stage.id!=="rigid-coupling");
const initialScene=()=>labSceneValue.read(new URLSearchParams(window.location.search).get(labSceneValue.key));
interface Camera {zoom:number;x:number;y:number}
const fit:Camera={zoom:1,x:0,y:0};
function transform(v:UniformView,width:number,height:number,camera:Camera){
  const scale=Math.min((width-64)/(v.nx*v.cellSize[0]),(height-64)/(v.ny*v.cellSize[1]))*camera.zoom;
  return {sx:scale*v.cellSize[0],sy:scale*v.cellSize[1],ox:width/2-v.nx*scale*v.cellSize[0]/2+camera.x,oy:height/2+v.ny*scale*v.cellSize[1]/2+camera.y};
}
function draw(canvas:HTMLCanvasElement,v:UniformView,lens:Lens,grid:boolean,camera:Camera){
  const rect=canvas.getBoundingClientRect(),dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,Math.round(rect.width*dpr));canvas.height=Math.max(1,Math.round(rect.height*dpr));
  const g=canvas.getContext("2d")!;g.scale(dpr,dpr);
  const style=getComputedStyle(canvas);const colour=(key:string,fallback:string)=>style.getPropertyValue(key).trim()||fallback;
  g.fillStyle=colour("--slice-ground","#131820");g.fillRect(0,0,rect.width,rect.height);
  const {sx,sy,ox,oy}=transform(v,rect.width,rect.height,camera);
  g.translate(ox,oy);g.scale(sx,-sy);
  const liquid=colour("--slice-liquid","#4f9ae0"),solid=colour("--slice-solid","#887c68");
  let maximum=1e-8;if(lens==="pressure")for(const p of v.pressure)maximum=Math.max(maximum,Math.abs(p));
  if(lens==="velocity")for(let i=0;i<v.nx*v.ny;i++)maximum=Math.max(maximum,Math.hypot(v.velocity[2*i]!,v.velocity[2*i+1]!));
  const tc=Math.ceil(v.nx/4);
  for(let y=0;y<v.ny;y++)for(let x=0;x<v.nx;x++){
    const i=x+v.nx*y;if(v.capacity[i]!<=1e-5){g.fillStyle=solid;g.fillRect(x,y,1,1);continue;}
    if(lens==="surface"||lens==="release"){
      const a=x+(v.nx+1)*y;
      g.fillStyle=liquid;g.beginPath();
      for(const t of advanceRdfTriangles(x,y,v.phi[a]!,v.phi[a+1]!,v.phi[a+v.nx+2]!,v.phi[a+v.nx+1]!)){
        const p=clippedScalarTriangle(t);if(p.length<6)continue;
        g.moveTo(p[0]!,p[1]!);for(let k=2;k<p.length;k+=2)g.lineTo(p[k]!,p[k+1]!);g.closePath();
      }g.fill();
    }else if(lens==="tiles"){
      const bits=v.tiles[Math.floor(x/4)+tc*Math.floor(y/4)]??0;
      g.fillStyle=bits&1?"#3679ac":bits&2?"#537654":bits&4?"#88642e":"#333941";g.fillRect(x,y,1,1);
    }else{
      const value=lens==="volume"?v.volume[i]!/Math.max(v.capacity[i]!,1e-6):lens==="pressure"?Math.abs(v.pressure[i]!)/maximum:Math.hypot(v.velocity[2*i]!,v.velocity[2*i+1]!)/maximum;
      if(value>0){g.globalAlpha=Math.min(1,value);g.fillStyle=lens==="pressure"?(v.pressure[i]!<0?"#dd9955":"#7e9ee5"):liquid;g.fillRect(x,y,1,1);g.globalAlpha=1;}
    }
    if(lens==="release"&&v.released[i]){
      const bits=v.released[i]!;g.strokeStyle="#f5be52";g.lineWidth=2/Math.max(sx,sy);g.beginPath();
      if(bits&1){g.moveTo(x+1,y);g.lineTo(x+1,y+1);}if(bits&2){g.moveTo(x,y+1);g.lineTo(x+1,y+1);}
      if(bits&4){g.moveTo(x,y);g.lineTo(x,y+1);}if(bits&8){g.moveTo(x,y);g.lineTo(x+1,y);}g.stroke();
    }
  }
  if(lens==="velocity"){
    const stride=Math.max(1,Math.ceil(15/Math.min(sx,sy)));g.strokeStyle=colour("--slice-ink","#ddd");g.lineWidth=1/Math.max(sx,sy);g.beginPath();
    for(let y=0;y<v.ny;y+=stride)for(let x=0;x<v.nx;x+=stride){const i=x+v.nx*y;const u=v.velocity[2*i]!/maximum,w=v.velocity[2*i+1]!/maximum;g.moveTo(x+0.5,y+0.5);g.lineTo(x+0.5+u*stride*.8,y+0.5+w*stride*.8);}g.stroke();
  }
  g.strokeStyle=colour("--slice-grid","#777");g.lineWidth=1/Math.max(sx,sy);g.beginPath();
  if(grid&&Math.min(sx,sy)>=5){for(let x=0;x<=v.nx;x++){g.moveTo(x,0);g.lineTo(x,v.ny);}for(let y=0;y<=v.ny;y++){g.moveTo(0,y);g.lineTo(v.nx,y);}}
  g.rect(0,0,v.nx,v.ny);g.stroke();
}

export function UniformLab(){
  const [sceneId,setSceneId]=useState(initialScene),[restart,setRestart]=useState(0);
  const [view,setView]=useState<UniformView>(),[loading,setLoading]=useState(true),[error,setError]=useState<string>();
  const [playing,setPlaying]=useState(true),[busy,setBusy]=useState(false),[cost,setCost]=useState(0);
  const [lens,setLens]=useState<Lens>("surface"),[grid,setGrid]=useState(false),[camera,setCamera]=useState<Camera>(fit);
  const [stageId,setStageId]=useState(stages[0]!.id),[probe,setProbe]=useState<number>(),[pinned,setPinned]=useState(false);
  const controller=useRef<UniformLabController>(undefined),inflight=useRef(false),canvas=useRef<HTMLCanvasElement>(null);
  const pan=useRef<{x:number;y:number;camera:Camera}|undefined>(undefined);
  const [paint,setPaint]=useState(0);
  const beginLoad=()=>{setLoading(true);setError(undefined);setView(undefined);setProbe(undefined);setPinned(false);setCamera(fit);setCost(0);};
  useEffect(()=>{const onPop=()=>{const next=initialScene();if(next!==sceneId){beginLoad();setSceneId(next);}};window.addEventListener("popstate",onPop);return()=>window.removeEventListener("popstate",onPop);},[sceneId]);
  useEffect(()=>{
    let alive=true;let owner:UniformLabController|undefined;
    void(async()=>{try{
      const definition=findSceneDefinition(sceneId);if(!definition)throw new Error("Unknown scene");
      const scene=sceneDocument(definition);const limitation=uniformLabSceneLimitation(scene);if(limitation)throw new Error(limitation);
      owner=await UniformLabController.create();if(!alive){await owner.destroy();return;}
      controller.current=owner;const initial=await owner.load(scene);
      if(alive){setView(initial);setLoading(false);}
    }catch(e){if(alive){setError(String(e instanceof Error?e.message:e));setLoading(false);setPlaying(false);}}})();
    return()=>{alive=false;if(controller.current===owner)controller.current=undefined;if(owner)void owner.destroy().catch(()=>{});};
  },[sceneId,restart]);
  const advance=async()=>{
    const owner=controller.current;if(!owner||inflight.current)return;
    inflight.current=true;setBusy(true);const start=performance.now();
    try{const next=await owner.advance(UNIFORM_PAPER_DT_S);if(controller.current===owner){setView(next);setCost(performance.now()-start);}}
    catch(e){if(controller.current===owner){setError(String(e instanceof Error?e.message:e));setPlaying(false);}}
    finally{inflight.current=false;setBusy(false);}
  };
  useEffect(()=>{
    if(!playing||loading||error)return;
    let alive=true,id=0,last=0;
    const tick=(now:number)=>{if(!alive)return;if(now-last>=1000*UNIFORM_PAPER_DT_S&&!inflight.current){last=now;void advance();}id=requestAnimationFrame(tick);};
    id=requestAnimationFrame(tick);return()=>{alive=false;cancelAnimationFrame(id);};
    // A frame publication must not restart the playback clock.
  },[playing,loading,error]);
  useEffect(()=>{const el=canvas.current;if(!el)return;const observer=new ResizeObserver(()=>setPaint(n=>n+1));observer.observe(el);const theme=new MutationObserver(()=>setPaint(n=>n+1));theme.observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});const media=matchMedia("(prefers-color-scheme: dark)");const refresh=()=>setPaint(n=>n+1);media.addEventListener("change",refresh);return()=>{observer.disconnect();theme.disconnect();media.removeEventListener("change",refresh);};},[]);
  useEffect(()=>{if(canvas.current&&view)draw(canvas.current,view,lens,grid,camera);},[view,lens,grid,camera,paint]);
  const chooseScene=(id:string)=>{beginLoad();const url=new URL(location.href);if(id===DEFAULT_LAB_SCENE_ID)url.searchParams.delete("scene");else url.searchParams.set("scene",id);history.replaceState(history.state,"",url);setSceneId(id);};
  const stage=stages.find(s=>s.id===stageId)!;
  const keys=new Set((stage.controls??[]).flatMap(c=>"param"in c?[c.param]:[]));
  const parameters=UNIFORM_GEOMETRIC_PARAMS.filter(p=>keys.has(p.key)&&!["activeRegion","pressureWindow","sharpeningWorkMap"].includes(p.key));
  const total=view?.volume.reduce((a,b)=>a+b,0)??0,initial=Number(view?.receipt.initialVolume??0);
  const pressure=(view?.receipt.uniform as {pressure?:{residual?:number;cycles?:number;converged?:boolean}}|undefined)?.pressure;
  return <main className={`${base.lab} ${css.root}`}>
    <header className={css.bar}>
      <label>Scene <select aria-label="Scene" value={sceneId} onChange={e=>chooseScene(e.target.value)}>{catalog.map(s=><option key={s.id} value={s.id} disabled={!!s.limitation}>{s.name}{s.limitation?" · unavailable":""}</option>)}</select></label>
      <div className={css.playback}><button onClick={()=>setPlaying(p=>!p)} disabled={loading||!!error}>{playing?"Pause":"Play"}</button><button onClick={()=>{setPlaying(false);void advance();}} disabled={loading||busy||playing||!!error}>Step</button><button onClick={()=>{beginLoad();setRestart(n=>n+1);}} disabled={loading}>Reset</button></div>
      <output data-testid="uniform-clock">Frame {view?.revision.frame??0} · {(view?.revision.time??0).toFixed(2)} s · {cost.toFixed(1)} ms/step</output>
    </header>
    <div className={css.layout}>
      <section className={css.viewport} aria-label="Uniform simulation">
        <div className={css.tools}><select aria-label="Field" value={lens} onChange={e=>setLens(e.target.value as Lens)}>{lenses.map(([id,label])=><option key={id} value={id}>{label}</option>)}</select><label><input type="checkbox" checked={grid} onChange={e=>setGrid(e.target.checked)}/> Grid</label><button onClick={()=>setCamera(fit)}>Fit</button></div>
        <canvas ref={canvas} tabIndex={0} aria-label="Uniform Geometric 2D fluid" onContextMenu={e=>e.preventDefault()}
          onKeyDown={e=>{if(e.key==="0")setCamera(fit);if(e.key===" "){e.preventDefault();if(!loading&&!error)setPlaying(p=>!p);}if(e.key==="Escape"){setPinned(false);setProbe(undefined);}}}
          onWheel={e=>{const rect=e.currentTarget.getBoundingClientRect();const factor=Math.exp(-e.deltaY*.001);setCamera(c=>{const zoom=Math.min(32,Math.max(1,c.zoom*factor)),r=zoom/c.zoom;const x=e.clientX-rect.left-rect.width/2,y=e.clientY-rect.top-rect.height/2;return {zoom,x:x-(x-c.x)*r,y:y-(y-c.y)*r};});}}
          onPointerDown={e=>{if(e.button===1||e.shiftKey){e.preventDefault();pan.current={x:e.clientX,y:e.clientY,camera};e.currentTarget.setPointerCapture(e.pointerId);}else setPinned(p=>!p);}}
          onPointerUp={e=>{pan.current=undefined;if(e.currentTarget.hasPointerCapture(e.pointerId))e.currentTarget.releasePointerCapture(e.pointerId);}}
          onPointerCancel={()=>{pan.current=undefined;}}
          onPointerMove={e=>{if(pan.current){const p=pan.current;setCamera({...p.camera,x:p.camera.x+e.clientX-p.x,y:p.camera.y+e.clientY-p.y});return;}if(!view||pinned)return;const rect=e.currentTarget.getBoundingClientRect(),t=transform(view,rect.width,rect.height,camera);const x=Math.floor((e.clientX-rect.left-t.ox)/t.sx),y=Math.floor((t.oy-(e.clientY-rect.top))/t.sy);setProbe(x>=0&&y>=0&&x<view.nx&&y<view.ny?x+view.nx*y:undefined);}}/>
        {loading&&<div className={css.message} role="status">Loading Uniform Geometric…</div>}
        {error&&<div className={css.message} role="alert">{error}<p>Choose another scene or reset the run.</p></div>}
        <footer className={css.caption}>{view?`${view.nx} × ${view.ny} cells · central XY slice · Δt = 1/30 s`:""}<span>Wheel to zoom · shift-drag to pan · click to pin a cell</span></footer>
      </section>
      <aside className={css.sidebar}>
        <h1>Uniform Geometric</h1><p className={css.muted}>Shared 3D defaults · whole-domain solve</p>
        <dl><dt>Liquid area</dt><dd>{view?(total*view.cellSize[0]*view.cellSize[1]).toFixed(5):"—"} m²</dd><dt>Volume change</dt><dd>{initial?((total/initial-1)*100).toFixed(5):"0"}%</dd><dt>Pressure residual</dt><dd>{view?.revision.frame?pressure?.residual?.toExponential(2):"—"}</dd><dt>Pressure cycles</dt><dd>{view?.revision.frame?`${pressure?.cycles??0} · ${pressure?.converged?"converged":"budget reached"}`:"—"}</dd></dl>
        <h2>Completed frame</h2><p className={css.muted}>Field views show the last completed advance. Stage descriptions and defaults are shared with 3D.</p>
        <nav className={css.stages} aria-label="Uniform stages">{stages.map(s=><button key={s.id} aria-pressed={s.id===stageId} onClick={()=>setStageId(s.id)}>{s.label}</button>)}</nav>
        <details open key={stageId}><summary>{stage.label}</summary><p>{stage.tip.summary}</p>{parameters.length>0&&<dl>{parameters.map(p=><div key={p.key}><dt title={p.hint}>{p.label}</dt><dd>{p.kind==="select"?p.options.find(o=>o.value===UNIFORM_LAB_VALUES[p.key])?.label:`${UNIFORM_LAB_VALUES[p.key]} ${p.unit??""}`}</dd></div>)}</dl>}</details>
        {probe!==undefined&&view&&<section aria-label="Cell inspection"><h2>Cell {probe%view.nx}, {Math.floor(probe/view.nx)}{pinned?" · pinned":""}</h2><dl><dt>V / capacity</dt><dd>{view.volume[probe]!.toFixed(5)} / {view.capacity[probe]!.toFixed(3)}</dd><dt>Pressure</dt><dd>{view.pressure[probe]!.toPrecision(5)} Pa</dd><dt>+X / +Y velocity</dt><dd>{view.velocity[2*probe]!.toFixed(4)} / {view.velocity[2*probe+1]!.toFixed(4)} m/s</dd><dt>Release mask</dt><dd>{view.released[probe]}</dd></dl></section>}
        {lens==="tiles"&&<p>Blue: fine · green: extension shell · amber: transport · grey: coarse air</p>}
      </aside>
    </div>
  </main>;
}
