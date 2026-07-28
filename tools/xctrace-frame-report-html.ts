/**
 * Render a {@link FrameReport} as one self-contained interactive page.
 *
 * Everything is inlined -- no network, no build step -- so the file can be
 * opened straight from `artifacts/`, committed as evidence, or published.
 * The page is organised around the question the profile exists to answer:
 * which task is on the GPU, and how much of the machine is it actually using.
 */
import type { FrameReport } from "./xctrace-frame-report";

/** Colour a 0..1 utilisation on a red -> amber -> green ramp. */
const RAMP = `
function ramp(u,fallback){
  if(u===null||u===undefined||Number.isNaN(u)) return fallback||'#8a91a0';
  const t=Math.max(0,Math.min(1,u));
  const stops=[[0.0,[214,69,69]],[0.35,[214,140,54]],[0.7,[196,186,64]],[1.0,[72,170,110]]];
  for(let i=1;i<stops.length;i++){
    if(t<=stops[i][0]){
      const [a,ca]=stops[i-1],[b,cb]=stops[i];const k=(t-a)/(b-a);
      const c=ca.map((v,j)=>Math.round(v+(cb[j]-v)*k));
      return 'rgb('+c.join(',')+')';
    }
  }
  return 'rgb(72,170,110)';
}`;

export const renderFrameReportHtml = (report: FrameReport): string => {
  const data = JSON.stringify(report).replace(/</g, "\\u003c");

  return `<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mini dam break — GPU frame profile</title>
<style>
:root{
  --bg:#ffffff; --panel:#f6f7f9; --panel2:#eceef2; --ink:#14161a; --muted:#606874;
  --line:#d9dde3; --accent:#2f6fd0; --empty:#c8ccd4;
  --mono:ui-monospace,SFMono-Regular,Menlo,monospace;
}
@media (prefers-color-scheme:dark){
  :root{--bg:#0f1115;--panel:#171a20;--panel2:#1e222a;--ink:#e8eaee;--muted:#98a1b0;
        --line:#2a2f38;--accent:#5b9bff;--empty:#2c313a;}
}
:root[data-theme="dark"]{--bg:#0f1115;--panel:#171a20;--panel2:#1e222a;--ink:#e8eaee;
  --muted:#98a1b0;--line:#2a2f38;--accent:#5b9bff;--empty:#2c313a;}
:root[data-theme="light"]{--bg:#ffffff;--panel:#f6f7f9;--panel2:#eceef2;--ink:#14161a;
  --muted:#606874;--line:#d9dde3;--accent:#2f6fd0;--empty:#c8ccd4;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.5 ui-sans-serif,-apple-system,system-ui,sans-serif;}
.wrap{max-width:1500px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:15px;margin:34px 0 10px;letter-spacing:.02em;text-transform:uppercase;color:var(--muted)}
.sub{color:var(--muted);margin-bottom:22px;font-size:13px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(165px,1fr));gap:10px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.card .k{font-size:11px;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
.card .v{font-size:22px;font-variant-numeric:tabular-nums;margin-top:3px}
.card .n{font-size:11px;color:var(--muted);margin-top:2px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{text-align:right;padding:6px 9px;white-space:nowrap;border-bottom:1px solid var(--line)}
th:first-child,td:first-child{text-align:left;white-space:normal;min-width:260px}
th{font-size:11px;text-transform:uppercase;letter-spacing:.03em;color:var(--muted);
  cursor:pointer;user-select:none;position:sticky;top:0;background:var(--panel)}
tbody tr{cursor:pointer}
tbody tr:hover{background:var(--panel2)}
tbody tr.sel{background:color-mix(in srgb,var(--accent) 18%,transparent)}
.num{font-variant-numeric:tabular-nums;font-family:var(--mono);font-size:12px}
.bar{height:7px;border-radius:4px;background:var(--accent);display:inline-block;vertical-align:middle}
.pill{display:inline-block;padding:1px 7px;border-radius:20px;font-size:11px;
  background:var(--panel2);color:var(--muted);margin-left:6px}
.warn{border-left:3px solid #d68c36;padding-left:10px}
canvas{display:block;width:100%;border-radius:8px}
.legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:12px;margin-top:8px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px}
.detail{margin-top:12px;padding:12px;background:var(--panel2);border-radius:8px;font-size:13px}
.detail h3{margin:0 0 8px;font-size:14px;font-family:var(--mono)}
.kv{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:8px 18px}
.kv div span{color:var(--muted);font-size:11px;display:block;text-transform:uppercase}
.muted{color:var(--muted)}
.flame{font-family:var(--mono);font-size:11px}
.flame rect{stroke:var(--bg);stroke-width:.5px;cursor:pointer}
.flame text{pointer-events:none;fill:#12141a}
.tip{position:fixed;pointer-events:none;background:var(--panel);border:1px solid var(--line);
  border-radius:6px;padding:6px 9px;font-size:12px;max-width:520px;z-index:9;display:none;
  box-shadow:0 6px 20px rgba(0,0,0,.25);font-family:var(--mono)}
.row{display:flex;gap:14px;align-items:center;flex-wrap:wrap;margin-bottom:10px}
button{background:var(--panel2);color:var(--ink);border:1px solid var(--line);border-radius:6px;
  padding:5px 11px;font-size:12px;cursor:pointer}
button:hover{border-color:var(--accent)}
input[type=search]{background:var(--panel2);color:var(--ink);border:1px solid var(--line);
  border-radius:6px;padding:5px 10px;font-size:12px;min-width:220px}
</style>
<div class="wrap">
<h1>Mini dam break — GPU frame profile</h1>
<div class="sub" id="sub"></div>
<div class="cards" id="cards"></div>

<h2>One advance, end to end</h2>
<div class="panel">
  <div class="row">
    <span class="muted">Every GPU interval in one advance. Height = occupancy where measured, colour = utilisation. Click a block to drill in. <b>Pick which advance</b> below — the timeline and the occupancy grid follow it.</span>
  </div>
  <div class="row">
    <label class="muted">advance <select id="frameSel"></select></label>
    <button id="frameFirst">« first</button>
    <button id="framePrev">‹</button>
    <button id="frameNext">›</button>
    <button id="frameLast">last »</button>
    <button id="frameRep">representative</button>
    <span class="muted" id="framenote"></span>
  </div>
  <canvas id="timeline" height="260"></canvas>
  <canvas id="counters" height="120"></canvas>
  <div class="legend" id="tl-legend"></div>
  <div class="detail" id="tl-detail" style="display:none"></div>
</div>

<h2>Advance to advance — is every frame the same frame?</h2>
<div class="panel">
  <div class="row">
    <span class="muted">Each analysed advance measured on its own, in capture order. <b>Every advance must carry the same encoder and pass count</b> — that is the check that a frame boundary really is one advance and not a fragment. The cost trend shows whether the frame the profile describes is the frame the run settles into.</span>
  </div>
  <canvas id="drift" height="260"></canvas>
  <div class="legend" id="drift-legend"></div>
  <div class="row" style="margin-top:12px">
    <label class="muted">compare <select id="cmpA"></select></label>
    <label class="muted">against <select id="cmpB"></select></label>
    <span class="muted" id="cmpnote"></span>
  </div>
  <div class="scroll"><table id="cmp"><thead></thead><tbody></tbody></table></div>
</div>

<h2>Throughput and occupancy by task</h2>
<div class="panel">
  <div class="row">
    <input type="search" id="q" placeholder="filter passes…">
    <button id="csv">copy as TSV</button>
    <span class="muted" id="passnote"></span>
  </div>
  <div class="scroll"><table id="passes"><thead></thead><tbody></tbody></table></div>
  <div class="detail" id="pass-detail" style="display:none"></div>
</div>

<h2>Occupancy grid — task × time</h2>
<div class="panel">
  <div class="row"><span class="muted">Rows are the costliest tasks, columns are time bins across one advance, so the filled span shows <b>when</b> each task owns the GPU. Colour is that task\u2019s mean occupancy over the whole window &mdash; it is constant along a row, not resolved within it. For occupancy against time, read the counter traces under the timeline above.</span></div>
  <canvas id="grid" height="380"></canvas>
  <div class="legend"><span><i style="background:rgb(214,69,69)"></i>0%</span><span><i style="background:rgb(214,140,54)"></i>35%</span><span><i style="background:rgb(196,186,64)"></i>70%</span><span><i style="background:rgb(72,170,110)"></i>100%</span><span><i style="background:var(--empty)"></i>idle</span></div>
</div>

<h2>Resident SIMD groups over time</h2>
<div class="panel">
  <div class="row">
    <span class="muted"><b>This is an area chart, not a unit matrix.</b> The counter reports a single number every 10&nbsp;\u00b5s \u2014 how many SIMD groups were resident device-wide \u2014 with no per-unit identity, so there is no honest way to draw one row per execution unit. Height is that count, colour is the task that owned the GPU. A per-unit matrix would need a hardware wave trace, which Apple does not expose (see below).</span>
  </div>
  <div class="row">
    <label class="muted">Vertical resolution <select id="waverows">
      <option value="96">96</option>
      <option value="192" selected>192</option>
      <option value="384">384</option>
      <option value="768">768</option>
    </select></label>
    <span class="muted" id="wavenote"></span>
  </div>
  <canvas id="waves" height="420"></canvas>
  <div class="legend" id="waves-legend"></div>
</div>

<h2>Work placement \u2014 task \u00d7 GPU partition</h2>
<div class="panel">
  <div class="row">
    <span class="muted">Measured occupancy on each hardware counter stream. On this part the streams behave as the GPU\u2019s four partitions: a wide dispatch reads within 1\u20132% across all four, while a starved one concentrates on partition 0. <b>A row that is lit alone is work pinned to a quarter of the machine.</b> Columns are tasks, width \u221d GPU time.</span>
  </div>
  <div class="row">
    <label class="muted">Colour
      <select id="pmode">
        <option value="rel">relative to the task\u2019s own peak partition \u2014 shows imbalance</option>
        <option value="abs">absolute occupancy \u2014 shows machine usage</option>
      </select>
    </label>
  </div>
  <canvas id="placement" height="230"></canvas>
  <div class="legend" id="placement-legend"></div>
  <div class="detail" id="placement-detail" style="display:none"></div>
</div>

<h2>Machine utilisation \u2014 task \u00d7 GPU units</h2>
<div class="panel">
  <div class="row">
    <span class="muted" id="machine-scope"></span>
  </div>
  <div class="row">
    <label class="muted">Y axis granularity
      <select id="units"></select>
    </label>
    <span class="muted" id="wastenote"></span>
  </div>
  <canvas id="machine" height="430"></canvas>
  <div class="legend">
    <span><i style="background:rgb(214,69,69)"></i>lit, ALU 0%</span>
    <span><i style="background:rgb(214,140,54)"></i>35%</span>
    <span><i style="background:rgb(196,186,64)"></i>70%</span>
    <span><i style="background:rgb(72,170,110)"></i>100%</span>
    <span><i style="background:#737b89"></i>unresolved composite work</span>
    <span><i style="background:var(--empty)"></i>idle \u2014 no work resident</span>
  </div>
  <div class="detail" id="machine-detail" style="display:none"></div>
</div>

<h2>Shader entry points</h2>
<div class="panel"><div class="scroll"><table id="shaders"><thead></thead><tbody></tbody></table></div></div>

<h2>CPU flame graph</h2>
<div class="panel">
  <div class="row">
    <button id="flame-reset">reset zoom</button>
    <input type="search" id="fq" placeholder="highlight frames…">
    <span class="muted" id="flamenote"></span>
  </div>
  <div id="flame"></div>
</div>

<h2>Measurement integrity</h2>
<div class="panel" id="integrity"></div>
</div>
<div class="tip" id="tip"></div>
<script>
const R = ${data};
${RAMP}
const $ = (id) => document.getElementById(id);
const EXACT_PASSES = R.passes.filter(p=>p.exactAttribution===true);
const COMPOSITE_PASSES = R.passes.filter(p=>p.exactAttribution!==true);
const compositeMs = R.gpu.compositeMsPerFrame
  ?? COMPOSITE_PASSES.reduce((s,p)=>s+p.gpuMsPerFrame,0);
const compositeWeighted = (field) => {
  const known=COMPOSITE_PASSES.filter(p=>p[field]!==undefined&&p[field]!==null);
  const weight=known.reduce((s,p)=>s+p.gpuMsPerFrame,0);
  return weight>0 ? known.reduce((s,p)=>s+p.gpuMsPerFrame*p[field],0)/weight : null;
};
const UNRESOLVED_COMPOSITE = {
  label:'Unresolved non-target/composite interval buckets',
  gpuMsPerFrame:compositeMs,
  share:compositeMs/Math.max(R.gpu.intervalMsPerFrame||compositeMs,1e-9),
  occupancy:compositeWeighted('occupancy'),
  alu:compositeWeighted('alu'),
  counterSamples:COMPOSITE_PASSES.reduce((s,p)=>s+(p.counterSamples||0),0),
  unresolved:true,
};
const MACHINE_PASSES = EXACT_PASSES.length
  ? [UNRESOLVED_COMPOSITE,...EXACT_PASSES] : R.passes;
const primary = (label) => label.split(' \u00b7 ')[0];
const extra = (label) => label.split(' \u00b7 ').length - 1;
const fmt = (v,d=2) => (v===undefined||v===null||Number.isNaN(v)) ? '—' : v.toFixed(d);
const pct = (v,d=0) => (v===undefined||v===null||Number.isNaN(v)) ? '—' : (100*v).toFixed(d)+'%';
const tip = $('tip');
const showTip = (e,html) => { tip.innerHTML=html; tip.style.display='block';
  const x=Math.min(e.clientX+14, innerWidth-tip.offsetWidth-10);
  tip.style.left=x+'px'; tip.style.top=(e.clientY+16)+'px'; };
const hideTip = () => { tip.style.display='none'; };

// ---------- header ----------
$('sub').textContent = R.scene+' · grid '+R.grid+' · lane '+R.lane
  + ' · '+R.frames.count+' advances analysed · captured '+R.capturedAt.replace('T',' ').slice(0,19)+'Z';

const cards = [
  ['frame wall', fmt(R.gpu.wallMsPerFrame)+' ms', 'p10 '+fmt(R.frames.p10Ms)+' / p90 '+fmt(R.frames.p90Ms)],
  ['GPU busy', fmt(R.gpu.busyMsPerFrame)+' ms', pct(R.gpu.occupancy)+' of frame wall'],
  ['interval overlap', fmt(R.gpu.overlapMsPerFrame)+' ms', 'excluded from GPU busy'],
  ['exact-stage coverage', EXACT_PASSES.length?pct(R.gpu.exactIntervalCoverage,1):'\u2014',
     EXACT_PASSES.length?fmt(R.gpu.exactMsPerFrame)+' of '+fmt(R.gpu.intervalMsPerFrame)+' interval ms':'no targeted isolation'],
  ['GPU idle gaps', fmt(R.gpu.gapMsPerFrame)+' ms', 'per advance'],
  ['resident threads', R.counters.available&&R.counters.totalThreads
      ? Math.round((R.counters.meanOccupancy||0)*R.counters.totalThreads).toLocaleString():'—',
     R.counters.totalThreads? 'of '+R.counters.totalThreads.toLocaleString()+' the GPU can hold':'—'],
  ['compute occupancy', R.counters.available? pct(R.counters.meanOccupancy,1):'—',
     R.counters.available? 'mean while our work ran':'counters not captured'],
  ['ALU utilisation', R.counters.available? pct(R.counters.meanAlu,1):'—',
     R.counters.available? 'share of peak ALU':'—'],
  ['GPU traffic', R.counters.available? fmt(R.counters.meanReadGBs,1)+' + '+fmt(R.counters.meanWriteGBs,1)+' GB/s':'—','read + write'],
  ['compute passes', fmt(R.gpu.passesPerFrame,0), fmt(R.gpu.encodersPerFrame,0)+' Metal encoders/advance'],
  ['tracing cost', R.wall.distortion? fmt(R.wall.distortion)+'×':'—',
     R.wall.baselineMsPerAdvance? 'untraced '+fmt(R.wall.baselineMsPerAdvance)+' ms':'no baseline'],
];
$('cards').innerHTML = cards.map(([k,v,n])=>
  '<div class="card"><div class="k">'+k+'</div><div class="v">'+v+'</div><div class="n">'+n+'</div></div>').join('');

$('machine-scope').innerHTML = EXACT_PASSES.length
  ? '<b>Complete attributed-interval coverage:</b> exact stages are shown individually; the '
    +fmt(compositeMs)+' ms/advance that was outside this targeted capture is the grey unresolved column. '
    +'Only '+fmt(R.gpu.exactMsPerFrame)+' ms ('+pct(R.gpu.exactIntervalCoverage,1)
    +') has exact stage identity. Interval records overlap by '+fmt(R.gpu.overlapMsPerFrame)
    +' ms, so widths are <b>not</b> a wall-clock partition.'
  : 'No stages were exactly isolated, so columns are composite encoder buckets. '
    +'Column width is attributed GPU interval time; height is measured occupancy. '
    +'<b>Do not interpret a composite bucket\u2019s first label as its complete contents.</b>';

// ---------- frame selection ----------
// The report retains whole frames, not just a representative one, so the
// timeline can be pointed at any of them. Advance numbers are only shown when
// the capture spans the run; otherwise a frame is identified by its position
// in the analysed window.
const CAPS = (R.frames.captures||[]);
const SAMPLES = (R.frames.samples||[]);
const FIRST_ADVANCE = R.frames.firstAdvance;
const frameName = (i) => FIRST_ADVANCE!==undefined&&FIRST_ADVANCE!==null
  ? 'advance '+(FIRST_ADVANCE+i) : 'frame '+(i+1);
let capIndex = Math.min(R.frames.representative||0, Math.max(CAPS.length-1,0));
const capture = () => CAPS[capIndex] || {index:0,durationUs:R.timeline.frameDuration,intervals:R.timeline.intervals};
let TL = {frameDuration: capture().durationUs, intervals: capture().intervals};
const isRepresentative = () => capIndex===(R.frames.representative||0);
// Lanes come from every retained frame, not just the one on screen: a channel
// that appears in only one advance must still have a row to draw into.
const lanes = [...new Set([].concat(R.timeline.intervals||[],
  ...CAPS.map(c=>c.intervals)).map(i=>i.channel))];
const laneY = {}; lanes.forEach((l,i)=>laneY[l]=i);
const tlc = $('timeline'), tctx = tlc.getContext('2d');
let tlSel = null;

function drawTimeline(){
  const w = tlc.clientWidth, h = 260;
  tlc.width = w*devicePixelRatio; tlc.height = h*devicePixelRatio;
  tctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  tctx.clearRect(0,0,w,h);
  const pad=54, laneH=(h-pad-14)/Math.max(lanes.length,1);
  const sx = (t)=> pad + (t/TL.frameDuration)*(w-pad-12);
  const css = getComputedStyle(document.documentElement);
  tctx.font='11px ui-monospace,monospace';
  lanes.forEach((l,i)=>{
    tctx.fillStyle=css.getPropertyValue('--muted');
    tctx.fillText(l, 6, 14+i*laneH+laneH/2);
    tctx.strokeStyle=css.getPropertyValue('--line');
    tctx.beginPath(); tctx.moveTo(pad,14+i*laneH+laneH); tctx.lineTo(w-12,14+i*laneH+laneH); tctx.stroke();
  });
  TL.intervals.forEach((iv,idx)=>{
    const x=sx(iv.start), x2=sx(iv.start+iv.duration);
    const y=14+laneY[iv.channel]*laneH;
    const occ = iv.occupancy;
    const hgt = occ!==null&&occ!==undefined ? Math.max(3,(laneH-8)*Math.max(0.05,occ)) : laneH-8;
    tctx.fillStyle = occ!==null&&occ!==undefined ? ramp(occ)
      : 'hsl('+(205+(hash(iv.label)%120))+',45%,'+(38+(hash(iv.label)%22))+'%)';
    tctx.globalAlpha = (tlSel===null||tlSel===idx)?1:0.32;
    tctx.fillRect(x, y+(laneH-8-hgt)+2, Math.max(1.2,x2-x), hgt);
  });
  tctx.globalAlpha=1;
  // axis
  tctx.fillStyle=css.getPropertyValue('--muted');
  for(let k=0;k<=5;k++){
    const t=TL.frameDuration*k/5;
    tctx.fillText((t/1000).toFixed(1)+' ms', sx(t), h-2);
  }
  $('tl-legend').innerHTML='<span>frame duration '+(TL.frameDuration/1000).toFixed(2)+' ms</span>'
    +'<span>'+TL.intervals.length+' GPU intervals</span>'
    +'<span class="muted">bar height = occupancy, colour = utilisation</span>';
}
function tlHit(e){
  const r=tlc.getBoundingClientRect(), w=tlc.clientWidth, h=260;
  const pad=54, laneH=(h-pad-14)/Math.max(lanes.length,1);
  const t=((e.clientX-r.left)-pad)/(w-pad-12)*TL.frameDuration;
  const lane=Math.floor(((e.clientY-r.top)-14)/laneH);
  let best=null,bd=1e18;
  TL.intervals.forEach((iv,idx)=>{
    if(laneY[iv.channel]!==lane) return;
    if(t>=iv.start && t<=iv.start+iv.duration){ const d=Math.abs(t-iv.start); if(d<bd){bd=d;best=idx;} }
  });
  return best;
}
tlc.addEventListener('mousemove',(e)=>{
  const i=tlHit(e); if(i===null){hideTip();return;}
  const iv=TL.intervals[i];
  showTip(e,'<b>'+esc(primary(iv.label))+(extra(iv.label)?' (+'+extra(iv.label)+' passes)':'')+'</b><br>'+(iv.duration).toFixed(1)+' µs · '+iv.channel
    +(iv.occupancy!=null?'<br>occupancy '+pct(iv.occupancy,1)+' · ALU '+pct(iv.alu,1):'')
    +(iv.merged?'<br><i>merged interval</i>':''));
});
tlc.addEventListener('mouseleave',hideTip);
tlc.addEventListener('click',(e)=>{
  const i=tlHit(e); tlSel = (i===tlSel)?null:i; drawTimeline();
  const d=$('tl-detail');
  if(tlSel===null){ d.style.display='none'; return; }
  const iv=TL.intervals[tlSel];
  d.style.display='block';
  d.innerHTML='<h3>'+esc(primary(iv.label))+'</h3><div class="kv">'
    +'<div><span>duration</span>'+iv.duration.toFixed(1)+' µs</div>'
    +'<div><span>starts at</span>'+(iv.start/1000).toFixed(3)+' ms</div>'
    +'<div><span>channel</span>'+iv.channel+'</div>'
    +'<div><span>occupancy</span>'+pct(iv.occupancy,1)+'</div>'
    +'<div><span>ALU</span>'+pct(iv.alu,1)+'</div>'
    +'</div>';
  selectPass(iv.label);
});

// ---------- counter strip ----------
const cc=$('counters'), cctx=cc.getContext('2d');
function drawCounters(){
  const w=cc.clientWidth,h=120;
  cc.width=w*devicePixelRatio; cc.height=h*devicePixelRatio;
  cctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  cctx.clearRect(0,0,w,h);
  const css=getComputedStyle(document.documentElement);
  const series=R.counters.frameSeries||[];
  if(!series.length){
    cctx.fillStyle=css.getPropertyValue('--muted'); cctx.font='12px ui-sans-serif';
    cctx.fillText('GPU counters were not captured for this run (rerun with --counters).',54,60);
    return;
  }
  // The counter traces were reduced for the representative advance only, so
  // they are not redrawn against a different frame's clock -- that would align
  // one advance's samples to another advance's task boundaries.
  if(!isRepresentative()){
    cctx.fillStyle=css.getPropertyValue('--muted'); cctx.font='12px ui-sans-serif';
    cctx.fillText('Counter traces are recorded for the representative advance; '
      +'switch back to it to read them against time.',54,60);
    return;
  }
  const pad=54;
  const sx=(t)=>pad+(t/TL.frameDuration)*(w-pad-12);
  const colours={'Compute Occupancy':'#5b9bff','ALU Utilization':'#48aa6e',
    'GPU Last Level Cache Utilization':'#d68c36','Buffer Load Utilization':'#b070d0'};
  cctx.strokeStyle=css.getPropertyValue('--line');
  cctx.beginPath();cctx.moveTo(pad,h-16);cctx.lineTo(w-12,h-16);cctx.stroke();
  series.forEach(s=>{
    cctx.strokeStyle=colours[s.name]||'#888'; cctx.lineWidth=1.5; cctx.beginPath();
    s.points.forEach((p,i)=>{ const x=sx(p.t), y=(h-16)-(p.v/100)*(h-30);
      i?cctx.lineTo(x,y):cctx.moveTo(x,y); });
    cctx.stroke();
  });
  cctx.fillStyle=css.getPropertyValue('--muted'); cctx.font='11px ui-monospace,monospace';
  cctx.fillText('100%',6,16); cctx.fillText('0%',6,h-16);
  let lx=pad;
  series.forEach(s=>{ cctx.fillStyle=colours[s.name]||'#888';
    cctx.fillRect(lx,h-9,9,9); cctx.fillStyle=css.getPropertyValue('--muted');
    cctx.fillText(s.name,lx+13,h-1); lx+=cctx.measureText(s.name).width+34; });
}

// ---------- pass table ----------
const COLS = [
  ['label','task',s=>s],
  ['gpuMsPerFrame','ms/advance',v=>fmt(v,3)],
  ['share','% GPU',v=>pct(v,1)],
  ['callsPerFrame','calls',v=>fmt(v,1)],
  ['meanMicroseconds','µs each',v=>fmt(v,0)],
  ['occupancy','occupancy',v=>pct(v,1)],
  ['alu','ALU',v=>pct(v,1)],
  ['readGBs','read GB/s',v=>fmt(v,1)],
  ['writeGBs','write GB/s',v=>fmt(v,1)],
  ['limiter','limiter',v=>v||'—'],
  ['imbalance','placement',v=>v===undefined?'—':fmt(v,1)+'×'],
];
let sortKey='gpuMsPerFrame', sortDir=-1, selected=null;
function renderPasses(){
  const q=$('q').value.toLowerCase();
  const rows=R.passes.filter(p=>!q||p.label.toLowerCase().includes(q));
  rows.sort((a,b)=>{const x=a[sortKey],y=b[sortKey];
    return (typeof x==='string'? x.localeCompare(y) : (x||0)-(y||0))*sortDir;});
  const max=Math.max(...R.passes.map(p=>p.gpuMsPerFrame),1e-9);
  $('passes').querySelector('thead').innerHTML='<tr>'+COLS.map(c=>
    '<th data-k="'+c[0]+'">'+c[1]+(sortKey===c[0]?(sortDir<0?' ▾':' ▴'):'')+'</th>').join('')+'</tr>';
  $('passes').querySelector('tbody').innerHTML=rows.map(p=>
    '<tr data-l="'+escapeAttr(p.label)+'"'+(selected===p.label?' class="sel"':'')+'>'
    +'<td><span title="'+escapeAttr(p.label)+'">'+esc(primary(p.label))+'</span>'
      +(extra(p.label)?'<span class="pill">+'+extra(p.label)+' passes</span>':'')
      +(p.exactAttribution===false?'<span class="pill">composite / outside target</span>':'')
      +'<div><span class="bar" style="width:'+(100*p.gpuMsPerFrame/max).toFixed(1)+'%"></span></div></td>'
    +COLS.slice(1).map(c=>'<td class="num">'+c[2](p[c[0]])+'</td>').join('')+'</tr>').join('');
  $('passnote').textContent=rows.length+' of '+R.passes.length+' tasks';
  $('passes').querySelectorAll('thead th').forEach(th=>th.onclick=()=>{
    const k=th.dataset.k; if(k===sortKey) sortDir=-sortDir; else {sortKey=k;sortDir=-1;} renderPasses();});
  $('passes').querySelectorAll('tbody tr').forEach(tr=>tr.onclick=()=>selectPass(tr.dataset.l));
}
function esc(s){const d=document.createElement('div');d.textContent=s;return d.innerHTML;}
function escapeAttr(s){return esc(s).replace(/"/g,'&quot;');}
function selectPass(label){
  selected = (selected===label)? null : label;
  renderPasses();
  const d=$('pass-detail');
  const p=R.passes.find(x=>x.label===selected);
  if(!p){ d.style.display='none'; return; }
  d.style.display='block';
  const sh=(p.shaders||[]).map(s=>'<tr><td>'+esc(s.name)+'</td><td class="num">'+s.samples+'</td></tr>').join('');
  const constituents = p.label.split(' \u00b7 ');
  d.innerHTML='<h3>'+esc(primary(p.label))+'</h3><div class="kv">'
   +'<div><span>GPU per advance</span>'+fmt(p.gpuMsPerFrame,3)+' ms</div>'
   +'<div><span>share of GPU</span>'+pct(p.share,1)+'</div>'
   +'<div><span>invocations</span>'+fmt(p.callsPerFrame,1)+'</div>'
   +'<div><span>mean</span>'+fmt(p.meanMicroseconds,0)+' µs</div>'
   +'<div><span>compute occupancy</span>'+pct(p.occupancy,1)+'</div>'
   +'<div><span>ALU utilisation</span>'+pct(p.alu,1)+'</div>'
   +'<div><span>bandwidth</span>'+fmt(p.readGBs,1)+' r / '+fmt(p.writeGBs,1)+' w GB/s</div>'
   +'<div><span>limiter</span>'+(p.limiter||'—')+'</div>'
   +'<div><span>counter samples</span>'+(p.counterSamples||0)+'</div>'
   +'<div><span>uncontended</span>'+pct(p.exclusiveShare,0)+' of its GPU time</div>'
   +'</div>'
   +(p.counters?'<div style="margin-top:10px"><b>all measured counters</b><table>'
      +Object.entries(p.counters).sort((a,b)=>b[1]-a[1])
        .map(([k,v])=>'<tr><td>'+esc(k)+'</td><td class="num">'+v.toFixed(2)
          +(/Bandwidth/.test(k)?' GB/s':'%')+'</td></tr>').join('')
      +'</table></div>':'')
   +(constituents.length>1?'<div style="margin-top:10px"><b>'+constituents.length
      +' compute passes in this Metal encoder</b><ol style="margin:6px 0 0;padding-left:22px">'
      +constituents.map(c=>'<li>'+esc(c)+'</li>').join('')+'</ol></div>':'')
   +(sh?'<div style="margin-top:10px"><b>shader entry points</b><table>'+sh+'</table></div>':'');
}
$('q').oninput=renderPasses;
$('csv').onclick=()=>{
  const head=COLS.map(c=>c[1]).join('\\t');
  const body=R.passes.map(p=>COLS.map(c=>p[c[0]]).join('\\t')).join('\\n');
  navigator.clipboard.writeText(head+'\\n'+body);
  $('csv').textContent='copied'; setTimeout(()=>$('csv').textContent='copy as TSV',1200);
};

// ---------- occupancy grid ----------
const gc=$('grid'), gctx=gc.getContext('2d');
const GRID_BINS=90;
/** When each task owns the GPU in the frame on screen, as GRID_BINS buckets. */
function gridBins(pass){
  const bins=new Array(GRID_BINS).fill(null);
  const bw=TL.frameDuration/GRID_BINS;
  TL.intervals.forEach(iv=>{
    if(iv.label!==pass.label) return;
    const from=Math.max(0,Math.floor(iv.start/bw));
    const to=Math.min(GRID_BINS-1,Math.floor((iv.start+iv.duration)/bw));
    for(let b=from;b<=to;b++) bins[b]=(pass.occupancy===undefined?null:pass.occupancy);
  });
  return bins;
}
function drawGrid(){
  const top=R.passes.slice(0,18);
  const BINS=GRID_BINS;
  const w=gc.clientWidth, rowH=19, h=top.length*rowH+34;
  gc.width=w*devicePixelRatio; gc.height=h*devicePixelRatio; gc.style.height=h+'px';
  gctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  const css=getComputedStyle(document.documentElement);
  gctx.clearRect(0,0,w,h);
  const pad=270, cw=(w-pad-12)/BINS;
  gctx.font='11px ui-sans-serif';
  top.forEach((p,r)=>{
    gctx.fillStyle=css.getPropertyValue('--ink');
    const lead=primary(p.label);
    const name=(lead.length>36?lead.slice(0,35)+'…':lead)+(extra(p.label)?' (+'+extra(p.label)+')':'');
    gctx.fillText(name,6,16+r*rowH+11);
    // Binned from the frame on screen rather than from the precomputed grid,
    // so switching advances moves the spans instead of leaving the
    // representative frame's layout behind.
    const bins=gridBins(p);
    for(let b=0;b<BINS;b++){
      const v=bins[b];
      gctx.fillStyle = (v===undefined||v===null)? css.getPropertyValue('--empty') : ramp(v);
      gctx.globalAlpha = (v===undefined||v===null)?0.35:1;
      gctx.fillRect(pad+b*cw, 16+r*rowH+2, Math.max(1,cw-1), rowH-5);
    }
  });
  gctx.globalAlpha=1;
  gctx.fillStyle=css.getPropertyValue('--muted');
  for(let k=0;k<=5;k++) gctx.fillText((TL.frameDuration*k/5/1000).toFixed(1)+' ms', pad+(w-pad-12)*k/5, h-6);
}

// ---------- execution units x time ----------
const wc=$('waves'), wctx=wc.getContext('2d');
let WROWS=192;
function taskColour(label){
  if(!label) return null;
  const h=hash(label);
  return 'hsl('+(h%360)+',68%,'+(52+(h%3)*7)+'%)';
}
function drawWaves(){
  const trace=(R.counters&&R.counters.occupancyTrace)||[];
  const w=wc.clientWidth;
  const css=getComputedStyle(document.documentElement);
  if(!trace.length){
    wc.height=46*devicePixelRatio; wc.style.height='46px';
    wctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
    wctx.clearRect(0,0,w,46);
    wctx.fillStyle=css.getPropertyValue('--muted'); wctx.font='12px ui-sans-serif';
    wctx.fillText('No occupancy samples in the representative advance. Rerun with --counters.',8,26);
    $('waves-legend').textContent=''; return;
  }
  const padL=52,padR=10,padT=8,padB=26;
  const rowH=2, plotH=WROWS*rowH, h=padT+padB+plotH;
  wc.width=w*devicePixelRatio; wc.height=h*devicePixelRatio; wc.style.height=h+'px';
  wctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  wctx.clearRect(0,0,w,h);
  const plotW=Math.max(10,w-padL-padR);
  const dur=TL.frameDuration||1;
  const sx=(t)=>padL+(t/dur)*plotW;
  wctx.fillStyle=css.getPropertyValue('--empty'); wctx.globalAlpha=0.28;
  wctx.fillRect(padL,padT,plotW,plotH);
  wctx.globalAlpha=1;
  for(let i=0;i<trace.length;i++){
    const s=trace[i];
    // Device occupancy: the mean across the hardware streams.
    let occ=0; for(let k=0;k<s.p.length;k++) occ+=s.p[k];
    occ = s.p.length? occ/s.p.length : 0;
    const lit=Math.round(occ*WROWS);
    if(lit<=0) continue;
    const x0=sx(s.t);
    const x1=i+1<trace.length? sx(trace[i+1].t) : x0+plotW/Math.max(trace.length,1);
    wctx.fillStyle=taskColour(s.label)||'#6b7280';
    wctx.fillRect(x0, padT+plotH-lit*rowH, Math.max(1,x1-x0-0.3), lit*rowH-0.4);
  }
  wctx.fillStyle=css.getPropertyValue('--muted'); wctx.font='10px ui-monospace,monospace';
  wctx.textAlign='right';
  const totalSlots=R.counters.totalSlots||WROWS;
  for(let k=0;k<=4;k++){
    const frac=k/4;
    wctx.fillText(String(Math.round(frac*totalSlots)), padL-6, padT+plotH-frac*plotH+3);
  }
  wctx.textAlign='left';
  for(let k=0;k<=5;k++) wctx.fillText((dur*k/5/1000).toFixed(1)+' ms', sx(dur*k/5), h-8);
  const slotsPerRow=Math.round(totalSlots/WROWS);
  $('wavenote').textContent=trace.length+' samples across '+(dur/1000).toFixed(1)
    +' ms \u00b7 full scale '+totalSlots+' SIMD groups = '
    +(totalSlots*(R.counters.threadsPerSlot||32)).toLocaleString()+' threads';
  const top=R.passes.slice(0,10).filter(p=>p.gpuMsPerFrame>0.05);
  $('waves-legend').innerHTML=top.map(p=>'<span><i style="background:'+taskColour(p.label)
    +'"></i>'+esc(primary(p.label).slice(0,34))+'</span>').join('');
}
$('waverows').onchange=()=>{WROWS=Number($('waverows').value);drawWaves();};

// ---------- work placement (task x GPU partition) ----------
const pc=$('placement'), pctx=pc.getContext('2d');
let pcols=[], pSel=null, PMODE='rel';
function drawPlacement(){
  const P=R.counters.partitionCount||0;
  const w=pc.clientWidth;
  if(!P){ pc.height=40*devicePixelRatio; pc.style.height='40px';
    pctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
    pctx.clearRect(0,0,w,40);
    pctx.fillStyle=getComputedStyle(document.documentElement).getPropertyValue('--muted');
    pctx.font='12px ui-sans-serif';
    pctx.fillText('GPU counters were not captured, so placement is unknown. Rerun with --counters.',8,24);
    $('placement-legend').textContent=''; return; }
  const tasks=R.passes.filter(p=>p.gpuMsPerFrame>0.005&&p.partitions&&p.partitions.length===P).slice(0,26);
  const total=tasks.reduce((s,p)=>s+p.gpuMsPerFrame,0)||1;
  const padL=92,padR=10,padT=12,padB=126;
  const h=P*30+padT+padB;
  pc.width=w*devicePixelRatio; pc.height=h*devicePixelRatio; pc.style.height=h+'px';
  pctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  const css=getComputedStyle(document.documentElement);
  pctx.clearRect(0,0,w,h);
  const plotW=Math.max(10,w-padL-padR), rowH=30;
  pctx.font='11px ui-monospace,monospace';
  for(let r=0;r<P;r++){
    pctx.fillStyle=css.getPropertyValue('--muted');
    pctx.fillText('partition '+r, 6, padT+r*rowH+19);
  }
  let x=padL; pcols=[];
  tasks.forEach((p,i)=>{
    const cw=Math.max(2, plotW*p.gpuMsPerFrame/total);
    const dim=(pSel!==null&&pSel!==i)?0.35:1;
    for(let r=0;r<P;r++){
      pctx.globalAlpha=dim;
      if(PMODE==='abs'){ pctx.fillStyle=ramp(p.partitions[r]); }
      else {
        // Share of this task's own best partition. A lopsided task shows one
        // bright cell over dark ones no matter how small its absolute
        // occupancy is; a balanced task is a flat column.
        const peak=Math.max(...p.partitions);
        const rel=peak>0? p.partitions[r]/peak : 0;
        const level=Math.round(28+197*rel);
        pctx.fillStyle='rgb('+Math.round(28+40*rel)+','+Math.round(60+90*rel)+','+level+')';
      }
      pctx.fillRect(x, padT+r*rowH+2, Math.max(1,cw-1), rowH-4);
    }
    pctx.globalAlpha=1;
    if(cw>=8){
      pctx.save();
      pctx.translate(x+cw/2+4, padT+P*rowH+9);
      pctx.rotate(-Math.PI/4); pctx.textAlign='right';
      pctx.fillStyle=css.getPropertyValue(pSel===i?'--ink':'--muted');
      const lead=primary(p.label);
      pctx.fillText(lead.length>24?lead.slice(0,23)+'\u2026':lead,0,0);
      pctx.restore(); pctx.textAlign='left';
    }
    pcols.push({x,cw,p});
    x+=cw;
  });
  const worst=[...tasks].sort((a,b)=>(b.imbalance||0)-(a.imbalance||0))[0];
  $('placement-legend').innerHTML=(PMODE==='rel'
      ? '<span>bright = this task\u2019s busiest partition, dark = idle relative to it</span>'
      : '<span>colour is absolute occupancy on the red\u2192green utilisation ramp</span>')
    +'<span>device mean per partition: '
    +R.counters.partitionOccupancy.map(v=>pct(v,1)).join(' \u00b7 ')+'</span>'
    +(worst?'<span>most lopsided: <b>'+esc(primary(worst.label))+'</b> at '
      +fmt(worst.imbalance,1)+'\u00d7 peak-over-mean</span>':'');
}
function pHit(e){const r=pc.getBoundingClientRect(),px=e.clientX-r.left;
  for(let i=0;i<pcols.length;i++){const c=pcols[i]; if(px>=c.x&&px<c.x+c.cw)return i;} return null;}
pc.addEventListener('mousemove',(e)=>{
  const i=pHit(e); if(i===null){hideTip();return;}
  const p=pcols[i].p;
  showTip(e,'<b>'+esc(primary(p.label))+'</b><br>'+fmt(p.gpuMsPerFrame,3)+' ms/advance<br>'
    +p.partitions.map((v,r)=>'partition '+r+': '+pct(v,2)).join('<br>')
    +'<br>peak/mean '+fmt(p.imbalance,2)+'\u00d7');
});
pc.addEventListener('mouseleave',hideTip);
pc.addEventListener('click',(e)=>{
  const i=pHit(e); pSel=(i===pSel)?null:i; drawPlacement();
  const d=$('placement-detail');
  if(pSel===null){d.style.display='none';return;}
  const p=pcols[pSel].p; d.style.display='block';
  d.innerHTML='<h3>'+esc(primary(p.label))+'</h3><div class="kv">'
    +p.partitions.map((v,r)=>'<div><span>partition '+r+'</span>'+pct(v,2)+'</div>').join('')
    +'<div><span>peak / mean</span>'+fmt(p.imbalance,2)+'\u00d7</div>'
    +'<div><span>device occupancy</span>'+pct(p.occupancy,2)+'</div></div>'
    +'<div class="muted" style="margin-top:8px">'
    +(p.imbalance>1.5
      ? 'Work is concentrated on one partition \u2014 too few workgroups to reach the whole GPU.'
      : 'Work is spread evenly across partitions; low occupancy here is per-core, not placement.')
    +'</div>';
  selectPass(p.label);
});

// ---------- machine utilisation (task x GPU units) ----------
const mc=$('machine'), mctx=mc.getContext('2d');
let UNITS=32, mcols=[], mSel=null;
// The hardware model is recovered from counter quantisation, not hardcoded:
// occupancy readings land on exact multiples of 100/slotsPerPartition.
const HW=R.counters;
const UNIT_CHOICES = HW.totalSlots ? [
  {n:32, label:'32 \u2014 GPU cores'},
  {n:HW.partitionCount, label:HW.partitionCount+' \u2014 GPU partitions'},
  {n:HW.totalSlots, label:HW.totalSlots+' \u2014 SIMD groups ('+HW.threadsPerSlot+' threads each)'},
  {n:HW.totalThreads, label:HW.totalThreads.toLocaleString()+' \u2014 threads'},
] : [{n:32,label:'32 \u2014 GPU cores'}];
$('units').innerHTML = UNIT_CHOICES.map(c=>'<option value="'+c.n+'">'+c.label+'</option>').join('');
const unitName = () => (UNIT_CHOICES.find(c=>c.n===UNITS)||UNIT_CHOICES[0]).label.split(' \u2014 ')[1];
function drawMachine(){
  const tasks=MACHINE_PASSES.filter(p=>p.gpuMsPerFrame>0.001).slice(0,96);
  const total=tasks.reduce((s,p)=>s+p.gpuMsPerFrame,0)||1;
  const w=mc.clientWidth, h=430;
  mc.width=w*devicePixelRatio; mc.height=h*devicePixelRatio;
  mctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  const css=getComputedStyle(document.documentElement);
  const idle=css.getPropertyValue('--empty');
  mctx.clearRect(0,0,w,h);
  const padL=58, padR=10, padT=10, padB=132;
  const plotW=Math.max(10,w-padL-padR), plotH=h-padT-padB;
  const cellH=plotH/UNITS;
  mctx.font='11px ui-monospace,monospace';
  // Y axis: gridlines every quarter of the machine.
  mctx.strokeStyle=css.getPropertyValue('--line');
  mctx.fillStyle=css.getPropertyValue('--muted');
  mctx.textAlign='right';
  for(let k=0;k<=4;k++){
    const y=padT+plotH-(k/4)*plotH;
    mctx.globalAlpha=k===0?1:0.5;
    mctx.beginPath(); mctx.moveTo(padL,y); mctx.lineTo(padL+plotW,y); mctx.stroke();
    mctx.globalAlpha=1;
    const tick=Math.round(k/4*UNITS);
    mctx.fillText((tick>=10000?(tick/1000).toFixed(0)+'k':String(tick))+(k===4?'':''), padL-6, y+3);
  }
  mctx.textAlign='left';
  mctx.save(); mctx.translate(11, padT+plotH/2); mctx.rotate(-Math.PI/2);
  mctx.textAlign='center'; mctx.fillText(unitName()+' occupied', 0, 0); mctx.restore();
  mctx.textAlign='left';
  let x=padL; mcols=[];
  let idleMs=0;
  tasks.forEach((p,i)=>{
    const cw=Math.max(2, plotW*p.gpuMsPerFrame/total);
    const occ=(p.occupancy===null||p.occupancy===undefined)?null:p.occupancy;
    const litExact=occ===null?null:occ*UNITS;
    if(occ!==null) idleMs += p.gpuMsPerFrame*(1-occ);
    const dim=(mSel!==null&&mSel!==i)?0.4:1;
    for(let u=0;u<UNITS;u++){
      const y=padT+plotH-(u+1)*cellH;
      let frac=0;
      if(litExact!==null){ frac = Math.max(0, Math.min(1, litExact-u)); }
      if(frac>0){
        mctx.globalAlpha=dim; mctx.fillStyle=p.unresolved?'#737b89'
          :ramp(p.alu===null||p.alu===undefined?occ:p.alu);
        mctx.fillRect(x, y+(1-frac)*cellH, Math.max(1,cw-1), Math.max(0.7,cellH*frac-0.5));
      }
      if(frac<1){
        mctx.globalAlpha=dim*0.30; mctx.fillStyle=idle;
        mctx.fillRect(x, y, Math.max(1,cw-1), Math.max(0.7,cellH*(1-frac)-0.5));
      }
    }
    mctx.globalAlpha=1;
    mcols.push({x,cw,p,occ});
    // rotated task label
    if(cw>=8){
      mctx.save();
      mctx.translate(x+cw/2+4, padT+plotH+9);
      mctx.rotate(-Math.PI/4);
      mctx.textAlign='right';
      mctx.fillStyle=css.getPropertyValue(mSel===i?'--ink':'--muted');
      const lead=primary(p.label);
      const t=lead.length>24?lead.slice(0,23)+'\u2026':lead;
      mctx.fillText(t,0,0);
      mctx.restore();
      mctx.textAlign='left';
    }
    x+=cw;
  });
  const litMs=tasks.reduce((s,p)=>s+(p.occupancy?p.gpuMsPerFrame*p.occupancy:0),0);
  $('wastenote').innerHTML = R.counters.available
    ? '<b>'+idleMs.toFixed(1)+' ms/advance of idle machine</b> against '+litMs.toFixed(2)
      +' ms of occupied machine \u2014 '+(100*litMs/(litMs+idleMs)).toFixed(1)
      +'% of the attributed interval GPU-time area is doing work'
      +(EXACT_PASSES.length?' (grey includes unresolved stage mixtures).':'.')
    : 'GPU counters were not captured, so occupancy is unknown; rerun with --counters.';
}
function mHit(e){
  const r=mc.getBoundingClientRect(); const px=e.clientX-r.left;
  for(let i=0;i<mcols.length;i++){ const c=mcols[i]; if(px>=c.x&&px<c.x+c.cw) return i; }
  return null;
}
mc.addEventListener('mousemove',(e)=>{
  const i=mHit(e); if(i===null){hideTip();return;}
  const c=mcols[i];
  showTip(e,'<b>'+esc(primary(c.p.label))+'</b><br>'+fmt(c.p.gpuMsPerFrame,3)+' ms/advance \u00b7 '
    +pct(c.p.share,1)+' of GPU<br>occupancy '+pct(c.occ,1)+' = '
    +(c.occ===null?'unknown':(c.occ*UNITS).toFixed(2)+' of '+UNITS+' units')
    +(c.p.residentThreads!==undefined
      ? '<br><b>'+Math.round(c.p.residentThreads).toLocaleString()+' resident threads</b> of '
        +HW.totalThreads.toLocaleString()+' ('+fmt(c.p.residentSlots,0)+' SIMD groups)' : '')
    +'<br>ALU '+pct(c.p.alu,1)+' \u00b7 '+(c.p.limiter||'no limiter engaged')
    +(c.p.unresolved?'<br><b>Aggregate only: recapture these labels for micro-stage identity.</b>':''));
});
mc.addEventListener('mouseleave',hideTip);
mc.addEventListener('click',(e)=>{
  const i=mHit(e); mSel=(i===mSel)?null:i; drawMachine();
  const d=$('machine-detail');
  if(mSel===null){d.style.display='none';return;}
  const c=mcols[mSel]; d.style.display='block';
  const idleMs=c.occ===null?null:c.p.gpuMsPerFrame*(1-c.occ);
  d.innerHTML='<h3>'+esc(primary(c.p.label))+'</h3><div class="kv">'
    +'<div><span>GPU per advance</span>'+fmt(c.p.gpuMsPerFrame,3)+' ms</div>'
    +'<div><span>units occupied</span>'+(c.occ===null?'\u2014':(c.occ*UNITS).toFixed(2)+' / '+UNITS)+'</div>'
    +'<div><span>resident threads</span>'+(c.p.residentThreads===undefined?'\u2014'
        :Math.round(c.p.residentThreads).toLocaleString()+' / '+HW.totalThreads.toLocaleString())+'</div>'
    +'<div><span>SIMD groups</span>'+(c.p.residentSlots===undefined?'\u2014'
        :fmt(c.p.residentSlots,0)+' / '+HW.totalSlots)+'</div>'
    +'<div><span>idle machine-time</span>'+(idleMs===null?'\u2014':fmt(idleMs,3)+' ms/advance')+'</div>'
    +'<div><span>ALU utilisation</span>'+pct(c.p.alu,1)+'</div>'
    +'<div><span>limiter</span>'+(c.p.limiter||'none engaged')+'</div>'
    +'<div><span>counter samples</span>'+(c.p.counterSamples||0)+'</div>'
    +'</div>';
  if(!c.p.unresolved) selectPass(c.p.label);
});
$('pmode').onchange=()=>{PMODE=$('pmode').value;drawPlacement();};
$('units').onchange=()=>{UNITS=Number($('units').value);drawMachine();};

// ---------- shaders ----------
function renderShaders(){
  const rows=R.shaders||[];
  const t=$('shaders');
  if(!rows.length){ t.parentElement.innerHTML='<span class="muted">No shader-profiler samples in this capture. '
    +'The shader profiler registers pipelines when they are created, so a counter run that attaches after '
    +'construction never sees them. For per-shader attribution, record in launch mode with a small <code>--steps</code>.</span>'; return; }
  t.querySelector('thead').innerHTML='<tr><th>shader entry point</th><th>samples</th><th>share</th><th>pipelines</th></tr>';
  t.querySelector('tbody').innerHTML=rows.slice(0,60).map(s=>
    '<tr><td>'+esc(s.name)+'</td><td class="num">'+s.samples+'</td><td class="num">'+pct(s.share,1)
    +'</td><td class="num">'+s.pipelines+'</td></tr>').join('');
}

// ---------- flame graph ----------
function drawFlame(){
  const host=$('flame'); const W=host.clientWidth||1200; const H=17;
  let root=R.cpu.flame, focus=root;
  const q=($('fq').value||'').toLowerCase();
  const depthOf=(n)=>1+Math.max(0,...n.children.map(depthOf));
  const total=depthOf(root);
  const svg=['<svg class="flame" width="'+W+'" height="'+(total*H+4)+'">'];
  const walk=(n,d,x0,w0)=>{
    if(w0<0.4) return;
    const hit=q&&n.name.toLowerCase().includes(q);
    const hue=hit?'#d68c36':'hsl('+(28+(hash(n.name)%26))+',72%,'+(58+(hash(n.name)%14))+'%)';
    svg.push('<rect x="'+x0.toFixed(1)+'" y="'+(d*H)+'" width="'+w0.toFixed(1)+'" height="'+(H-1)
      +'" fill="'+hue+'" data-n="'+escapeAttr(n.name)+'" data-v="'+n.value+'"/>');
    if(w0>36) svg.push('<text x="'+(x0+3).toFixed(1)+'" y="'+(d*H+12)+'">'
      +esc(n.name.length>Math.floor(w0/6.2)?n.name.slice(0,Math.floor(w0/6.2)-1)+'…':n.name)+'</text>');
    let cx=x0;
    for(const c of n.children){ const cw=w0*c.value/n.value; walk(c,d+1,cx,cw); cx+=cw; }
  };
  walk(focus,0,0,W);
  svg.push('</svg>');
  host.innerHTML=svg.join('');
  host.querySelectorAll('rect').forEach(r=>{
    r.onmousemove=(e)=>showTip(e,'<b>'+r.dataset.n+'</b><br>'+r.dataset.v+' samples ('
      +(100*r.dataset.v/R.cpu.flame.value).toFixed(1)+'% of CPU time)');
    r.onmouseleave=hideTip;
  });
  $('flamenote').textContent=R.cpu.samples+' samples · '+R.cpu.runningSamples+' running';
}
function hash(s){let h=0;for(let i=0;i<s.length;i++)h=(h*31+s.charCodeAt(i))|0;return Math.abs(h);}
$('fq').oninput=drawFlame; $('flame-reset').onclick=()=>{$('fq').value='';drawFlame();};

// ---------- advance-to-advance drift ----------
const dc=$('drift'), dctx=dc.getContext('2d');
const DRIFT_TASKS=6;
function drawDrift(){
  const w=dc.clientWidth,h=260;
  dc.width=w*devicePixelRatio; dc.height=h*devicePixelRatio;
  dctx.setTransform(devicePixelRatio,0,0,devicePixelRatio,0,0);
  dctx.clearRect(0,0,w,h);
  const css=getComputedStyle(document.documentElement);
  if(SAMPLES.length<2){
    dctx.fillStyle=css.getPropertyValue('--muted'); dctx.font='12px ui-sans-serif';
    dctx.fillText('Fewer than two advances were analysed.',54,60); return;
  }
  const pad=52, padR=14, padB=26, padT=10;
  const labels=R.frames.taskLabels||[];
  const tasks=labels.slice(0,DRIFT_TASKS);
  const top=Math.max(...SAMPLES.map(s=>s.durationMs))*1.08;
  const sx=(i)=>pad+(SAMPLES.length<2?0:i/(SAMPLES.length-1))*(w-pad-padR);
  const sy=(v)=>h-padB-(v/top)*(h-padB-padT);
  dctx.strokeStyle=css.getPropertyValue('--line'); dctx.lineWidth=1;
  dctx.beginPath(); dctx.moveTo(pad,h-padB); dctx.lineTo(w-padR,h-padB); dctx.stroke();
  dctx.font='11px ui-monospace,monospace'; dctx.fillStyle=css.getPropertyValue('--muted');
  for(let k=0;k<=4;k++){ const v=top*k/4; const y=sy(v);
    dctx.fillText(v.toFixed(0)+' ms',4,y+4);
    dctx.strokeStyle=css.getPropertyValue('--line'); dctx.globalAlpha=.5;
    dctx.beginPath(); dctx.moveTo(pad,y); dctx.lineTo(w-padR,y); dctx.stroke(); dctx.globalAlpha=1; }
  // Highlight the frame the timeline is showing, and the two being compared.
  const marks=[[capture().index,css.getPropertyValue('--accent')],
    [(CAPS[+$('cmpA').value]||{}).index,'#d68c36'],[(CAPS[+$('cmpB').value]||{}).index,'#48aa6e']];
  marks.forEach(([i,colour])=>{ if(i===undefined) return;
    dctx.strokeStyle=colour; dctx.globalAlpha=.55; dctx.beginPath();
    dctx.moveTo(sx(i),padT); dctx.lineTo(sx(i),h-padB); dctx.stroke(); dctx.globalAlpha=1; });
  const line=(pick,colour,width)=>{
    dctx.strokeStyle=colour; dctx.lineWidth=width; dctx.beginPath();
    SAMPLES.forEach((s,i)=>{ const y=sy(pick(s)); i?dctx.lineTo(sx(i),y):dctx.moveTo(sx(i),y); });
    dctx.stroke();
  };
  line(s=>s.durationMs,css.getPropertyValue('--ink'),1.6);
  line(s=>s.busyMs,'#5b9bff',1.2);
  tasks.forEach((label,t)=>line(s=>s.tasks[t]||0,taskColour(label),1));
  dctx.fillStyle=css.getPropertyValue('--muted'); dctx.lineWidth=1;
  dctx.fillText(frameName(0),pad,h-8);
  dctx.fillText(frameName(SAMPLES.length-1),w-padR-64,h-8);
  const first=SAMPLES[0], last=SAMPLES[SAMPLES.length-1];
  // Mark the advances that do not carry the modal amount of work. They are
  // full-length frames doing more or less, not evidence of a bad boundary.
  const tally=new Map();
  SAMPLES.forEach(s=>{const k=s.encoders+'/'+s.passes; tally.set(k,(tally.get(k)||0)+1);});
  const modal=[...tally.entries()].sort((a,b)=>b[1]-a[1])[0];
  SAMPLES.forEach((s,i)=>{
    if(s.encoders+'/'+s.passes===modal[0]) return;
    dctx.fillStyle='#d68c36'; dctx.globalAlpha=.75;
    dctx.fillRect(sx(i)-1,h-padB-4,2.5,4); dctx.globalAlpha=1;
  });
  $('drift-legend').innerHTML=
    '<span><i style="background:'+css.getPropertyValue('--ink')+'"></i>frame wall</span>'
    +'<span><i style="background:#5b9bff"></i>GPU busy</span>'
    +tasks.map(l=>'<span><i style="background:'+taskColour(l)+'"></i>'+esc(primary(l)).slice(0,34)+'</span>').join('')
    +(tally.size>1?'<span><i style="background:#d68c36"></i>advance off the modal shape</span>':'')
    +'<span>'+(tally.size===1
      ? 'every advance carries '+first.encoders+' encoders / '+first.passes+' passes'
      : modal[1]+' of '+SAMPLES.length+' advances carry '+modal[0].replace('/',' encoders / ')
        +' passes; '+(SAMPLES.length-modal[1])+' do more or less work')+'</span>'
    +'<span>'+frameName(0)+' '+fmt(first.durationMs)+' ms → '+frameName(SAMPLES.length-1)+' '
      +fmt(last.durationMs)+' ms ('+(last.durationMs>=first.durationMs?'+':'')
      +fmt(100*(last.durationMs/first.durationMs-1),1)+'%)</span>';
}
dc.addEventListener('mousemove',(e)=>{
  if(SAMPLES.length<2) return;
  const r=dc.getBoundingClientRect(), pad=52, padR=14;
  const i=Math.round(((e.clientX-r.left)-pad)/(dc.clientWidth-pad-padR)*(SAMPLES.length-1));
  const s=SAMPLES[Math.max(0,Math.min(SAMPLES.length-1,i))]; if(!s) return;
  showTip(e,'<b>'+frameName(s.index)+'</b><br>'+fmt(s.durationMs)+' ms wall · '+fmt(s.busyMs)
    +' ms busy · '+fmt(s.gapMs)+' ms idle<br>'+s.encoders+' encoders · '+s.passes+' passes'
    +(s.occupancy!==undefined&&s.occupancy!==null?'<br>occupancy '+pct(s.occupancy,1)
      +' ('+s.counterSamples+' samples)':''));
});
dc.addEventListener('mouseleave',hideTip);
dc.addEventListener('click',(e)=>{
  const r=dc.getBoundingClientRect(), pad=52, padR=14;
  const i=Math.round(((e.clientX-r.left)-pad)/(dc.clientWidth-pad-padR)*(SAMPLES.length-1));
  // Snap to the nearest retained frame: only those can be drawn.
  let best=0,bd=1e18;
  CAPS.forEach((c,k)=>{const d=Math.abs(c.index-i); if(d<bd){bd=d;best=k;}});
  selectFrame(best);
});

// ---------- frame comparison ----------
function renderCompare(){
  const a=CAPS[+$('cmpA').value], b=CAPS[+$('cmpB').value];
  if(!a||!b){ $('cmpnote').textContent='needs two retained frames'; return; }
  const sa=SAMPLES[a.index], sb=SAMPLES[b.index];
  const labels=R.frames.taskLabels||[];
  const rows=labels.map((label,t)=>({label, a:sa.tasks[t]||0, b:sb.tasks[t]||0}))
    .map(r=>({...r, d:r.b-r.a}))
    .filter(r=>r.a>0||r.b>0)
    .sort((x,y)=>Math.abs(y.d)-Math.abs(x.d));
  const worst=Math.max(...rows.map(r=>Math.abs(r.d)),1e-9);
  $('cmp').querySelector('thead').innerHTML='<tr><th>task</th><th>'+frameName(a.index)
    +'</th><th>'+frameName(b.index)+'</th><th>delta</th><th>change</th></tr>';
  $('cmp').querySelector('tbody').innerHTML=rows.map(r=>
    '<tr data-l="'+escapeAttr(r.label)+'"><td><span title="'+escapeAttr(r.label)+'">'
    +esc(primary(r.label))+'</span>'+(extra(r.label)?'<span class="pill">+'+extra(r.label)+' passes</span>':'')
    +'<div><span class="bar" style="width:'+(100*Math.abs(r.d)/worst).toFixed(1)+'%;background:'
    +(r.d>0?'#d64545':'#48aa6e')+'"></span></div></td>'
    +'<td class="num">'+fmt(r.a,3)+'</td><td class="num">'+fmt(r.b,3)+'</td>'
    +'<td class="num">'+(r.d>=0?'+':'')+fmt(r.d,3)+'</td>'
    +'<td class="num">'+(r.a>0?((r.d>=0?'+':'')+fmt(100*r.d/r.a,1)+'%'):'new')+'</td></tr>').join('');
  $('cmp').querySelectorAll('tbody tr').forEach(tr=>{ tr.onclick=()=>selectPass(tr.dataset.l); });
  $('cmpnote').textContent=fmt(sa.durationMs)+' ms → '+fmt(sb.durationMs)+' ms wall · '
    +fmt(sa.busyMs)+' → '+fmt(sb.busyMs)+' ms busy · '
    +(sa.encoders===sb.encoders&&sa.passes===sb.passes
      ? 'same '+sa.encoders+' encoders / '+sa.passes+' passes'
      : 'DIFFERENT shape: '+sa.encoders+'/'+sa.passes+' vs '+sb.encoders+'/'+sb.passes);
  drawDrift();
}

// ---------- frame selector ----------
function selectFrame(next){
  capIndex=Math.max(0,Math.min(CAPS.length-1,next));
  const c=capture();
  TL={frameDuration:c.durationUs, intervals:c.intervals};
  tlSel=null; $('tl-detail').style.display='none';
  $('frameSel').value=String(capIndex);
  const s=SAMPLES[c.index]||{};
  $('framenote').textContent=frameName(c.index)+' · '+fmt(s.durationMs)+' ms wall · '
    +fmt(s.busyMs)+' ms busy · '+s.encoders+' encoders · '+s.passes+' passes'
    +(CAPS.length<SAMPLES.length? ' · '+CAPS.length+' of '+SAMPLES.length+' advances retained in full':'')
    +(isRepresentative()?' · representative':'');
  drawTimeline(); drawCounters(); drawGrid(); drawDrift();
}
{
  const options=CAPS.map((c,k)=>'<option value="'+k+'">'+frameName(c.index)+' — '
    +fmt((SAMPLES[c.index]||{}).durationMs)+' ms</option>').join('');
  ['frameSel','cmpA','cmpB'].forEach(id=>{ $(id).innerHTML=options; });
  $('cmpA').value='0'; $('cmpB').value=String(Math.max(CAPS.length-1,0));
  $('frameSel').onchange=()=>selectFrame(+$('frameSel').value);
  $('frameFirst').onclick=()=>selectFrame(0);
  $('framePrev').onclick=()=>selectFrame(capIndex-1);
  $('frameNext').onclick=()=>selectFrame(capIndex+1);
  $('frameLast').onclick=()=>selectFrame(CAPS.length-1);
  $('frameRep').onclick=()=>selectFrame(R.frames.representative||0);
  $('cmpA').onchange=renderCompare; $('cmpB').onchange=renderCompare;
}

// ---------- integrity ----------
$('integrity').innerHTML =
  '<div class="kv">'
  +'<div><span>anchor pass</span>'+esc(R.frames.anchor)+'</div>'
  +'<div><span>advances analysed</span>'+R.frames.count+'</div>'
  +'<div><span>pass attribution</span>'+pct(1-R.gpu.mergedShare,1)+' single-encoder</div>'
  +'<div><span>hardware model</span>'+(R.counters.totalSlots
      ? R.counters.partitionCount+' partitions \u00d7 '+R.counters.slotsPerPartition+' slots \u00d7 '
        +R.counters.threadsPerSlot+' threads = '+R.counters.totalThreads.toLocaleString()
        +' ('+pct(R.counters.slotConfidence,0)+' of readings fit)'
      : 'not detected')+'</div>'
  +'<div><span>counter coverage</span>'+(R.counters.available?pct(R.counters.exclusiveCoverage,1)+' uncontended':'not captured')+'</div>'
  +'<div><span>untraced baseline</span>'+(R.wall.baselineMsPerAdvance?fmt(R.wall.baselineMsPerAdvance)+' ms/advance':'—')+'</div>'
  +'<div><span>traced wall</span>'+(R.wall.tracedMsPerAdvance?fmt(R.wall.tracedMsPerAdvance)+' ms/advance':'—')+'</div>'
  +'</div>'
  + (R.contention.length? '<div class="warn" style="margin-top:12px"><b>Other processes used the GPU during this capture.</b> '
      +'Per-task GPU times below are filtered to our process, but device-wide counters are only sampled '
      +'in windows where nothing else was on the GPU ('+pct(R.counters.exclusiveCoverage||0,1)+' coverage).<br>'
      + R.contention.slice(0,6).map(c=>esc(c.process)+' — '+fmt(c.gpuMs,0)+' ms across '+c.intervals+' intervals').join('<br>')
      +'</div>' : '<div style="margin-top:12px" class="muted">No other process used the GPU during the measured window.</div>');

addEventListener('resize',()=>{drawTimeline();drawCounters();drawGrid();drawMachine();drawPlacement();drawWaves();drawFlame();drawDrift();});
drawTimeline(); drawCounters(); renderPasses(); drawGrid(); drawMachine(); drawPlacement(); drawWaves(); renderShaders(); drawFlame();
renderCompare(); selectFrame(capIndex);
</script>`;
};
