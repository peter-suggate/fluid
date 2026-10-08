import {
  UNIFORM_DETAIL_CRITERIA, UNIFORM_STAGE_REASON as R, type UniformDetailCriterion,
} from "../methods/uniform/uniform-stage-grids";

/** Shared semantic catalog. Order is the composition order, never click order. */
export const VISUAL_LAYERS = [
  { id: "phi", requires: ["phi"], label: "Signed distance", color: "#ed7829", opacity: 0.65, mode: 3, description: "Signed distance in cell widths: blue liquid, white interface, orange air. Mixed Uniform keeps it at owner resolution: h vertices in h tiles, 4h corners under 4h owners (interpolated between), as in an all-4h run." },
  { id: "volume", requires: ["volume"], label: "Liquid volume", color: "#4f9ae0", opacity: 0.7, mode: 21, description: "Transported volume over the open capacity of each cell (the cell volume where no capacity is published): dilute residue on a log ramp from purple to green, liquid blue, overfull cells red. Opacity follows fullness with a floor so residue stays visible. Mixed Uniform draws the h and 4h bulk cells the volume lives on, with orange resolution seams." },
  { id: "pressure", requires: ["pressure"], label: "Pressure", color: "#7e9ee5", opacity: 0.65, mode: 5, description: "Signed pressure (blue positive, amber negative; full intensity at 10 kPa) on the exact cells the last solve ran on. Mixed Uniform draws each pressure cell at its own size: h cells (the two-stage h band re-solved over the 4h solve, with the band's own pressure) are tinted green, 4h cells untinted, with resolution seams in lilac. Tiles where the level set crosses zero are outlined in yellow; under dynamic coarsening every such tile should be green, so one held at 4h (a surface the head census missed) is outlined in magenta." },
  { id: "tiles", requires: ["tiles"], label: "Work tiles", color: "#3fae8f", opacity: 0.45, mode: 22, description: "Fine velocity tiles, extension shell, and transport reach where published. Mixed Uniform draws each tile's live h or 4h cells and colours every h tile by why it is h, the first rule the frame head's census fired: crossing, closure (shared vertices at dt = 0), travel within this frame's departure, source, host join, solid or body promotion, bulk importance (no surface, but an importance criterion requires it), or static (authored / host layout). 4h tiles the census traced without banding are tinted faintly; untinted 4h tiles were skipped by the O(1) reach test. Hatched h tiles take the general velocity sampler because a 4h tile lies inside their signed reach box (cross-hatched: reach saturated); unhatched h tiles are certified regular. A crossing tile held at 4h is magenta." },
  { id: "importance", requires: ["tiles"], label: "Detail importance", color: "#ee4a5a", opacity: 0.75, mode: 28, description: "Mixed Uniform under dynamic coarsening: the frame head census's detail importance of every tile holding liquid, h or 4h. Each criterion scores its measure over its threshold, so 1 triggers. All: each tile in its highest-scoring criterion's colour, faint below 1 and solid from 1; tiles the census requires at h for importance are outlined, hatched where only the hold still requires them, cross-hatched where a criterion triggered but the budget dropped the tile. One criterion: a heat map of its score, transparent at 0, its colour at 1 and warming toward white near 4, with triggered tiles outlined." },
  { id: "pages", requires: ["pages"], label: "Domain pages", color: "#b39bea", opacity: 0.65, mode: 27, description: "Page states from the last step: teal = transport active; amber = sharpening only; faint purple = resident without volume work. Absent pages are hidden. Residency can include pressure/interface support; authored pages currently remain allocated." },
  { id: "window", requires: ["window"], label: "Working window", color: "#5fb4e6", opacity: 0.8, mode: 23, description: "Actual dispatched window, seed box, launch slack and clipping. Dense scheduling uses the whole domain." },
  { id: "surface", requires: ["phi"], label: "Liquid surface · φ = 0", color: "#ef9f35", opacity: 0.9, mode: 24, description: "Reconstructed liquid and its zero level-set interface. Mixed Uniform colours the interface by the velocity that advected it: teal where all three components were h velocity (h bulk, or retained and extended projected h velocity), amber where the 4h bulk sampler moved it, blended where only some components were h." },
  // The one layer that is not a slice: mode -1 keeps it out of the plane pass,
  // and the particle overlay draws it over the whole liquid instead.
  { id: "particles", requires: ["particles"], label: "Particles", color: "#8f8cdb", opacity: 1, mode: -1, description: "The method's own particles as shaded spheres through the whole liquid, coloured by speed (full scale at 2 m/s). Under the Simple surface they sit in the water: fogged by the liquid in front of them and hidden once it is deep enough. Narrow-band FLIP draws its velocity samples: the 4h surface band only, none in the Eulerian interior." },
  { id: "grid", requires: ["dimensions"], label: "Grid", color: "#a8c7d8", opacity: 0.7, mode: 0, description: "Represented cell boundaries, independently of field fills. Mixed Uniform draws its live h and 4h bulk owners with orange resolution seams, and hatches the tiles this frame's head relayout changed: orange refined to h, blue coarsened to 4h (a host relayout after the last transport hatches orange)." },
  { id: "velocity", requires: ["velocity"], label: "Velocity", color: "#dce9ee", opacity: 0.9, mode: 26, description: "Cell velocity magnitude and in-plane direction; full scale at 1 m/s. Mixed Uniform draws one arrow per h or 4h bulk owner, the grid momentum is advected on." },
  { id: "release", requires: ["releasedFaces"], label: "Released faces", color: "#f5be52", opacity: 1, mode: 25, description: "Solid faces released by the pressure projection." },
] as const;
export type VisualLayerId = typeof VISUAL_LAYERS[number]["id"];
/** What the importance layer shows: every criterion's winner, or one criterion's score. */
export type ImportanceView = "all" | UniformDetailCriterion;
export interface VisualLayerState {
  enabled: VisualLayerId[];
  visible: boolean;
  opacity: Partial<Record<VisualLayerId, number>>;
  /** The importance layer's view; absent is "all" (the canonical default). */
  importance?: ImportanceView;
}
export function visualLayers(enabled: readonly VisualLayerId[] = ["surface"]): VisualLayerState {
  return { enabled: VISUAL_LAYERS.filter(l => enabled.includes(l.id)).map(l => l.id), visible: true, opacity: {} };
}
export function layerOpacity(state: VisualLayerState, id: VisualLayerId): number {
  return state.opacity[id] ?? VISUAL_LAYERS.find(l => l.id === id)!.opacity;
}
export const importanceView = (state: VisualLayerState): ImportanceView => state.importance ?? "all";
export function setImportanceView(state: VisualLayerState, view: ImportanceView): VisualLayerState {
  const next: VisualLayerState = { ...state, importance: view };
  // "all" is stored as absence, so a state round-trips a link unchanged.
  if (view === "all") delete next.importance;
  return next;
}
/** The layer uniform's selection float: 0 = all, else 1 + the criterion id
 * (UNIFORM_DETAIL_CRITERIA index). */
export function importanceViewCode(state: VisualLayerState): number {
  const view = importanceView(state);
  return view === "all" ? 0 : 1 + UNIFORM_DETAIL_CRITERIA.indexOf(view);
}
/** Whether a layer is sampled on the slice plane, as every layer but the particle spheres is. */
export const isSliceLayer = (id: VisualLayerId): boolean => VISUAL_LAYERS.find(l => l.id === id)!.mode >= 0;
/** The plane pass has something to draw. */
export const sliceLayersShown = (state: VisualLayerState): boolean => state.visible && state.enabled.some(isSliceLayer);
export const particleLayerShown = (state: VisualLayerState): boolean => state.visible && state.enabled.includes("particles");
export function toggleVisualLayer(state: VisualLayerState, id: VisualLayerId): VisualLayerState {
  return { ...state, visible: true, enabled: VISUAL_LAYERS.filter(l => l.id === id ? !state.enabled.includes(id) : state.enabled.includes(l.id)).map(l => l.id) };
}
export function legacyVisualLayers(mode: string, grid = false): VisualLayerState {
  const aliases: Record<string, VisualLayerId[]> = {
    // Uniform defaults omit the lattice; it remains an explicit layer toggle.
    structure: ["surface"], "volume-levelset": ["volume", "surface"],
    "fine-tiles": ["tiles", "surface"], "solve-window": ["window", "surface"],
    speed: ["velocity"], "face-velocity": ["velocity"], release: ["surface", "release"],
    // Surface density and conserved volume are one liquid volume layer.
    density: ["volume"],
  };
  return visualLayers([...((Object.hasOwn(aliases, mode) ? aliases[mode] : undefined) ?? VISUAL_LAYERS.filter(l => l.id === mode).map(l => l.id)), ...(grid ? ["grid" as const] : [])]);
}
export function readVisualLayers(raw: string | null, fallback = visualLayers()): VisualLayerState {
  if (raw === null) return fallback;
  try {
    const value = JSON.parse(raw);
    if (!Array.isArray(value.enabled)) return fallback;
    const state = visualLayers(value.enabled);
    state.visible = value.visible !== false;
    for (const layer of VISUAL_LAYERS) {
      const n = value.opacity?.[layer.id];
      if (typeof n === "number" && Number.isFinite(n)) state.opacity[layer.id] = Math.max(0, Math.min(1, n));
    }
    const criterion = UNIFORM_DETAIL_CRITERIA.find(c => c === value.importance);
    if (criterion) state.importance = criterion;
    return state;
  } catch { return fallback; }
}
export const writeVisualLayers = (state: VisualLayerState): string => JSON.stringify(state);

/** Display-space ramps, shared with WGSL; physical scales don't jump per frame. */
export const LAYER_PALETTE = {
  liquid: [79, 154, 224], empty: [245, 245, 230], positive: [126, 158, 229], negative: [221, 153, 85],
  phiLiquid: [26, 115, 235], phiAir: [237, 120, 41], residue: [70, 50, 126], dilute: [35, 161, 134],
  densityLiquid: [126, 163, 193], excess: [201, 86, 63], fine: [63, 174, 143], shell: [166, 216, 198], transport: [136, 100, 46],
  surfaceBand: [245, 226, 122], overlayValid: [63, 193, 201], overlayFallback: [214, 132, 44],
  // Mixed Uniform band reasons (tiles layer), relayout marks (grid layer) and
  // the fault colour (a surface the layout held at 4h).
  reasonCrossing: [63, 174, 143], reasonClosure: [104, 136, 214], reasonTravel: [232, 170, 60],
  reasonJoin: [152, 110, 212], reasonSource: [72, 186, 232], reasonSolid: [196, 116, 78],
  reasonBulk: [176, 198, 72],
  reasonStatic: [168, 168, 160], reasonTraced: [146, 156, 170], coarsened: [92, 162, 236], fault: [255, 20, 140],
  // Mixed Uniform detail importance (importance layer): one hue per criterion,
  // saturated so they read apart from the muted band reasons above; the hot
  // end of a single criterion's heat map; the outline, held and dropped marks.
  importanceShape: [72, 204, 92], importanceThin: [28, 206, 214], importanceStrain: [246, 214, 48],
  importanceRotation: [206, 98, 240], importanceImpact: [238, 66, 74], importanceApproach: [98, 112, 255],
  importanceHot: [255, 244, 214], importanceOutline: [250, 248, 240], importanceHeld: [198, 202, 210], importanceDropped: [18, 22, 28],
  // Particle spheres by speed: at rest, half scale, full scale.
  particleSlow: [143, 140, 219], particleMid: [226, 222, 246], particleFast: [244, 158, 52],
} as const;

/** The tiles layer's key on mixed Uniform: why each h tile is h (first rule
 * the frame head's census fired; UNIFORM_STAGE_REASON), plus the sampler
 * certificate hatch. Swatches are the WGSL colours (umReasonPaint). */
const rgb = (c: readonly number[]) => `rgb(${c.join(",")})`;
export const MIXED_TILE_LEGEND = [
  { color: rgb(LAYER_PALETTE.reasonCrossing), label: "Crossing", title: "The level set crosses the tile: the tiles the surface needs." },
  { color: rgb(LAYER_PALETTE.reasonClosure), label: "Closure", title: "Touches a crossing owner's face, edge or corner: shared vertices need it at h even at dt = 0." },
  { color: rgb(LAYER_PALETTE.reasonTravel), label: "Travel", title: "The surface can reach it within this frame's departure." },
  { color: rgb(LAYER_PALETTE.reasonSource), label: "Source", title: "This frame's drop or inflow plug can fill it." },
  { color: rgb(LAYER_PALETTE.reasonJoin), label: "Join", title: "Joined by the host: a solid edit or a body pose." },
  { color: rgb(LAYER_PALETTE.reasonSolid), label: "Solid", title: "Liquid-conditional solid or rigid-body promotion." },
  { color: rgb(LAYER_PALETTE.reasonBulk), label: "Bulk importance", title: "No surface in the tile, but an importance criterion (or its hold) requires it at h." },
  { color: rgb(LAYER_PALETTE.reasonStatic), label: "Static", title: "h without a band reason: an authored fine region or the host layout." },
  { color: rgb(LAYER_PALETTE.reasonTraced), label: "Traced 4h", title: "A 4h tile the census traced and did not band (untinted 4h tiles were skipped in O(1))." },
  { color: rgb(LAYER_PALETTE.fault), label: "Crossing at 4h", title: "A crossing tile the layout held at 4h: a coarse-only region, or a fault." },
  { color: "repeating-linear-gradient(135deg,#333 0 2px,transparent 2px 5px)", label: "General sampler", title: "Hatched h tiles take the general velocity sampler: a 4h tile lies inside their signed reach box (cross-hatched: the reach saturated). Unhatched h tiles are certified regular." },
] as const;

/** Each importance criterion's palette entry, short name and measure
 * (UNIFORM_DETAIL_CRITERIA documents them). */
const IMPORTANCE_CRITERIA = {
  shape: { palette: "importanceShape", label: "Shape", title: "The surface error 4h corners would make, over its tolerance." },
  thin: { palette: "importanceThin", label: "Thin", title: "Liquid or air thinner across the surface than the thickness threshold." },
  strain: { palette: "importanceStrain", label: "Strain", title: "dt·‖sym ∇u‖ over its threshold." },
  rotation: { palette: "importanceRotation", label: "Spin", title: "dt·‖curl u‖ over its threshold." },
  impact: { palette: "importanceImpact", label: "Impact", title: "Surface travel into or up a closed wall or solid, over its threshold." },
  approach: { palette: "importanceApproach", label: "Approach", title: "Steps of horizon over the steps until the surface meets a wall, a solid or another surface." },
} as const satisfies Record<UniformDetailCriterion, { palette: keyof typeof LAYER_PALETTE; label: string; title: string }>;
/** The importance layer's selector: all criteria, or one criterion's score. */
export const IMPORTANCE_VIEW_OPTIONS: ReadonlyArray<{ value: ImportanceView; label: string; hint: string }> = [
  { value: "all", label: "All", hint: "Each tile in its highest-scoring criterion's colour." },
  ...UNIFORM_DETAIL_CRITERIA.map(c => ({ value: c, label: IMPORTANCE_CRITERIA[c].label, hint: IMPORTANCE_CRITERIA[c].title })),
];
const outlineSwatch = (c: string) => ["top/100% 2px", "bottom/100% 2px", "left/2px 100%", "right/2px 100%"].map(edge => `linear-gradient(${c},${c}) ${edge} no-repeat`).join(",");
const hatchSwatch = (c: string, angle: number) => `repeating-linear-gradient(${angle}deg,${c} 0 2px,transparent 2px 5px)`;
/** The importance layer's key in its "all" view. Swatches are the WGSL
 * colours (umImportancePaint and the layer's marks). */
export const IMPORTANCE_LEGEND = [
  ...UNIFORM_DETAIL_CRITERIA.map(c => ({ color: rgb(LAYER_PALETTE[IMPORTANCE_CRITERIA[c].palette]), label: IMPORTANCE_CRITERIA[c].label, title: `${IMPORTANCE_CRITERIA[c].title} The tile's highest score: faint below 1, solid once it triggers.` })),
  { color: outlineSwatch(rgb(LAYER_PALETTE.importanceOutline)), label: "Required", title: "The census requires the tile at h for importance: triggered and kept, or held." },
  { color: hatchSwatch(rgb(LAYER_PALETTE.importanceHeld), 135), label: "Held", title: "Required only by the hold: no criterion triggers now." },
  { color: `${hatchSwatch(rgb(LAYER_PALETTE.importanceDropped), 135)},${hatchSwatch(rgb(LAYER_PALETTE.importanceDropped), 45)},${rgb(LAYER_PALETTE.importanceHeld)}`, label: "Dropped", title: "A criterion triggered, but the budget dropped the tile: cross-hatched over its criterion's colour." },
] as const;
/** The key of the view the importance layer is showing: the criteria and
 * marks, or one criterion's score ramp and its trigger outline. */
export function importanceLegend(view: ImportanceView): ReadonlyArray<{ color: string; label: string; title: string; wide?: boolean }> {
  if (view === "all") return IMPORTANCE_LEGEND;
  const { palette, label, title } = IMPORTANCE_CRITERIA[view];
  return [
    // Score 0 to 4 left to right: the criterion's colour sits at 1.
    { color: `linear-gradient(90deg,transparent,${rgb(LAYER_PALETTE[palette])} 25%,${rgb(LAYER_PALETTE.importanceHot)})`, label: `${label} 0 · 1 · 4`, title: `${title} Score = measure / threshold: transparent at 0, this colour at 1, warming toward white near 4.`, wide: true },
    { color: outlineSwatch(rgb(LAYER_PALETTE.importanceOutline)), label: "Triggered", title: "This criterion's score reached 1 in the tile." },
  ];
}
export const LAYER_PRESSURE_SCALE = 10_000;
export const LAYER_SPEED_SCALE = 1;
/** Speed in m/s at which a particle sphere reaches the fast end of its ramp. */
export const LAYER_PARTICLE_SPEED_SCALE = 2;
/** The particle layer's key. The swatch is the overlay's WGSL ramp. */
export const PARTICLE_LEGEND = [
  { color: `linear-gradient(90deg,${rgb(LAYER_PALETTE.particleSlow)},${rgb(LAYER_PALETTE.particleMid)},${rgb(LAYER_PALETTE.particleFast)})`, label: `Speed 0 · ${LAYER_PARTICLE_SPEED_SCALE / 2} · ${LAYER_PARTICLE_SPEED_SCALE} m/s`, title: "Each sphere is one particle, coloured by its own speed.", wide: true },
] as const;
export function scalarLayerPaint(id: VisualLayerId, value: number): { color: readonly number[]; alpha: number } {
  const p = LAYER_PALETTE;
  const mix = (a: readonly number[], b: readonly number[], t: number) => a.map((v, i) => v + (b[i]! - v) * Math.max(0, Math.min(1, t)));
  switch (id) {
    case "phi": return { color: mix(p.empty, value < 0 ? p.phiLiquid : p.phiAir, Math.abs(value) / 4), alpha: 0.8 };
    case "pressure": return { color: value < 0 ? p.negative : p.positive, alpha: Math.min(1, Math.abs(value) / LAYER_PRESSURE_SCALE) };
    case "volume": return { color: value > 1 ? p.excess : value >= 0.5 ? p.densityLiquid : mix(p.residue, p.dilute, (Math.log10(Math.max(value, 1e-6)) + 6) / Math.log10(500000)), alpha: value > 1e-6 ? Math.max(0.35, Math.min(1, value)) : 0 };
    case "tiles": return { color: value & 1 ? p.fine : value & 2 ? p.shell : p.transport, alpha: value & 1 ? 0.66 : value & 2 ? 0.3 : value & 4 ? 0.35 : 0 };
    case "velocity": return { color: p.liquid, alpha: Math.min(1, Math.max(0, value) / LAYER_SPEED_SCALE) * 0.25 };
    default: return { color: p.liquid, alpha: 1 };
  }
}
export const visualLayerPaintWGSL = `
${Object.entries(LAYER_PALETTE).map(([key, rgb]) => `const LP_${key}:vec3f=vec3f(${rgb.map(n => (n / 255).toFixed(8)).join(",")});`).join("\n")}
fn scalarLayerPaint(mode:i32,value:f32)->vec4f {
  if(mode==3){return vec4f(mix(LP_empty,select(LP_phiAir,LP_phiLiquid,value<0.0),clamp(abs(value)/4.0,0.0,1.0)),0.8);}
  if(mode==5){return vec4f(select(LP_positive,LP_negative,value<0.0),min(1.0,abs(value)/${LAYER_PRESSURE_SCALE}.0));}
  if(mode==21){let dilute=mix(LP_residue,LP_dilute,clamp((log2(max(value,1e-6))/log2(10.0)+6.0)/log2(500000.0)*log2(10.0),0.0,1.0));return vec4f(select(select(dilute,LP_densityLiquid,value>=0.5),LP_excess,value>1.0),select(0.0,clamp(value,0.35,1.0),value>1e-6));}
  if(mode==22){let bits=u32(value);return vec4f(select(select(LP_transport,LP_shell,(bits&2u)!=0u),LP_fine,(bits&1u)!=0u),select(select(select(0.0,0.35,(bits&4u)!=0u),0.3,(bits&2u)!=0u),0.66,(bits&1u)!=0u));}
  return vec4f(LP_liquid,min(1.0,max(0.0,value)/${LAYER_SPEED_SCALE}.0)*0.25);
}
// Mixed Uniform band reason (UNIFORM_STAGE_REASON) of a tile, h or 4h.
fn umReasonPaint(reason:u32,h:bool)->vec4f{
  var c=LP_reasonStatic;
  switch(reason){
    case ${R.crossing}u:{c=LP_reasonCrossing;}
    case ${R.closure}u:{c=LP_reasonClosure;}
    case ${R.travel}u:{c=LP_reasonTravel;}
    case ${R.join}u:{c=LP_reasonJoin;}
    case ${R.source}u:{c=LP_reasonSource;}
    case ${R.solid}u:{c=LP_reasonSolid;}
    case ${R.bulk}u:{c=LP_reasonBulk;}
    default:{}
  }
  let band=reason>=${R.crossing}u;
  if(h){return vec4f(c,select(0.45,0.62,band));}
  if(reason==${R.crossing}u){return vec4f(LP_fault,0.85);}
  if(band){return vec4f(c,0.2);}
  return vec4f(LP_reasonTraced,select(0.0,0.16,reason==${R.traced}u));
}
// Mixed Uniform detail importance: a criterion (UNIFORM_DETAIL_CRITERIA
// index) at score = measure / threshold, so 1 triggers.
//   heat   one criterion's score: transparent at 0, its colour at 1, then
//          warming toward white as the score saturates near 4
//   else   the tile's winning criterion: a faint ramp below 1, so near
//          misses show, and a step up once it triggers
fn umImportancePaint(criterion:u32,score:f32,heat:bool)->vec4f{
  var c=LP_importanceHeld;
  switch(criterion){
${UNIFORM_DETAIL_CRITERIA.map((id, i) => `    case ${i}u:{c=LP_${IMPORTANCE_CRITERIA[id].palette};}`).join("\n")}
    default:{}
  }
  let under=clamp(score,0.0,1.0);let over=clamp((score-1.0)/3.0,0.0,1.0);
  if(heat){return vec4f(mix(c,LP_importanceHot,0.8*over),0.62*under+0.2*over);}
  return vec4f(c,select(0.3*under,0.6+0.12*over,score>=1.0));
}
`;
