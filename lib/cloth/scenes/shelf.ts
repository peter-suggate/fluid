import { defineScene, type SceneDefinition } from '../../core/scene-definition';
import type { CameraState, SceneDescription } from '../../core/model';
import { cachedClothTopology } from '../source-assets';
import { cachedSourceMesh } from '../source-mesh';
import { sourceCameraForScene } from '../source-camera';
import type { WetClothSourceDefinition } from '../source-types';
import { buildWetClothSourceScene, prepareWetClothSource } from './definitions';
import { wetClothSource, wetClothValidationSources } from './sources';

/** The product shelf for Fei et al. 2018 (libWetCloth) inputs. */
export const WETCLOTH_SHELF = 'Wet cloth — Fei et al. 2018';

/**
 * Shelf order: the small unit fixtures a reader can watch and edit first, the
 * figure pairs next, then the large paper scenes. Every released input that
 * carries cloth or yarn is offered; liquid-only controls (simple_fluid,
 * surface-tension tests) stay validation fixtures because they have no fabric.
 */
const ORDER = [
  'simple-cloth', 'simple-plastic', 'simple-yarn', 'simple-yarn-loop', 'capillary-suction',
  'dam-break-nonlinear-drag-small', 'buoyancy-test-with', 'buoyancy-test-without', 'splash-single-yarn',
  'dam-break-linear-drag', 'dam-break-nonlinear-drag', 'impermeable-small', 'splash-cloth-small',
  'splash-yarn-small', 'yarn-small', 'ring-test', 'ring-test-nonuniform',
  'pore-test-mid', 'pore-test-low', 'pore-test-high', 'pore-test-ft-low', 'pore-test-ft-high',
  'pore-test-fd-low', 'pore-test-fd-high', 'pore-test-zero', 'impermeable-medium', 'splash-cloth-medium',
  'splash-plastic-medium', 'splash-yarn-medium', 'fuzzy-cloth', 'tighten-towel', 'splash-cloth-large',
].map(name => `wetcloth-source-${name}`);

const VARIANTS = ['wetcloth-paper-figure10b-25um', 'wetcloth-paper-figure10j-90deg', 'wetcloth-exploratory-ring-material-packing', 'wetcloth-source-capillary-suction-upstream-dx'];

const titleCase = (name: string) => name.replace(/\b\w/g, c => c.toUpperCase());
const sourceWet = (source: WetClothSourceDefinition) => source.fields.some(f => f.usage === 'source');

export function wetClothSourcePrepared(id: string): boolean {
  const source = wetClothSource(id);
  if (!cachedClothTopology(source.asset)) return false;
  for (const field of source.fields) {
    if (field.kind !== 'file') continue;
    try { cachedSourceMesh(field.filename!); } catch { return false; }
  }
  return true;
}

/** The released camera when the input has one; otherwise a framing of the
 * built envelope. Neither is the paper's Houdini render camera. */
export function wetClothShelfCamera(scene: SceneDescription): Partial<CameraState> | undefined {
  if (!scene.wetClothSource) return undefined;
  const authored = sourceCameraForScene(scene);
  if (authored) return authored.camera;
  const {width_m, height_m, depth_m} = scene.container;
  return {target_m: {x: 0, y: height_m / 2, z: 0}, distance_m: Math.max(width_m, height_m, depth_m) * 1.6, roll_rad: 0};
}

function blurbFor(source: WetClothSourceDefinition): string {
  const {vertices, triangles, rods} = source.counts;
  const parts = [`${vertices.toLocaleString('en-US')} vertices`];
  if (triangles) parts.push(`${triangles.toLocaleString('en-US')} triangles`);
  if (rods) parts.push(`${rods.toLocaleString('en-US')} ${rods === 1 ? 'yarn' : 'yarns'}`);
  const kind = sourceWet(source) ? 'Water and fabric' : 'Dry fabric, no water';
  const variant = source.variant ? ` ${source.variant.reason}` : '';
  return `${kind}: ${parts.join(', ')}; ${source.duration_s} s at dt ${source.dt_s} s, h ${(source.cellSize_m * 1000).toPrecision(3)} mm, from the released ${source.family.replace(/_/g, ' ')} input.${variant}`;
}

function statusFor(source: WetClothSourceDefinition): string {
  const imported = source.variant
    ? `${source.variant.kind === 'paper-corrected' ? 'Paper-corrected' : 'Exploratory'} input, not reproduced`
    : 'Imported input, not reproduced';
  return sourceWet(source)
    ? `${imported}. Liquid–fabric coupling is still being completed; a step that cannot run reports its error.`
    : `${imported}. Dry mechanics run; the trajectory is not yet compared with the paper.`;
}

/** The released input, and any authored dx and dt that replace its own (the source's discretization field). */
export function wetClothShelfAttribution(source: WetClothSourceDefinition): string {
  const d = source.discretization, cm = (m: number) => +(m * 100).toPrecision(6), ms = (s: number) => +(s * 1000).toPrecision(6);
  const deviation = d ? ` · dx ${cm(d.cellSize_m)} cm, dt ${ms(d.dt_s)} ms (released ${cm(d.sourceCellSize_m)} cm, ${ms(d.sourceDt_s)} ms)` : '';
  return `libWetCloth ${source.source.commit.slice(0, 7)} · ${source.source.path}${deviation}`;
}

function shelfDefinition(source: WetClothSourceDefinition): SceneDefinition {
  return defineScene({
    id: source.id,
    name: source.variant ? source.variant.label : titleCase(source.name),
    blurb: blurbFor(source),
    audience: 'study',
    shelf: WETCLOTH_SHELF,
    environment: 'default',
    // A dry fixture has no fluid, which presentation must never hide.
    presentationMode: 'full-scene',
    containerShell: 'authored',
    methodProfile: {methodId: 'uniform-narrow-band-flip', quality: 'high',
      overrides: {timeStep: 'scene', ...(source.fields.some(f => f.usage === 'terminator') ? {bulkTier: 'off'} : {})}},
    build: () => buildWetClothSourceScene(source.id),
    prepare: () => prepareWetClothSource(source.id),
    prepared: () => wetClothSourcePrepared(source.id),
    cameraForDocument: wetClothShelfCamera,
    attribution: {label: wetClothShelfAttribution(source), url: source.source.url},
    status: statusFor(source),
  });
}

const byId = new Map(wetClothValidationSources.map(source => [source.id, source]));
for (const id of [...ORDER, ...VARIANTS]) if (!byId.has(id)) throw new Error(`Wet-cloth shelf names unknown source ${id}`);
const offered = new Set([...ORDER, ...VARIANTS]);
const missing = wetClothValidationSources.filter(s => s.counts.vertices > 0 && !offered.has(s.id));
if (missing.length) throw new Error(`Wet-cloth shelf omits fabric sources: ${missing.map(s => s.id).join(', ')}`);

export const wetClothShelfDefinitions: readonly SceneDefinition[] =
  [...ORDER, ...VARIANTS].map(id => shelfDefinition(byId.get(id)!));
