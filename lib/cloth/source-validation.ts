import type { WetClothSourceProgram } from './source-types';

/** Serialized source programs are untrusted scene data. Unknown shapes fail
 * before solver construction rather than being silently omitted. */
export function validateWetClothSourceProgram(program: WetClothSourceProgram): string[] {
  const errors: string[] = [];
  const vector = (v: unknown): boolean => Array.isArray(v) && v.length === 3 && v.every(Number.isFinite);
  const nonnegative = (v: unknown): boolean => typeof v === 'number' && Number.isFinite(v) && v >= 0;
  if (!program || program.schemaVersion !== 1) return ['Unsupported wet-cloth source program version'];
  if (!program.sourceId || !(program.timeStep_s > 0) || !Number.isFinite(program.timeStep_s) || !vector(program.sourceOffset_m)) errors.push('Invalid wet-cloth source identity, step or origin');
  if(program.variant){const v=program.variant;if(v.id!==program.sourceId||!v.sourceId||v.sourceId===v.id||!['paper-corrected','exploratory'].includes(v.kind)||v.reproductionValidated!==false||!v.label||!v.reason||!Array.isArray(v.evidence)||!v.evidence.length||!v.evidence.every(e=>typeof e==='string'&&e.startsWith('https://'))||!Array.isArray(v.overrides)||!v.overrides.length||!v.overrides.every(o=>typeof o.path==='string'&&Number.isFinite(o.sourceValue)&&Number.isFinite(o.value)))errors.push('Invalid source variant provenance');}
  if (!Array.isArray(program.fields) || !Array.isArray(program.scripts) || !Array.isArray(program.requiredCapabilities)) return [...errors, 'Malformed wet-cloth source program'];
  const indices = new Set<number>();
  for (const f of program.fields) {
    if (!f || !Number.isInteger(f.index) || f.index < 0 || indices.has(f.index)) { errors.push('Invalid source field index'); continue; }
    if (!['sphere', 'box', 'cylinder', 'intersect', 'file'].includes(f.kind)) errors.push(`Unsupported source field kind ${String(f.kind)}`);
    if (!['solid', 'source', 'terminator'].includes(f.usage) || !Number.isInteger(f.group) || f.group < 0) errors.push('Invalid source field usage or group');
    if (typeof f.inside !== 'boolean' || typeof f.sampled !== 'boolean' || typeof f.root !== 'boolean') errors.push('Invalid source field flags');
    if (f.kind === 'intersect') {
      if (!f.children?.length || f.children.some((i: number) => !indices.has(i))) errors.push('Source intersections must reference preceding fields');
    } else {
      if (!vector(f.center_m) || !vector(f.axis) || !Number.isFinite(f.angle_rad) || (f.angle_rad !== 0 && Math.hypot(...f.axis!) === 0)) errors.push('Invalid source field transform');
      if ((f.kind === 'sphere' || f.kind === 'cylinder') && !(f.radius_m! > 0)) errors.push('Source radius must be positive');
      if (f.kind === 'box' && (!vector(f.halfExtents_m) || f.halfExtents_m!.some((v: number) => v < 0))) errors.push('Invalid source box extents');
      if ((f.kind === 'box' || f.kind === 'cylinder') && !nonnegative(f.roundingRadius_m)) errors.push('Invalid source corner radius');
      if (f.kind === 'cylinder' && !nonnegative(f.halfLength_m)) errors.push('Invalid source cylinder length');
      if (f.kind === 'file' && (!f.filename || !(f.scale! > 0))) errors.push('Invalid external source geometry');
      for (const d of f.emission ?? []) if (!nonnegative(d.start_s) || !nonnegative(d.end_s) || d.end_s < d.start_s || !vector(d.velocity_m_s)) errors.push('Invalid source emission duration');
    }
    indices.add(f.index);
  }
  for(const i of program.movingSolidFieldIndices??[])if(!program.fields.some(f=>f.index===i&&f.root&&f.usage==='solid'))errors.push('Moving solid indices must reference solid roots');
  for (const s of program.scripts) {
    if (!s || !['translate', 'rotate'].includes(s.kind) || s.easing !== 'cubic' || !Number.isInteger(s.group) || s.group < 0) { errors.push('Unsupported source script'); continue; }
    if (!nonnegative(s.start_s) || !(s.end_s > s.start_s) || !Number.isFinite(s.end_s) || !nonnegative(s.easeStart_s) || !nonnegative(s.easeEnd_s) || s.easeStart_s + s.easeEnd_s > s.end_s - s.start_s + 1e-12) errors.push('Invalid source script timing');
    if (s.kind === 'translate' && !vector(s.displacement_m)) errors.push('Invalid source translation');
    if (s.kind === 'rotate' && (!vector(s.axis) || !Number.isFinite(s.angle_rad) || Math.hypot(...s.axis!) === 0)) errors.push('Invalid source rotation');
    if (s.origin_m !== undefined && !vector(s.origin_m)) errors.push('Invalid source rotation origin');
  }
  return errors;
}
