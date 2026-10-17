import type { UpdateImpact } from '../framework/composition';
import type { InteractionControl, InteractionDocument, JsonObject } from './contracts';
import type { Vec3 } from '../core/model';

/** Copy authored compositions, never recipes or runtime state. Behavior IDs are
 * local to each entity; only document-wide entity/connection IDs need remapping. */
export function duplicateEntities(document: InteractionDocument, ids: readonly string[], offset: Vec3, external = true): InteractionDocument {
  if (!ids.length || new Set(ids).size !== ids.length || ids.some(id => !document.entities.some(e => e.id === id))) throw new Error('Select existing entities to duplicate');
  if (!Object.values(offset).every(Number.isFinite)) throw new Error('Duplicate offset must be finite');
  const fresh = (base: string, used: Set<string>) => { let n=1,id=`${base}-copy`;while(used.has(id))id=`${base}-copy-${++n}`;used.add(id);return id; };
  const used = new Set(document.entities.map(e=>e.id));
  const mapping = new Map(ids.map(id=>[id,fresh(id,used)]));
  const entities = document.entities.filter(e=>mapping.has(e.id)).map(e=>({ ...structuredClone(e), id:mapping.get(e.id)!,name:`${e.name} copy`,
    transform:{...structuredClone(e.transform),position_m:{x:e.transform.position_m.x+offset.x,y:e.transform.position_m.y+offset.y,z:e.transform.position_m.z+offset.z}} }));
  const connectionIds = new Set(document.connections.map(c=>c.id));
  const connections = document.connections.filter(c=>external ? mapping.has(c.source.entity)||mapping.has(c.target.entity) : mapping.has(c.source.entity)&&mapping.has(c.target.entity)).map(c=>({
    id:fresh(c.id,connectionIds),source:{...c.source,entity:mapping.get(c.source.entity)??c.source.entity},target:{...c.target,entity:mapping.get(c.target.entity)??c.target.entity},
  }));
  return {...document,entities:[...document.entities,...entities],connections:[...document.connections,...connections]};
}

/** Undeclared fields are conservatively reset; the strongest changed field wins. */
export function configurationImpact(before: JsonObject, after: JsonObject, controls: readonly InteractionControl[]): UpdateImpact {
  let impact: UpdateImpact = 'live';
  for (const key of new Set([...Object.keys(before),...Object.keys(after)])) {
    if (JSON.stringify(before[key])===JSON.stringify(after[key])) continue;
    const effects=controls.filter(c=>c.setting===key).map(c=>c.update??'reset');
    if (!effects.length || effects.includes('reset')) return 'reset';
    if (effects.includes('rebuild')) impact='rebuild';
  }
  return impact;
}
