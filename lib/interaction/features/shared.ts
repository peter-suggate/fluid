import type { BehaviorPlugin, ConfigSchema, EditorMetadata, EntityRecord, JsonObject, PreviewPrimitive } from '../contracts';
import type { PhysicsBehaviorFactory } from '../runtime/physics';
export const metadata = (label: string, group: string, order: number): EditorMetadata => ({ label, group, order, hint: label, icon: 'M4 4h16v16H4z', controls: [] });
export function numbers(defaults: Record<string, number>, bounds: Record<string, readonly [number, number]> = {}): ConfigSchema {
  return { version: 1, defaults, validate: value => {
    const errors = Object.keys(value).filter(key => !(key in defaults)).map(key => `Unknown setting ${key}`);
    for (const key of Object.keys(defaults)) {
      const v = value[key], [low, high] = bounds[key] ?? [0.001, 10000];
      if (typeof v !== 'number' || !Number.isFinite(v) || v < low || v > high) errors.push(`${key} must be in [${low}, ${high}]`);
    }
    return errors;
  } };
}
export const emptySchema = numbers({});
export function controls(schema: ConfigSchema, impacts: Readonly<Record<string, 'live'|'rebuild'|'reset'>> = {}): EditorMetadata['controls'] {
  return Object.keys(schema.defaults).map(setting => ({ id: setting, setting, label: setting, kind: 'number', update: impacts[setting]??'reset' }));
}
export const number = (value: JsonObject, key: string) => value[key] as number;
export const box = (value: JsonObject): PreviewPrimitive => ({ shape: 'box', dimensions_m: { x: number(value, 'width'), y: number(value, 'height'), z: number(value, 'depth') } });
export const behavior = (id: string, label: string, provides: readonly string[] = []): BehaviorPlugin => ({ id, schema: emptySchema, editor: metadata(label, 'Behaviors', 0), provides, phases: [] });
export const inertRuntime: PhysicsBehaviorFactory['create'] = () => ({ tick() {}, dispose() {} });
export const instance = (type: string) => ({ id: 'main', type, version: 1, config: {} });
export const scalar = (entity: EntityRecord, key: string) => number(entity.config, key);
export const activePort = { id: 'active', representation: 'boolean', lifetime: 'frame' as const };
