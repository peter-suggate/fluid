/** CPU-only Pool Party collision edit profile.
 * node --import tsx tools/benchmark-box3d-solid-edits.ts > report.json
 */
import { createBackyardPoolPartyScene } from '../lib/core/backyard-pool-party-scene';
import { SceneInteractionHost } from '../lib/interaction/runtime/host';
import { Box3DWorld } from '../lib/interaction/adapters/box3d/world';
import { sceneWithSolidStroke } from '../lib/core/solid-world';

async function main() {
  const rows: { phase: string; operation: string; ms: number }[] = [];
  let phase = 'startup';
  for (const key of ['validateSolidEdit', 'updateSolids'] as const) {
    const original = Box3DWorld.prototype[key];
    Box3DWorld.prototype[key] = function (...args: Parameters<typeof original>) {
      const start = performance.now();
      try { return original.apply(this, args); }
      finally { rows.push({ phase, operation: key, ms: performance.now() - start }); }
    };
  }
  const base = createBackyardPoolPartyScene();
  base.systems = { ...base.systems, fluid: false };
  const host = new SceneInteractionHost(base);
  await host.ready;
  try {
    for (let i = 0; i < 4; i++) {
      phase = `draw${i}`;
      const next = sceneWithSolidStroke(base, Array.from({ length: i + 1 }, (_, j) => ({
        operation: 'fill' as const, kind: 'ellipsoid', minimum: [155 + j, 40, 55] as const,
        maximumExclusive: [158 + j, 43, 58] as const, materialId: 1,
      })));
      let start = performance.now();
      host.validateSolidEdit(next);
      rows.push({ phase, operation: 'preflight', ms: performance.now() - start });
      start = performance.now();
      await host.replaceScene(next);
      rows.push({ phase, operation: 'replaceScene', ms: performance.now() - start });
    }
    phase = 'undo';
    await host.replaceScene(base);
  } finally { await host.dispose(); }
  console.log(JSON.stringify(rows, null, 2));
}
main().catch(error => { console.error(error); process.exitCode = 1; });
