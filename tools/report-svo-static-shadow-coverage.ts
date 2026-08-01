#!/usr/bin/env node
/**
 * Reach-versus-coverage census for static-shadow VISIBLE certificates.
 *
 * This is the number that decides whether caching direct-light visibility is
 * worth wiring: the share of receiver-page/light channels whose remote static
 * occlusion is already proven clear, plotted against the metres of exact
 * tracing a certified receiver still owes. A certificate only pays if that
 * reach is small next to the scene diagonal printed alongside it — otherwise
 * the "cache" has merely shortened the ray it replaced.
 *
 *   node --import tsx tools/report-svo-static-shadow-coverage.ts
 *   FLUID_SHADOW_CENSUS_SCENES=hose-tank FLUID_SHADOW_CENSUS_REACHES=1,2,4 ...
 */
import { buildSvoSceneLights } from "../lib/svo-light-abi";
import { buildSvoStaticNodeMipPublication } from "../lib/svo-static-node-mips";
import { planSvoStaticShadowField } from "../lib/svo-static-shadow-field";
import { buildSvoStaticShadowVisibleProofs } from "../lib/svo-static-shadow-visible-proofs";
import { planSparseSceneDomain } from "../lib/sparse-scene-domain";
import { getScenePreset, scenePresets } from "../lib/scenes";
import { buildEnvironmentProxyCatalog, environmentProxyPrimitives } from "../lib/voxel-environments";

const requested = (process.env.FLUID_SHADOW_CENSUS_SCENES ?? "hose-tank,garden-hose,garden-svo-lighting")
  .split(",").map((entry) => entry.trim()).filter(Boolean);
const reaches = (process.env.FLUID_SHADOW_CENSUS_REACHES ?? "1,2,3,4,6")
  .split(",").map((entry) => Number(entry.trim())).filter((value) => Number.isFinite(value) && value >= 1);
const maximumLights = Number(process.env.FLUID_SHADOW_CENSUS_LIGHTS ?? "8");
// Production caps the directory at min(8192, maxTextureDimension2D); an
// uncapped run is what the certificate producer actually needs, so both are
// reported and the gap is called out.
const capacity = Number(process.env.FLUID_SHADOW_CENSUS_CAPACITY ?? "1000000");
const brickSize = 8;

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

for (const sceneId of requested) {
  if (!scenePresets.some(({ id }) => id === sceneId)) {
    console.log(`\n${sceneId}: no such preset`);
    continue;
  }
  const scene = getScenePreset(sceneId).create();
  const primitives = environmentProxyPrimitives(
    buildEnvironmentProxyCatalog(scene, scene.environment ?? "default"), true);
  const cell = scene.voxelDomain.finestCellSize_m;
  const dimensions = [
    Math.max(1, Math.round(scene.container.width_m / cell)),
    Math.max(1, Math.round(scene.container.height_m / cell)),
    Math.max(1, Math.round(scene.container.depth_m / cell)),
  ] as const;
  const domain = planSparseSceneDomain(
    scene, dimensions, brickSize,
    primitives.map((primitive) => ({ min: primitive.aabb_m.min, max: primitive.aabb_m.max })),
    { conservativePaddingCells: 1, worldBounds_m: scene.voxelDomain.bounds_m },
  );
  const mips = buildSvoStaticNodeMipPublication(scene, domain, primitives, {
    generation: 1, capacity, samplesPerAxis: 2,
  });
  const lights = buildSvoSceneLights(scene, { revision: 1, maximumRecords: maximumLights });
  const plan = planSvoStaticShadowField(mips.plan, lights);
  const sceneDiagonal = Math.hypot(scene.container.width_m, scene.container.height_m, scene.container.depth_m);

  console.log(`\n${sceneId} — ${mips.plan.requestedPageCount} pages requested, ${lights.records.length} lights,`
    + ` cell ${cell} m, scene diagonal ${sceneDiagonal.toFixed(2)} m`);
  if (mips.plan.requestedPageCount > 8_192) {
    console.log(`  WARNING: production caps the node-mip directory at 8192 pages;`
      + ` this scene needs ${mips.plan.requestedPageCount} and silently drops the remainder.`);
  }
  for (const diagonals of reaches) {
    const publication = buildSvoStaticShadowVisibleProofs(mips, lights.records, plan, {
      localTraceReachPageDiagonals: diagonals,
    });
    if (publication.failClosedReason) {
      console.log(`  reach ${diagonals}x: FAIL CLOSED (${publication.failClosedReason})`);
      continue;
    }
    const share = publication.localTraceReach_m / sceneDiagonal;
    const perLight = publication.coverage
      .map((light) => `${light.kind.slice(0, 4)}${light.lightId}=${percent(light.visibleFraction)}`).join(" ");
    console.log(`  reach ${diagonals}x = ${publication.localTraceReach_m.toFixed(2)} m`
      + ` (${percent(share)} of scene diagonal): certified ${percent(publication.visibleFraction)}   ${perLight}`);
  }
}
