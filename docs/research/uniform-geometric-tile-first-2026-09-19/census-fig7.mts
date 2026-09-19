/**
 * Live-tile occupancy census for `cm12-figure-7` under `uniformVolumeMethod`.
 *
 * Advances the scene exactly as the stage probe does (same scene document,
 * same resolved method values, dt = 1/30 s, one step per advance) and every
 * Nth step reads back the three fields that decide what a 4h tile map could
 * skip: the (n+1)^3 vertex phi, the conserved volume V, and the MAC velocity.
 *
 * Nothing here is timed: readbacks sit between advances.
 */
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { fileURLToPath, pathToFileURL } from "node:url";
import { resolveMethodValues } from "/Users/petersuggate/code/me/fluid/lib/core/method-contract";
import { getSceneDefinition } from "/Users/petersuggate/code/me/fluid/lib/core/scenes";
import { sceneDocument } from "/Users/petersuggate/code/me/fluid/lib/core/scene-definition";
import { uniformVolumeMethod } from "/Users/petersuggate/code/me/fluid/lib/methods/uniform/uniform-volume-method";
import { usePerformanceInstrumentationStore } from "/Users/petersuggate/code/me/fluid/lib/core/stores/performance-instrumentation-store";
import { requiredFluidDeviceLimits } from "/Users/petersuggate/code/me/fluid/lib/core/webgpu-device-limits";
import {
  acquireWebGPUExclusiveLock,
  releaseWebGPUExclusiveLock,
} from "/Users/petersuggate/code/me/fluid/lib/harness/webgpu-smoke-isolation";
import type { WebGPUUniformReferenceSolver } from "/Users/petersuggate/code/me/fluid/lib/methods/uniform/webgpu-uniform-reference";

const ADVANCES = Number(process.env.FLUID_CENSUS_ADVANCES ?? 70);
const SAMPLE_EVERY = Number(process.env.FLUID_CENSUS_EVERY ?? 5);
const OUT = process.env.FLUID_CENSUS_OUT ?? "./fig7-census.json";
const MAX_K = 8;

/** Copy a 3D texture to the CPU, undoing the 256-byte row padding. */
async function read(device: GPUDevice, texture: GPUTexture): Promise<Float32Array> {
  const components = texture.format === "rgba32float" ? 4 : 1;
  const width = texture.width * components;
  const row = Math.ceil(width * 4 / 256) * 256;
  const buffer = device.createBuffer({
    size: row * texture.height * texture.depthOrArrayLayers,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  try {
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: row, rowsPerImage: texture.height },
      [texture.width, texture.height, texture.depthOrArrayLayers]);
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const values = new Float32Array(buffer.getMappedRange());
    const result = new Float32Array(width * texture.height * texture.depthOrArrayLayers);
    for (let i = 0; i < texture.height * texture.depthOrArrayLayers; i++) {
      result.set(values.subarray(i * row / 4, i * row / 4 + width), i * width);
    }
    return result;
  } finally { buffer.unmap(); buffer.destroy(); }
}

/** One Chebyshev dilation step on the tile grid, done separably. */
function dilateOnce(map: Uint8Array, t: number): Uint8Array {
  const out = new Uint8Array(map.length);
  const tmp = new Uint8Array(map.length);
  // x
  for (let z = 0; z < t; z++) for (let y = 0; y < t; y++) {
    const base = t * (y + t * z);
    for (let x = 0; x < t; x++) {
      let hit = map[base + x]!;
      if (!hit && x > 0) hit = map[base + x - 1]!;
      if (!hit && x + 1 < t) hit = map[base + x + 1]!;
      tmp[base + x] = hit;
    }
  }
  // y
  const tmp2 = new Uint8Array(map.length);
  for (let z = 0; z < t; z++) for (let x = 0; x < t; x++) {
    for (let y = 0; y < t; y++) {
      const i = x + t * (y + t * z);
      let hit = tmp[i]!;
      if (!hit && y > 0) hit = tmp[i - t]!;
      if (!hit && y + 1 < t) hit = tmp[i + t]!;
      tmp2[i] = hit;
    }
  }
  // z
  for (let y = 0; y < t; y++) for (let x = 0; x < t; x++) {
    for (let z = 0; z < t; z++) {
      const i = x + t * (y + t * z);
      let hit = tmp2[i]!;
      if (!hit && z > 0) hit = tmp2[i - t * t]!;
      if (!hit && z + 1 < t) hit = tmp2[i + t * t]!;
      out[i] = hit;
    }
  }
  return out;
}

const count = (map: Uint8Array) => { let n = 0; for (let i = 0; i < map.length; i++) if (map[i]) n++; return n; };

function ladder(seed: Uint8Array, t: number): number[] {
  const counts: number[] = [count(seed)];
  let current = seed;
  for (let k = 1; k <= MAX_K; k++) { current = dilateOnce(current, t); counts.push(count(current)); }
  return counts;
}

await acquireWebGPUExclusiveLock("dawn-census", "fig7 live-tile occupancy census");
let device: GPUDevice | undefined;
try {
  usePerformanceInstrumentationStore.getState().setMode("off");
  const modulePath = process.env.WEBGPU_NODE_MODULE
    ?? fileURLToPath(new URL("../node_modules/webgpu/index.js", import.meta.url));
  const { create, globals } = await import(pathToFileURL(modulePath).href) as {
    create(options: string[]): GPU; globals: Record<string, unknown>;
  };
  Object.assign(globalThis, globals);
  const gpu = create([`backend=${process.env.FLUID_WEBGPU_BACKEND ?? "metal"}`]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
  assert.ok(adapter, "WebGPU did not expose an adapter");
  device = await adapter.requestDevice({ requiredLimits: requiredFluidDeviceLimits(adapter.limits) });
  const validationErrors: string[] = [];
  device.addEventListener("uncapturederror", (event) => {
    validationErrors.push((event as unknown as { error: { message: string } }).error.message);
  });

  const scene = structuredClone(sceneDocument(getSceneDefinition(process.env.FLUID_PROBE_SCENE ?? "cm12-figure-7")));
  const values = resolveMethodValues(uniformVolumeMethod, "balanced", { timeStep: "scene" });
  const solver = await uniformVolumeMethod.createSolverAsync!(
    device, scene, "balanced", values, undefined, () => {}) as unknown as WebGPUUniformReferenceSolver;
  const dt_s = scene.numerics.maxDt_s;
  const info = solver.info;
  const nx = info.nx, ny = info.ny, nz = info.nz;
  const h = scene.container.width_m / nx;
  const hy = scene.container.height_m / ny, hz = scene.container.depth_m / nz;
  const TILE = 4;
  const tx = Math.ceil(nx / TILE), ty = Math.ceil(ny / TILE), tz = Math.ceil(nz / TILE);
  assert.equal(tx, ty); assert.equal(ty, tz);
  const tileTotal = tx * ty * tz;
  const samples: Record<string, unknown>[] = [];

  const sampleNow = async (frame: number) => {
    const phi = await read(device!, solver.vertexPhiTexture!);
    const volume = await read(device!, solver.volumeTexture);
    const velocity = await read(device!, solver.velocityTexture);
    const stats = await solver.readStats() as unknown as Record<string, number>;

    // --- volume seeds, at several admission thresholds. Only `0` (strictly
    // V != 0) can back an exact skip; the others measure how much of the V
    // support is a vanishing semi-Lagrangian smear.
    const EPSILONS = [0, 1e-9, 1e-6, 1e-4, 1e-2];
    const vMaps = EPSILONS.map(() => new Uint8Array(tileTotal));
    const vCells = EPSILONS.map(() => 0);
    const vMass = EPSILONS.map(() => 0);
    let liquidSum = 0, nonZeroCells = 0, cellsAbove1e6 = 0, maxV = 0;
    let negativeCells = 0, negativeMass = 0, minV = 0, fullCells = 0;
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) {
      const rowBase = nx * (y + ny * z);
      const tileRow = (y >> 2) * tx + (z >> 2) * tx * ty;
      for (let x = 0; x < nx; x++) {
        const value = volume[rowBase + x]!;
        if (value !== 0) {
          nonZeroCells++; liquidSum += value;
          if (value > maxV) maxV = value;
          if (value < minV) minV = value;
          if (value < 0) { negativeCells++; negativeMass += value; }
          if (value >= 1 - 1e-6) fullCells++;
          if (value > 1e-6) cellsAbove1e6++;
          const tile = (x >> 2) + tileRow;
          const magnitude = Math.abs(value);
          for (let e = 0; e < EPSILONS.length; e++) {
            if (magnitude > EPSILONS[e]!) {
              vMaps[e]![tile] = 1;
              vCells[e] = vCells[e]! + 1;
              vMass[e] = vMass[e]! + value;
            }
          }
        }
      }
    }
    const vOnly = vMaps[0]!;
    // --- phi seeds; a tile owns the 5^3 vertex block [4t .. 4t+4]
    const band4 = 4 * Math.max(h, hy, hz), band2 = 2 * Math.max(h, hy, hz);
    const phi4 = new Uint8Array(tileTotal), phi2 = new Uint8Array(tileTotal);
    const vx = nx + 1, vy = ny + 1;
    let minPhi = Infinity, bandVertices4 = 0, bandVertices2 = 0;
    // A tile t owns cells [4t..4t+3], hence vertices [4t..4t+4]; a vertex v
    // therefore belongs to every tile t with ceil((v-4)/4) <= t <= floor(v/4).
    const lo = (v: number) => Math.max(0, Math.ceil((v - TILE) / TILE));
    for (let z = 0; z <= nz; z++) {
      const tzLo = lo(z), tzHi = Math.min(tz - 1, z >> 2);
      for (let y = 0; y <= ny; y++) {
        const tyLo = lo(y), tyHi = Math.min(ty - 1, y >> 2);
        const rowBase = vx * (y + vy * z);
        for (let x = 0; x <= nx; x++) {
          const value = phi[rowBase + x]!;
          if (value >= band4) continue;
          if (value < minPhi) minPhi = value;
          bandVertices4++;
          const inner2 = value < band2;
          if (inner2) bandVertices2++;
          const txLo = lo(x), txHi = Math.min(tx - 1, x >> 2);
          for (let c = tzLo; c <= tzHi; c++) for (let b = tyLo; b <= tyHi; b++) {
            const base = tx * (b + ty * c);
            for (let a = txLo; a <= txHi; a++) { phi4[a + base] = 1; if (inner2) phi2[a + base] = 1; }
          }
        }
      }
    }
    const seed4 = new Uint8Array(tileTotal), seed2 = new Uint8Array(tileTotal);
    for (let i = 0; i < tileTotal; i++) {
      seed4[i] = (vOnly[i] || phi4[i]) ? 1 : 0;
      seed2[i] = (vOnly[i] || phi2[i]) ? 1 : 0;
    }

    // --- tile CLASSES for a two-level scheme (coarse 4h grid + fine surface tiles).
    // `cm12-figure-7` has no rigid bodies, no terrain and a plain box container,
    // so `cellOpenFraction` = (1-solid)*(1-terrain) is identically 1: open
    // capacity is 1.0 and a partial cell is 0 < V < 1.
    const CAPACITY = 1;
    const surface = new Uint8Array(tileTotal);
    const lostTiles = new Uint8Array(tileTotal);
    // (a) any of the tile's 5^3 vertices has |phi| < 4h
    const anyBandVertex = new Uint8Array(tileTotal);
    const allVertexBelow = new Uint8Array(tileTotal).fill(1);  // every vertex phi <= -4h
    const allVertexAbove = new Uint8Array(tileTotal).fill(1);  // every vertex phi >=  4h
    for (let z = 0; z <= nz; z++) {
      const tzLo = lo(z), tzHi = Math.min(tz - 1, z >> 2);
      for (let y = 0; y <= ny; y++) {
        const tyLo = lo(y), tyHi = Math.min(ty - 1, y >> 2);
        const rowBase = vx * (y + vy * z);
        for (let x = 0; x <= nx; x++) {
          const value = phi[rowBase + x]!;
          const inBand = Math.abs(value) < band4;
          const below = value <= -band4, above = value >= band4;
          if (inBand || !below || !above) {
            const txLo = lo(x), txHi = Math.min(tx - 1, x >> 2);
            for (let c = tzLo; c <= tzHi; c++) for (let b = tyLo; b <= tyHi; b++) {
              const base = tx * (b + ty * c);
              for (let a = txLo; a <= txHi; a++) {
                const t = a + base;
                if (inBand) anyBandVertex[t] = 1;
                if (!below) allVertexBelow[t] = 0;
                if (!above) allVertexAbove[t] = 0;
              }
            }
          }
        }
      }
    }
    // (b) partial cells, and (c) V != 0 where the cell-centre phi is positive
    const anyVolumeCell = new Uint8Array(tileTotal);
    let partialCells = 0, lostCells = 0, lostMass = 0;
    for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) {
      const rowBase = nx * (y + ny * z);
      const tileRow = (y >> 2) * tx + (z >> 2) * tx * ty;
      for (let x = 0; x < nx; x++) {
        const value = volume[rowBase + x]!;
        if (value === 0) continue;
        const tile = (x >> 2) + tileRow;
        anyVolumeCell[tile] = 1;
        if (value > 0 && value < CAPACITY - 1e-6) { partialCells++; surface[tile] = 1; }
        // cell-centre phi = trilinear at the centre = mean of the 8 corners
        let centre = 0;
        for (let k = 0; k < 8; k++) {
          const vxi = x + (k & 1), vyi = y + ((k >> 1) & 1), vzi = z + ((k >> 2) & 1);
          centre += phi[vxi + vx * (vyi + vy * vzi)]!;
        }
        centre *= 0.125;
        if (centre > 0) { lostCells++; lostMass += value; lostTiles[tile] = 1; surface[tile] = 1; }
      }
    }
    let bulk = 0, farAir = 0, other = 0;
    for (let i = 0; i < tileTotal; i++) {
      if (anyBandVertex[i]) surface[i] = 1;
      if (surface[i]) continue;
      if (allVertexBelow[i]) bulk++;
      else if (allVertexAbove[i] && !anyVolumeCell[i]) farAir++;
      else other++;
    }
    const surfaceLadder = ladder(surface, tx);
    const DENSE_CELLS = nx * ny * nz;
    const twoLevelWork = surfaceLadder.slice(0, 2).map((tiles) => ({
      tiles, coarseCells: tileTotal, fineCells: 64 * tiles,
      totalCells: tileTotal + 64 * tiles,
      fractionOfDense: (tileTotal + 64 * tiles) / DENSE_CELLS,
    }));

    // --- velocity: max |component| over the MAC texture
    let maxComponent = 0;
    for (let i = 0; i < velocity.length; i += 4) {
      for (let c = 0; c < 3; c++) { const a = Math.abs(velocity[i + c]!); if (a > maxComponent) maxComponent = a; }
    }
    const displacementCells = dt_s * maxComponent / h;
    const reachA = 2 * displacementCells + 4;   // trace back + 4h band
    const reachB = displacementCells + 2;       // trace back + 2h band
    const sample = {
      frame,
      time_s: frame * dt_s,
      liquid: { volumeCellSum_readback: liquidSum, volumeCellSum_solver: stats.volumeCellSum,
        nonZeroCells, cellsAboveEpsilon: cellsAbove1e6, maxCellV: maxV, minCellV: minV,
        negativeCells, negativeMass, fullCells,
        fractionOfLattice: nonZeroCells / (nx * ny * nz) },
      phi: { minPhi_m: minPhi, bandVertices4h: bandVertices4, bandVertices2h: bandVertices2,
        vertexTotal: (nx + 1) * (ny + 1) * (nz + 1) },
      velocity: { maxComponent_m_s: maxComponent, maxSpeed_m_s: stats.maxSpeed_m_s,
        displacementCellsPerStep: displacementCells,
        reach_2d_plus_4_cells: reachA, k_for_reach_2d_plus_4: Math.ceil(reachA / TILE),
        reach_d_plus_2_cells: reachB, k_for_reach_d_plus_2: Math.ceil(reachB / TILE) },
      tiles: {
        total: tileTotal,
        vOnly: count(vOnly), phiOnly4h: count(phi4), phiOnly2h: count(phi2),
        seed4h: count(seed4), seed2h: count(seed2),
        ladder_seed4h: ladder(seed4, tx),
        ladder_seed2h: ladder(seed2, tx),
        ladder_vOnly: ladder(vOnly, tx),
        ladder_phiOnly4h: ladder(phi4, tx),
        ladder_phiOnly2h: ladder(phi2, tx),
        classes: {
          surface: surfaceLadder[0], surfaceLadder,
          bulkLiquid: bulk, farAir, other,
          partition: surfaceLadder[0]! + bulk + farAir + other,
          lostLiquidCells: lostCells, lostLiquidTiles: count(lostTiles), lostLiquidMass: lostMass,
          partialCells, twoLevelWork,
        },
        volumeThresholds: EPSILONS.map((eps, e) => ({
          epsilon: eps, cells: vCells[e], massFraction: vMass[e]! / liquidSum,
          tiles: count(vMaps[e]!),
          ladder: ladder(vMaps[e]!, tx),
          ladderWithPhi4h: ladder(Uint8Array.from(vMaps[e]!, (bit, i) => (bit || phi4[i]) ? 1 : 0), tx),
        })),
      },
    };
    samples.push(sample);
    const pctOf = (n: number) => (100 * n / tileTotal).toFixed(2);
    console.log(`frame ${frame}\tliquid ${nonZeroCells} cells (${(100 * nonZeroCells / (nx * ny * nz)).toFixed(2)}%)`
      + `\tdisp ${displacementCells.toFixed(2)} cells/step`
      + `\tseed4h ${sample.tiles.seed4h} (${pctOf(sample.tiles.seed4h)}%)`
      + `\tsurf ${surfaceLadder[0]} (${pctOf(surfaceLadder[0]!)}%) bulk ${bulk} far ${farAir} other ${other}`
      + `\tlost ${lostCells} cells`
      + `\t2lvl ${(100 * twoLevelWork[0]!.fractionOfDense).toFixed(2)}%/${(100 * twoLevelWork[1]!.fractionOfDense).toFixed(2)}%`);
  };

  await sampleNow(0);
  for (let frame = 1; frame <= ADVANCES; frame++) {
    while (!solver.advanceTo(frame * dt_s, [])) await new Promise(setImmediate);
    await device.queue.onSubmittedWorkDone();
    if (frame % SAMPLE_EVERY === 0 || frame === 1) await sampleNow(frame);
  }

  writeFileSync(OUT, JSON.stringify({
    phase: "uniform-geometric-fig7-live-tile-census",
    capturedAt: new Date().toISOString(),
    scene: scene.sceneId, method: uniformVolumeMethod.id,
    lattice: { nx, ny, nz, cells: nx * ny * nz },
    cellSize_m: [h, hy, hz], dt_s, tileGrid: [tx, ty, tz], tileTotal, maxK: MAX_K,
    advances: ADVANCES, sampleEvery: SAMPLE_EVERY,
    methodValues: { extensionFrontSweeps: values.extensionFrontSweeps,
      liquidCapacityBalancing: values.liquidCapacityBalancing,
      sharpeningWorkMap: values.sharpeningWorkMap, redistance: values.redistance,
      velocityTransport: values.velocityTransport, timeStep: values.timeStep },
    samples, validationErrors,
  }, null, 2));
  solver.destroy();
  assert.deepEqual(validationErrors, []);
} finally { device?.destroy(); await releaseWebGPUExclusiveLock(); }
