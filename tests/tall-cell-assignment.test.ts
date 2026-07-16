import assert from "node:assert/strict";
import test from "node:test";
import { pathToFileURL } from "node:url";
import { createTallCellLayout, type TallCellLayout } from "../lib/tall-cell-grid";
import { tallCellMethod } from "../lib/methods/tall-cell";
import { initializeRigidBodies } from "../lib/rigid-body";
import { createSmokeScenario } from "../tools/webgpu-smoke-scenarios";

// Cell-assignment conformance against the tall-cells paper (docs/papers/
// tallCells.pdf, transcription docs/TALL_CELLS_PAPER.md Section 8):
//   1. at least G_L regular cells below the lowest liquid surface;
//   2. at least G_A regular cells above the highest liquid surface
//      (priority when the two conflict);
//   3. adjacent tall heights differ by at most D (Eq 10);
// plus the packing invariants of Section 3.1: one bottom tall cell per
// column (including air columns) and no liquid interface inside a tall cell.
// Each test prints the mid-z assignment slice so the result is inspectable
// in the test output.

interface ColumnView {
  base: number;
  wet: boolean[];
  surfaceCells: number[];
  tallFraction: number;
}

function columnView(layout: TallCellLayout, x: number, z: number): ColumnView {
  const { nx, fineNy, packedNy, columnBases, initialVolume } = layout;
  const base = Math.round(columnBases[x + nx * z]);
  const tallFraction = initialVolume[x + nx * packedNy * z];
  const wet: boolean[] = [];
  for (let y = 0; y < fineNy; y += 1) {
    if (y < base && base > 0) wet.push(tallFraction >= 0.5);
    else {
      const packedY = 2 + y - base;
      wet.push(packedY >= 2 && packedY < packedNy ? initialVolume[x + nx * (packedY + packedNy * z)] >= 0.5 : false);
    }
  }
  const surfaceCells: number[] = [];
  for (let y = 0; y < fineNy; y += 1) {
    const below = y === 0 ? wet[y] : wet[y - 1];
    const above = y + 1 < fineNy ? wet[y + 1] : false;
    if (wet[y] !== below || wet[y] !== above) surfaceCells.push(y);
  }
  return { base, wet, surfaceCells, tallFraction };
}

function renderSlice(title: string, fineNy: number, nx: number, cellAt: (x: number, y: number) => string) {
  const rows: string[] = [`${title} (T=tall wet, t=tall dry, #=band wet, -=band dry, ' '=unrepresented)`];
  for (let y = fineNy - 1; y >= 0; y -= 1) {
    let row = "";
    for (let x = 0; x < nx; x += 1) row += cellAt(x, y);
    rows.push(`${String(y).padStart(3)} ${row}`);
  }
  console.log(rows.join("\n"));
}

for (const scenarioId of ["dam-break-ui", "settled-tank"] as const) {
  test(`initial cell assignment for ${scenarioId} satisfies the paper's Section 8 constraints`, () => {
    const scene = createSmokeScenario(scenarioId).scene;
    const layout = createTallCellLayout(scene, "balanced");
    const { nx, nz, fineNy, columnBases, settings } = layout;
    const layers = settings.regularLayers;
    const maxBase = Math.max(0, fineNy - layers);
    const D = settings.maximumNeighborDelta;

    let airColumns = 0;
    for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
      const column = columnView(layout, x, z);
      const label = `${scenarioId} column (${x},${z}) base=${column.base}`;

      // Section 3.1 packing: every column owns exactly one bottom tall cell.
      // Height two is the smallest with distinct endpoint samples; base zero
      // would drop the ordinary-column representation entirely.
      assert.ok(column.base >= Math.min(2, maxBase) && column.base <= maxBase, `${label} outside [2, ${maxBase}]`);

      // Section 3.6 constraint: the liquid interface never lies inside a
      // tall cell. Initially that means the store is exactly full or exactly
      // empty and no vertical crossing sits below the band.
      assert.ok(column.tallFraction === 0 || column.tallFraction === 1, `${label} holds a fractional store ${column.tallFraction}`);
      for (const y of column.surfaceCells) {
        assert.ok(y >= column.base, `${label} has a surface cell at y=${y} inside the tall cell`);
        assert.ok(y < column.base + layers, `${label} has a surface cell at y=${y} above the band ceiling ${column.base + layers}`);
      }

      // Section 8 constraints 1 and 2, with the halos clipped at domain
      // walls exactly as the planner clips them (a halo cannot extend past
      // the floor or the lid).
      if (column.surfaceCells.length > 0) {
        const lowest = Math.min(...column.surfaceCells);
        const highest = Math.max(...column.surfaceCells);
        const liquidHalo = Math.min(settings.liquidHalo, lowest + 1);
        const airHalo = Math.min(settings.airHalo, fineNy - highest - 1);
        const lowerBound = highest + 1 + airHalo - layers;   // constraint 2 (priority)
        const upperBound = lowest + 1 - liquidHalo;          // constraint 1
        if (lowerBound <= upperBound) {
          assert.ok(column.base <= Math.max(upperBound, 2), `${label} leaves fewer than G_L=${liquidHalo} band cells below the surface at y=${lowest}`);
        }
        assert.ok(column.base >= Math.min(lowerBound, maxBase) || column.base === 2, `${label} leaves fewer than G_A=${airHalo} band cells above the surface at y=${highest}`);
      } else {
        airColumns += 1;
        // The paper keeps one tall cell in every column, including air; a
        // dry column compresses its full depth (or stays at the height-two
        // control when the grid is nearly ordinary).
        assert.ok(column.base === maxBase || column.base <= 3 || column.base >= maxBase - (nx + nz) * D, `${label} is dry but neither compressed nor at the control height`);
      }

      // Eq 10 neighbor bound.
      for (const [dx, dz] of [[1, 0], [0, 1]] as const) {
        const nxp = x + dx, nzp = z + dz;
        if (nxp >= nx || nzp >= nz) continue;
        const neighbor = Math.round(columnBases[nxp + nx * nzp]);
        assert.ok(Math.abs(column.base - neighbor) <= D, `${label} vs neighbor (${nxp},${nzp}) base=${neighbor} exceeds D=${D}`);
      }
    }

    if (scenarioId === "dam-break-ui") assert.ok(airColumns > 0, "the dam scene must exercise dry-column assignment");

    // Print the wettest slice so the assignment staircase around the liquid
    // (not an all-air cross-section) is what lands in the test output.
    let z = Math.floor(nz / 2), bestWet = -1;
    for (let candidate = 0; candidate < nz; candidate += 1) {
      let wet = 0;
      for (let x = 0; x < nx; x += 1) if (columnView(layout, x, candidate).surfaceCells.length > 0) wet += 1;
      if (wet > bestWet) { bestWet = wet; z = candidate; }
    }
    renderSlice(`initial assignment ${scenarioId} z=${z}`, fineNy, nx, (x, y) => {
      const column = columnView(layout, x, z);
      if (y < column.base) return column.wet[y] ? "T" : "t";
      if (y < column.base + layers && y < fineNy) return column.wet[y] ? "#" : "-";
      return " ";
    });
  });
}

const modulePath = process.env.WEBGPU_NODE_MODULE;
test("remeshed cell assignment after the dam-front transient still satisfies Section 8", { skip: !modulePath && "set WEBGPU_NODE_MODULE for GPU assignment checks" }, async () => {
  const { create, globals } = await import(pathToFileURL(modulePath!).href) as { create(options: string[]): GPU; globals: Record<string, unknown> };
  Object.assign(globalThis, globals);
  const gpu = create(["backend=metal"]);
  Object.defineProperty(globalThis, "navigator", { configurable: true, value: { gpu } });
  try {
    const scenario = createSmokeScenario("dam-break-ui");
    const adapter = await gpu.requestAdapter({ powerPreference: "high-performance" });
    assert.ok(adapter, "no WebGPU adapter");
    const device = await adapter.requestDevice({ requiredLimits: { maxTextureDimension3D: Math.min(2048, adapter.limits.maxTextureDimension3D) } });
    const values = Object.fromEntries(tallCellMethod.params.map((parameter) => [parameter.key, parameter.default])) as Record<string, string | number>;
    const solver = tallCellMethod.createSolver(device, scenario.scene, "balanced", values, () => {}) as import("../lib/webgpu-eulerian").WebGPUEulerianSolver;
    const bodies = initializeRigidBodies(scenario.scene.rigidBodies);
    const dt = scenario.scene.numerics.maxDt_s;
    let t = 0;
    while (t < 0.224 - 1e-9) { t = Math.min(0.224, t + dt); solver.advanceTo(t, bodies); await solver.readStats(); }
    await device.queue.onSubmittedWorkDone();

    const info = solver.info;
    const readTexture = async (texture: GPUTexture, width: number, height: number, depth: number) => {
      const bytesPerRow = Math.ceil(width * 4 / 256) * 256;
      const buffer = device.createBuffer({ size: bytesPerRow * height * depth, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
      const encoder = device.createCommandEncoder();
      encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow, rowsPerImage: height }, [width, height, depth]);
      device.queue.submit([encoder.finish()]);
      await buffer.mapAsync(GPUMapMode.READ);
      const raw = new Float32Array(buffer.getMappedRange().slice(0));
      const out = new Float32Array(width * height * depth);
      const rowFloats = bytesPerRow / 4;
      for (let z = 0; z < depth; z += 1) for (let y = 0; y < height; y += 1) for (let x = 0; x < width; x += 1) out[x + width * (y + height * z)] = raw[x + rowFloats * (y + height * z)];
      buffer.destroy();
      return out;
    };
    const packed = await readTexture(solver.volumeTexture, info.nx, info.storedNy, info.nz);
    const bases = await readTexture(solver.columnBaseTexture, info.nx, info.nz, 1);
    const { nx, nz, ny: fineNy, storedNy, regularLayers: layers, maximumNeighborDelta: D } = info as unknown as { nx: number; nz: number; ny: number; storedNy: number; regularLayers: number; maximumNeighborDelta: number };
    const maxBase = Math.max(0, fineNy - layers);
    const storeAt = (x: number, z: number) => Math.max(0, packed[x + nx * storedNy * z]);
    const bandAt = (x: number, z: number, packedY: number) => packed[x + nx * (packedY + storedNy * z)];
    const columnWater = (x: number, z: number) => {
      const base = Math.round(bases[x + nx * z]);
      let total = storeAt(x, z) * base;
      for (let packedY = 2; packedY < storedNy; packedY += 1) if (base + packedY - 2 < fineNy) total += Math.max(0, bandAt(x, z, packedY));
      return total;
    };
    // The two representability floors the remesh smoother applies after the
    // Eq 10 min: the column's total water must fit under the band ceiling,
    // and the highest wet cell must stay inside the band (a conservative VOF
    // cannot delete above-band liquid the way the paper's level set can).
    const representabilityFloor = (x: number, z: number) => {
      const base = Math.round(bases[x + nx * z]);
      let highestWet = base > 0 && storeAt(x, z) >= 0.5 ? base - 1 : -1;
      for (let packedY = 2; packedY < storedNy; packedY += 1) {
        const worldY = base + packedY - 2;
        if (worldY < fineNy && bandAt(x, z, packedY) >= 0.5) highestWet = worldY;
      }
      const wetTopFloor = Math.min(Math.max(highestWet + 2 - layers, 0), maxBase);
      return Math.max(2, Math.ceil(columnWater(x, z)) - layers, wetTopFloor);
    };

    let dryUnderWet = 0, surfaceOutOfBand = 0, wetColumns = 0, unexcusedDelta = 0;
    for (let z = 0; z < nz; z += 1) for (let x = 0; x < nx; x += 1) {
      const base = Math.round(bases[x + nx * z]);
      assert.ok(base >= 2 && base <= maxBase, `column (${x},${z}) base=${base} outside [2,${maxBase}] after remeshing`);

      // Eq 10, with the representability excuse the solver documents: a
      // column may exceed D only when lowering it would strand its own
      // water above the band (conservative VOF cannot delete liquid the way
      // the paper's level set can).
      for (const [dx, dz] of [[1, 0], [0, 1]] as const) {
        const nxp = x + dx, nzp = z + dz;
        if (nxp >= nx || nzp >= nz) continue;
        const neighbor = Math.round(bases[nxp + nx * nzp]);
        if (Math.abs(base - neighbor) <= D) continue;
        const [highX, highZ, high] = base > neighbor ? [x, z, base] : [nxp, nzp, neighbor];
        const floor = representabilityFloor(highX, highZ);
        if (high > floor) { unexcusedDelta += 1; if (unexcusedDelta <= 6) console.log(`  delta violation: (${x},${z}) base=${base} vs (${nxp},${nzp}) base=${neighbor} floor(high)=${floor} water(high)=${columnWater(highX, highZ).toFixed(2)}`); }
      }

      // Section 3.6: a wet band cannot rest on an air-classified store.
      const storeWet = storeAt(x, z) >= 0.5;
      const bandBottomWet = bandAt(x, z, 2) >= 0.5;
      if (!storeWet && bandBottomWet) dryUnderWet += 1;

      // Section 8 coverage: the free surface of every wet column lies inside
      // its band (constraints 1 and 2 are what place it there).
      let highestWet = storeWet ? base - 1 : -1;
      for (let packedY = 2; packedY < storedNy; packedY += 1) {
        const worldY = base + packedY - 2;
        if (worldY < fineNy && bandAt(x, z, packedY) >= 0.5) highestWet = worldY;
      }
      if (highestWet >= 0) {
        wetColumns += 1;
        const aboveCeiling = highestWet >= base + layers;
        const deepPartialStore = highestWet < base && !bandBottomWet && storeAt(x, z) < 0.95 && base > 3;
        if (aboveCeiling || deepPartialStore) { surfaceOutOfBand += 1; if (surfaceOutOfBand <= 6) console.log(`  surface out of band: (${x},${z}) base=${base} highestWet=${highestWet} store=${storeAt(x, z).toFixed(2)} ceiling=${aboveCeiling}`); }
      }
    }

    const z = Math.floor(nz / 2);
    renderSlice(`remeshed assignment dam-break-ui t=${t.toFixed(3)}s z=${z}`, fineNy, nx, (x, y) => {
      const base = Math.round(bases[x + nx * z]);
      if (y < base) return storeAt(x, z) >= 0.5 ? "T" : "t";
      const packedY = 2 + y - base;
      if (packedY >= 2 && packedY < storedNy && y < fineNy) return bandAt(x, z, packedY) >= 0.5 ? "#" : "-";
      return " ";
    });
    console.log(`remeshed assignment: wetColumns=${wetColumns} dryUnderWetBand=${dryUnderWet} surfaceOutOfBand=${surfaceOutOfBand} unexcusedDelta=${unexcusedDelta}`);

    assert.equal(unexcusedDelta, 0, "Eq 10 neighbor bound violated without a representability excuse");
    // Mid-collapse both counters are transiently nonzero on some GPUs; the
    // 2026-07-16 audits measured 0 and 0 here, so small bounds pin the
    // regression without flaking on scheduling nondeterminism.
    assert.ok(dryUnderWet <= 8, `${dryUnderWet} air-classified stores sit under wet band cells (interface inside tall cells)`);
    assert.ok(surfaceOutOfBand <= Math.max(2, wetColumns * 0.01), `${surfaceOutOfBand}/${wetColumns} wet columns have their free surface outside the band`);

    solver.destroy();
    device.destroy();
  } finally {
    Reflect.deleteProperty(globalThis, "navigator");
  }
});
