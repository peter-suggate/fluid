/** Residue probe for the voxel ramp scenes (lib/core/nbflip-ramp-scene.ts):
 * runs each arm in turn on one device and prints what the ramp still holds.
 * Simulation only, no rendering. Browser GPU must be off.
 *
 * WEBGPU_NODE_MODULE=<repo>/node_modules/webgpu/index.js node --import tsx tools/probe-ramp-residue.ts <arm,arm,...>
 *
 * arm = <variant>:<hz>[:option;option...]   variant: channel | staircase | flat | basin
 *   lane=off     zero the bed-slope lane after the build's bake: every tread a level floor
 *   lane=check   no solver and no run: the bake alone on the scene's voxel mask
 *                (seven pipelines), then the LANE line below
 *   pool=live    the scene's own page pool. Field readback is then not valid, so
 *                the row carries the solver's statistics and the samples only.
 *   level=<n>    basin only: the waterline in cells, in place of the scene's
 *   slice        print the centre line of the ramp, a row a wet column, at the end
 *   trace=<n>    from step n on, a row a step: the accepted pressure residual
 *                (1/s), the cycles run of those encoded, and the h band's
 *   rows         with each trace row, and with a fatal, the three rows of the
 *                pressure root the last measured cycle left with the largest
 *                residual: the 4h tile (the ramp's are x < 24, its treads'
 *                cut tiles y = 3), the residual word, the row's phi over 4h,
 *                its pressure and its pressure's lower bound
 *   div          with each half-second row, the five liquid cells whose velocity
 *                diverges most (1/s, the solver's own faces, closed where a
 *                voxel is), as x, height over the bed, z
 *
 * Every half second of simulated time a row: the liquid in cells over the ramp
 * (columns x < 96, volume times open fraction), in the pit and in all, the
 * solver's own total and its drift against the sources, the samples on the
 * ramp and how many of them are ballistic, and the largest speed. An arm ends
 * with its residue: the ramp's liquid as a fraction of what the stream emitted.
 * A fatal prints with its frame and ends the arm; the next arm still runs.
 *
 * An arm's first row, and the LANE line of a lane=check arm, compare the baked
 * lane with the ramp's own slope (-1/run along x, none along z): the floor
 * cells of the ramp (open, on a tread), how many of them the bake sloped, the
 * largest difference on them, and the non-zero entries anywhere else.
 *
 * Without pool=live every page of the box is resident from the build and
 * pinned, which is what makes the dense field readback valid: the host's page
 * table is the build's. */
import "../lib/methods";
import { fluidSolidWorldForScene } from "../lib/core/solid-world";
import { SolidOccupancyMask } from "../lib/core/solid-occupancy-mask";
import { uniformLatticeAddress } from "../lib/methods/uniform/uniform-lattice-address";
import {
  NBFLIP_RAMP_CELL_SIZE_M, NBFLIP_RAMP_GRID, NBFLIP_RAMP_LENGTH, NBFLIP_RAMP_VARIANTS,
  createNbflipRamp, nbflipRampBed, nbflipRampEmittedCells, nbflipRampShape, type NbflipRampVariant,
} from "../lib/core/nbflip-ramp-scene";
import { UniformBedSlopeBake } from "../lib/methods/uniform/uniform-bed-slope-bake";
import { uniformSolidBedSlopeAt, uniformSolidBedSlopeEntry, uniformSolidMaskPages, uniformSolidScratchLayout } from "../lib/methods/uniform/uniform-mixed-solid.wgsl";
import { advanceUniform, createUniformSolver, readUniformFields, withUniformDevice } from "../tests/helpers/uniform-geometric";
import { readMixedBuffer, readMixedTexture } from "../tests/helpers/uniform-mixed-native-fields";

const [nx, ny, nz] = NBFLIP_RAMP_GRID, H = NBFLIP_RAMP_CELL_SIZE_M, RAMP = NBFLIP_RAMP_LENGTH;
const arms = (process.argv[2] ?? "channel:60").split(",").map(text => {
  const [variant, hz, options = ""] = text.split(":");
  if (!(NBFLIP_RAMP_VARIANTS as readonly string[]).includes(variant!)) throw new Error(`Unknown ramp variant ${variant}`);
  const option = new Map(options.split(";").filter(Boolean).map(pair => pair.split("=") as [string, string?]).map(([key, value]) => [key, value ?? "on"]));
  return { text, variant: variant as NbflipRampVariant, hz: Number(hz ?? 60), option };
});

async function readPrefix(device: GPUDevice, source: GPUBuffer, bytes: number): Promise<Float32Array> {
  const size = Math.max(48, Math.min(source.size, Math.ceil(bytes / 4) * 4)), read = device.createBuffer({ size, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(source, 0, read, 0, size); device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ); return new Float32Array(read.getMappedRange()).slice();
  } finally { if (read.mapState === "mapped") read.unmap(); read.destroy(); }
}

/** `words` u32 of a buffer from its word `first`. */
async function readWords(device: GPUDevice, source: GPUBuffer, first: number, words: number): Promise<Uint32Array> {
  const read = device.createBuffer({ size: 4 * words, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  try {
    const encoder = device.createCommandEncoder(); encoder.copyBufferToBuffer(source, 4 * first, read, 0, 4 * words); device.queue.submit([encoder.finish()]);
    await read.mapAsync(GPUMapMode.READ); return new Uint32Array(read.getMappedRange()).slice();
  } finally { if (read.mapState === "mapped") read.unmap(); read.destroy(); }
}

const summaries: object[] = [];
await withUniformDevice("ramp residue probe", async device => {
  for (const arm of arms) {
    const { variant, hz, option } = arm, shape = nbflipRampShape(variant), dt = 1 / hz;
    const live = option.get("pool") === "live", laneOff = option.get("lane") === "off";
    const scene = createNbflipRamp(variant, hz), steps = Math.round(scene.duration_s * hz), every = Math.round(hz / 2);
    if (!live) scene.fluid.pagePool = { blocks: nx / 32 + 2, air: [] };
    const pool = scene.fluid.initialLiquidVolumes?.[0];
    if (option.has("level") && pool?.shape === "box") pool.max_m = { ...pool.max_m, y: Number(option.get("level")) * H };
    const mask = new SolidOccupancyMask(uniformSolidMaskPages(uniformLatticeAddress(NBFLIP_RAMP_GRID, scene.container.top === "open"))); mask.update(fluidSolidWorldForScene(scene));
    const haloed = (x: number, y: number, z: number) => (x + 1) + (nx + 2) * ((y + 1) + (ny + 2) * (z + 1));
    const solid = (x: number, y: number, z: number) => mask.solid(x, y, z);
    /** A lane against the ramp's own slope, as the lane's rounding stores it. */
    const laneReport = (words: Uint32Array) => {
      const want = uniformSolidBedSlopeAt(Uint32Array.of(uniformSolidBedSlopeEntry(shape.level ? 0 : -1 / shape.run, 0)), 0)[0];
      let floor = 0, sloped = 0, worst = 0, elsewhere = 0;
      for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
        const [slopeX, slopeZ] = uniformSolidBedSlopeAt(words, haloed(x, y, z));
        if (x < RAMP && y === nbflipRampBed(variant, x) && !solid(x, y, z) && solid(x, y - 1, z)) {
          floor++; if (slopeX !== 0 || slopeZ !== 0) sloped++;
          worst = Math.max(worst, Math.abs(slopeX - want), Math.abs(slopeZ));
        } else if (slopeX !== 0 || slopeZ !== 0) elsewhere++;
      }
      return { lane_floor_cells: floor, lane_sloped_cells: sloped, lane_analytic: +want.toFixed(5), lane_max_difference: +worst.toFixed(5), lane_elsewhere: elsewhere };
    };
    const started = performance.now();
    if (option.get("lane") === "check") {
      const layout = uniformSolidScratchLayout(NBFLIP_RAMP_GRID);
      const scratch = device.createBuffer({ label: "Ramp probe solid scratch", size: mask.words.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
      const bake = new UniformBedSlopeBake(device, scratch, { size: NBFLIP_RAMP_GRID, laneBaseWord: layout.bedSlopeBaseWord });
      try {
        device.queue.writeBuffer(scratch, 0, mask.words);
        await bake.initialize();
        const encoder = device.createCommandEncoder(); bake.encode(encoder); device.queue.submit([encoder.finish()]);
        const report = laneReport(await readWords(device, scratch, layout.bedSlopeBaseWord, layout.bedSlopeWords));
        console.log(`LANE ${JSON.stringify({ arm: arm.text, built_s: +((performance.now() - started) / 1000).toFixed(1), ...report })}`);
      } finally { bake.destroy(); scratch.destroy(); }
      continue;
    }
    const solver = await createUniformSolver(device, scene);
    const frame = (solver as unknown as { mixedFrame: {
      pagePool: { pin(pages: { x: number; y: number; z: number }[]): void }; narrowBandFlip: { activeParticles: GPUBuffer; state: GPUBuffer };
      receipts: readonly { frame: number; words: Uint32Array; floats: Float32Array }[];
      levels: readonly { residual: GPUBufferBinding; phi: GPUBufferBinding; pressure: GPUBufferBinding; minimum: readonly GPUBufferBinding[] }[];
    } }).mixedFrame;
    /** A frame's pressure verdict from its receipt: the norm it started at, the one accepted, and the cycles run (1/s, h-equivalent). */
    const verdict = (step: number) => {
      const slot = frame.receipts.find(receipt => receipt.frame === step);
      return slot ? `initial ${slot.floats[2]!.toPrecision(4)} accepted ${slot.floats[1]!.toPrecision(4)} candidate ${slot.floats[0]!.toPrecision(4)} cycles ${slot.words[7]}` : "no receipt";
    };
    /** The pressure root's three rows with the largest residual after the last measured cycle. Its owners are the box's 4h tiles, by box index. */
    const worstRows = async () => {
      const root = frame.levels[0]!, tx = nx / 4, ty = ny / 4, owners = tx * ty * (nz / 4);
      const words = async (view: GPUBufferBinding) => new Float32Array((await readWords(device, view.buffer, (view.offset ?? 0) / 4, owners)).buffer);
      const residual = await words(root.residual), phi = await words(root.phi), pressure = await words(root.pressure), bound = await words(root.minimum[0]!);
      return Array.from(residual.keys()).sort((a, b) => residual[b]! - residual[a]!).slice(0, 3)
        .map(i => `(${i % tx},${Math.floor(i / tx) % ty},${Math.floor(i / (tx * ty))}) residual ${residual[i]!.toPrecision(3)} phi/4h ${(phi[i]! / (4 * H)).toFixed(2)} p ${pressure[i]!.toPrecision(3)} bound ${bound[i]!.toPrecision(3)}`).join(" | ");
    };
    if (!live) frame.pagePool.pin(Array.from({ length: nx / 32 }, (_, x) => ({ x, y: 0, z: 0 })));

    // The lane as the solver's bake wrote it at the build, against the ramp's
    // own slope; lane=off then zeroes it, which is every tread a level floor.
    const lane = solver.solidBedSlopeLane, baked = laneReport(await readWords(device, lane.buffer, lane.baseWord, lane.words));
    if (laneOff) device.queue.writeBuffer(lane.buffer, 4 * lane.baseWord, new Uint32Array(lane.words));

    const emittedAll = nbflipRampEmittedCells(variant), rate = shape.stream ? emittedAll / shape.stream.seconds : 0;
    const treads = Math.ceil(RAMP / shape.run), treadColumns = Array.from({ length: treads }, (_, k) => (Math.min(RAMP, (k + 1) * shape.run) - k * shape.run) * (shape.wall ? shape.wall[1] - shape.wall[0] : nz));
    console.log(JSON.stringify({ arm: arm.text, created_s: +((performance.now() - started) / 1000).toFixed(1), dt, steps, lane: laneOff ? "zeroed" : "baked", ...baked, pool: live ? "live" : "pinned", emitted_cells: Math.round(emittedAll) }));
    const history: number[] = [];
    let last: Record<string, unknown> = {}, fatal: { frame: number; message: string } | undefined, stepMs = 0, speedMax = 0;
    try {
      for (let step = 1; step <= steps; step++) {
        const t0 = performance.now();
        await advanceUniform(solver, step * dt);
        stepMs += performance.now() - t0;
        if (option.has("trace") && step >= Number(option.get("trace"))) {
          const info = solver.info as unknown as Record<string, number>;
          console.log(`trace ${arm.text} step ${step}: ${verdict(step)} of ${info.uniformPressureCyclesEncoded} encoded; band ${info.uniformPressureBandTiles} tiles ${info.uniformPressureBandCycles} cycles residual ${info.uniformPressureBandResidual?.toPrecision(4)}`);
          if (option.has("rows")) console.log(`rows ${arm.text} step ${step}: ${await worstRows()}`);
        }
        if (step % every !== 0 && step !== steps) continue;
        const stats = await solver.readStats(), time = step * dt;
        const count = new Uint32Array((await readMixedBuffer(device, frame.narrowBandFlip.state)).buffer)[0]!;
        const particles = await readPrefix(device, frame.narrowBandFlip.activeParticles, 48 * count);
        let rampSamples = 0, rampBallistic = 0;
        for (let i = 0; i < count; i++) {
          const x = particles[12 * i]!;
          if (!(x >= 0) || x >= RAMP) continue;
          rampSamples++; if (particles[12 * i + 11] === 1) rampBallistic++;
        }
        let ramp: number | undefined, pit: number | undefined, tread: number[] | undefined;
        if (!live) {
          const fields = await readUniformFields(device, solver);
          ramp = 0; pit = 0; tread = new Array<number>(treads).fill(0);
          for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
            const d = solid(x, y, z) ? 0 : fields.density[x + nx * (y + ny * z)]!;
            if (!(d > 0)) continue;
            if (x < RAMP) { ramp += d; tread[Math.floor(x / shape.run)]! += d; } else pit += d;
          }
          if (option.has("div")) {
            const velocity = await readMixedTexture(device, solver.velocityTexture), worst: [number, string][] = [];
            const open = (x: number, y: number, z: number) => x >= 0 && y >= 0 && z >= 0 && x < nx && y < ny && z < nz && !solid(x, y, z);
            for (let z = 0; z < nz; z++) for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
              if (!open(x, y, z) || fields.widthAt(x, y, z) !== 1) continue;
              let phi = 0; for (let k = 0; k < 8; k++) phi += fields.vertex(x + (k & 1), y + (k >> 1 & 1), z + (k >> 2 & 1)) / 8;
              if (!(phi < 0)) continue;
              let div = 0;
              for (let axis = 0; axis < 3; axis++) {
                const e = [+(axis === 0), +(axis === 1), +(axis === 2)] as const;
                if (open(x + e[0], y + e[1], z + e[2])) div += velocity[4 * (x + nx * (y + ny * z)) + axis]!;
                if (open(x - e[0], y - e[1], z - e[2])) div -= velocity[4 * (x - e[0] + nx * (y - e[1] + ny * (z - e[2]))) + axis]!;
              }
              div /= H;
              if (worst.length < 5 || Math.abs(div) > worst[4]![0]) {
                worst.push([Math.abs(div), `(${x},+${y - nbflipRampBed(variant, x)},${z}) div ${div.toFixed(3)} phi/h ${(phi / H).toFixed(2)} v ${fields.density[x + nx * (y + ny * z)]!.toFixed(2)}`]);
                worst.sort((a, b) => b[0] - a[0]); worst.length = Math.min(5, worst.length);
              }
            }
            console.log(`div ${arm.text} step ${step}: ${worst.map(w => w[1]).join(" | ")}`);
          }
          if (step === steps && option.has("slice")) {
            const velocity = await readMixedTexture(device, solver.velocityTexture), mid = nz / 2;
            console.log(`slice z=${mid} at step ${step} (${arm.text}): x bed | phi/h at vertices bed..bed+3 | u(+x face) m/s at bed, bed+1 | volume at bed, bed+1, bed+2`);
            for (let x = 0; x < RAMP + 2; x++) {
              const bed = nbflipRampBed(variant, x), at = (k: number) => fields.density[x + nx * (bed + k + ny * mid)]!;
              if (!(at(0) + at(1) + at(2) > 0.02)) continue;
              console.log(`${x} ${bed} | ${[0, 1, 2, 3].map(k => (fields.vertex(x, bed + k, mid) / H).toFixed(2)).join(" ")} | ${[0, 1].map(k => velocity[4 * (x + nx * (bed + k + ny * mid))]!.toFixed(3)).join(" ")} | ${[0, 1, 2].map(k => at(k).toFixed(2)).join(" ")}`);
            }
          }
        }
        speedMax = Math.max(speedMax, stats.maxSpeed_m_s ?? 0);
        const emitted = rate * Math.min(time, shape.stream?.seconds ?? 0);
        last = {
          arm: arm.text, step, t: +time.toFixed(2), ms_per_step: +(stepMs / (step % every || every)).toFixed(1), emitted: Math.round(emitted),
          liquid: ramp === undefined ? null : Math.round(ramp + pit!), stats_liquid: Math.round(stats.volumeCellSum ?? NaN), drift: +(stats.volumeDrift ?? NaN).toFixed(4),
          ramp: ramp === undefined ? null : +ramp.toFixed(1), pit: pit === undefined ? null : Math.round(pit),
          ...(tread ? shape.stream ? { ramp_by_tread: tread.map(v => Math.round(v)) } : { level_by_tread: tread.map((v, k) => +(nbflipRampBed(variant, k * shape.run) + v / treadColumns[k]!).toFixed(3)) } : {}),
          samples: count, ramp_samples: rampSamples, ramp_ballistic: rampBallistic, max_speed_m_s: +(stats.maxSpeed_m_s ?? NaN).toFixed(4),
        };
        console.log(JSON.stringify(last));
        if (ramp !== undefined) history.push(+ramp.toFixed(1));
        stepMs = 0;
      }
    } catch (error) {
      const message = String((error as Error).message ?? error), frameNumber = Number(/frame (\d+)/.exec(message)?.[1] ?? NaN);
      fatal = { frame: frameNumber, message: message.slice(0, 400) };
      console.log(`FATAL (${arm.text}) frame ${frameNumber}: ${fatal.message}`);
      console.log(`receipts (${arm.text}): ${frame.receipts.map(receipt => `frame ${receipt.frame}: ${verdict(receipt.frame)}`).join("; ")}`);
      if (option.has("rows")) console.log(`rows ${arm.text} at the fatal: ${await worstRows()}`);
    } finally { solver.destroy(); }
    const complete = !fatal, ramp = last.ramp as number | null | undefined;
    const summary = {
      arm: arm.text, completed: complete, fatal,
      residue_cells: complete ? ramp ?? null : null,
      residue_fraction: complete && typeof ramp === "number" && emittedAll > 0 ? +(ramp / emittedAll).toFixed(5) : null,
      emitted_cells: Math.round(emittedAll), ramp_samples: complete ? last.ramp_samples : null, ramp_ballistic: complete ? last.ramp_ballistic : null,
      pit: complete ? last.pit : null, liquid: complete ? last.liquid : null, stats_liquid: complete ? last.stats_liquid : null, drift: complete ? last.drift : null,
      max_speed_m_s: +speedMax.toFixed(4), last_speed_m_s: last.max_speed_m_s, ...(last.level_by_tread ? { level_by_tread: last.level_by_tread } : {}),
      ramp_history: history,
    };
    summaries.push(summary);
    console.log(`SUMMARY ${JSON.stringify(summary)}`);
  }
});
console.log(`probe done: ${summaries.length} arms`);
