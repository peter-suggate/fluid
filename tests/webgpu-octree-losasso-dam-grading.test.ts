import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import test from "node:test";
import { fileURLToPath } from "node:url";

interface ControlSummary {
  epoch: number;
  published?: number;
  errors: number;
}

interface DamSample {
  t_s: number;
  frontCells: number;
  topologyLeaves: number;
  leafCountsBySize: Record<string, number>;
  surfaceRowSizeHistogram: Record<string, number>;
  airRowSizeHistogram: Record<string, number>;
  reconstructionSignMismatches: number;
  topologyTransition: {
    acceptedAuthority: ControlSummary;
    candidateAuthority: ControlSummary;
    acceptedGraph: ControlSummary;
    candidateGraph: ControlSummary;
    mass: { valid: number; errors: number; missingRecipients: number };
    candidateVelocityStencil: Record<string, number>;
  };
  candidateAuthority?: number[];
  candidateGraph?: number[];
  massControl?: number[];
  velocityMigration?: number[];
}

interface DamProbeResult {
  phase: "dam-surface-shape";
  dimensions: [number, number, number];
  validationErrors: unknown[];
  samples: DamSample[];
}

function probeResult(stdout: string): DamProbeResult | undefined {
  return stdout.split("\n").flatMap((line) => {
    try {
      const value = JSON.parse(line) as Partial<DamProbeResult>;
      return value.phase === "dam-surface-shape" ? [value as DamProbeResult] : [];
    } catch {
      return [];
    }
  }).at(-1);
}

test("Dawn preserves graded dam topology while its front crosses 0.156 s", {
  skip: !process.env.WEBGPU_NODE_MODULE
    && "set WEBGPU_NODE_MODULE for the focused Losasso dam grading gate",
  timeout: 180_000,
}, () => {
  // Do not inherit experiments which silently change the topology policy. The
  // test intentionally exercises the same factor-one adaptive lane as the UI.
  const cleanEnvironment = Object.fromEntries(
    Object.entries(process.env).filter(([key]) => !key.startsWith("FLUID_")),
  );
  const root = fileURLToPath(new URL("..", import.meta.url));
  const child = spawnSync(process.execPath,
    ["--import", "tsx", "tools/probe-dam-surface-shape.ts"], {
      cwd: root,
      encoding: "utf8",
      timeout: 150_000,
      killSignal: "SIGKILL",
      maxBuffer: 32 * 1024 * 1024,
      env: {
        ...cleanEnvironment,
        NODE_ENV: process.env.NODE_ENV ?? "test",
        FLUID_WEBGPU_BACKEND: process.env.FLUID_WEBGPU_BACKEND ?? "metal",
        ...(process.env.FLUID_WEBGPU_ADAPTER
          ? { FLUID_WEBGPU_ADAPTER: process.env.FLUID_WEBGPU_ADAPTER } : {}),
        // Match the UI cadence which exposed loss of a retained, near-zero
        // negative phi value during the candidate handoff at 0.136 s.
        FLUID_SAMPLE_TIMES_S: "0.08,0.136,0.16,0.20",
        // Match the UI and long-horizon parity cadence. The adaptive evidence
        // transaction is intentionally evaluated every 0.004 s step.
        FLUID_MAX_DT: "0.004",
        FLUID_SURFACE_COMPACT: "1",
        FLUID_TOPOLOGY_TRANSITION_DIAGNOSTICS: "1",
      },
    });
  assert.equal(child.status, 0,
    `dam grading probe failed\nstdout:\n${child.stdout}\nstderr:\n${child.stderr}`);
  const result = probeResult(child.stdout);
  assert.ok(result, `dam grading probe emitted no result record\n${child.stdout}`);
  assert.deepEqual(result.validationErrors, []);
  assert.equal(result.samples.length, 4);

  const domainCells = result.dimensions.reduce((product, size) => product * size, 1);
  for (const sample of result.samples) {
    const representedCells = Object.entries(sample.leafCountsBySize)
      .reduce((sum, [size, count]) => sum + Number(size) ** 3 * count, 0);
    assert.equal(representedCells, domainCells,
      `t=${sample.t_s}: leaf census must exactly cover the domain`);

    const coarseCells = Object.entries(sample.leafCountsBySize)
      .filter(([size]) => Number(size) >= 2)
      .reduce((sum, [size, count]) => sum + Number(size) ** 3 * count, 0);
    assert.ok(coarseCells / domainCells >= 0.40,
      `t=${sample.t_s}: pressure topology lost its coarse bulk/far-field grading`);
    assert.ok((sample.leafCountsBySize["4"] ?? 0) > 0,
      `t=${sample.t_s}: the size-4 far-field tier disappeared`);
    assert.ok((sample.leafCountsBySize["2"] ?? 0) > 0,
      `t=${sample.t_s}: the mandatory 2:1 transition tier disappeared`);
    assert.ok(sample.topologyLeaves <= 4_200,
      `t=${sample.t_s}: ${sample.topologyLeaves} leaves regressed toward the 6,912-cell dense grid`);
    // The cold authored graph is still retiring at 0.08 s. Once the moving
    // dam front owns the adaptive evidence, Ando's interface tier must remain
    // finest; the rho=.5 shell may not alternate back to size two.
    if (sample.t_s >= 0.136) {
      assert.deepEqual(Object.keys(sample.surfaceRowSizeHistogram)
        .filter((size) => (sample.surfaceRowSizeHistogram[size] ?? 0) > 0), ["1"],
      `t=${sample.t_s}: an octree leaf crossing the free surface was coarsened`);
    }

    // Ando-style factor one remeshes the pressure/level-set octree while the
    // graph-owned fixed-point mass survives the handoff. The free-surface
    // histogram above proves the crossing tier; these receipts prove that the
    // graded transaction did not retain a prior graph or lose mass recipients.
    const transition = sample.topologyTransition;
    assert.equal(transition.acceptedAuthority.errors, 0, `t=${sample.t_s}: accepted authority`);
    assert.equal(transition.acceptedGraph.errors, 0, `t=${sample.t_s}: accepted graph`);
    assert.equal(transition.acceptedGraph.published, transition.acceptedGraph.epoch,
      `t=${sample.t_s}: accepted graph publication`);
    const hasCandidate = transition.candidateAuthority.epoch !== 0;
    if (hasCandidate) {
      assert.equal(transition.candidateAuthority.errors, 0, `t=${sample.t_s}: candidate authority`);
      assert.equal(transition.candidateGraph.errors, 0, `t=${sample.t_s}: candidate graph`);
      assert.equal(transition.candidateGraph.published, transition.candidateGraph.epoch,
        `t=${sample.t_s}: candidate graph publication`);
      assert.equal(transition.mass.valid, 1, `t=${sample.t_s}: mass authority`);
      assert.equal(transition.mass.errors, 0, `t=${sample.t_s}: mass receipt`);
      assert.equal(transition.mass.missingRecipients, 0,
        `t=${sample.t_s}: transport escaped the adaptive graph`);
    }
    // Leaf-average rho and retained nodal phi are distinct sub-cell
    // representations across a 2:1 handoff. A few sign disagreements are
    // diagnostic; the conservative/finite mass receipt above is authoritative.
    assert.ok(sample.reconstructionSignMismatches <= 8,
      `t=${sample.t_s}: rho/phi handoff disagreement escaped the local interface`);

    if (hasCandidate) {
      const stencil = transition.candidateVelocityStencil;
      assert.equal(stencil["4"], 0, `t=${sample.t_s}: candidate velocity stencil`);
      assert.equal(stencil["3"], stencil["0"], `t=${sample.t_s}: stencil publication`);
      assert.ok((stencil["6"] ?? Infinity) <= 4,
        `t=${sample.t_s}: nodal velocity stencil included non-incident faces`);
    }
  }

  const at = (time: number) => result.samples.find((sample) => sample.t_s === time)!;
  assert.ok(at(0.16).frontCells >= at(0.08).frontCells + 1,
    "dam front did not advance through the first graded transition");
  assert.ok(at(0.20).frontCells >= at(0.16).frontCells + 1,
    "dam front stalled immediately after the first graded transition");
});

test("Dawn consumes every default dam step snapshot through 0.080 s", {
  skip: !process.env.WEBGPU_NODE_MODULE
    && "set WEBGPU_NODE_MODULE for the Losasso step-snapshot gate",
  timeout: 180_000,
}, () => {
  const cleanEnvironment = Object.fromEntries(
    Object.entries(process.env).filter(([key]) => !key.startsWith("FLUID_")),
  );
  const root = fileURLToPath(new URL("..", import.meta.url));
  const child = spawnSync(process.execPath,
    ["--import", "tsx", "tools/probe-dam-surface-shape.ts"], {
      cwd: root,
      encoding: "utf8",
      timeout: 150_000,
      killSignal: "SIGKILL",
      maxBuffer: 32 * 1024 * 1024,
      env: {
        ...cleanEnvironment,
        NODE_ENV: process.env.NODE_ENV ?? "test",
        FLUID_WEBGPU_BACKEND: process.env.FLUID_WEBGPU_BACKEND ?? "metal",
        ...(process.env.FLUID_WEBGPU_ADAPTER
          ? { FLUID_WEBGPU_ADAPTER: process.env.FLUID_WEBGPU_ADAPTER } : {}),
        FLUID_SAMPLE_TIMES_S: Array.from({ length: 20 }, (_, index) =>
          ((index + 1) * 0.004).toFixed(3)).join(","),
        FLUID_MAX_DT: "0.004",
        FLUID_SURFACE_COMPACT: "1",
        FLUID_TOPOLOGY_TRANSITION_DIAGNOSTICS: "1",
      },
    });
  assert.equal(child.status, 0,
    `per-step snapshot probe failed\nstdout:\n${child.stdout}\nstderr:\n${child.stderr}`);
  const result = probeResult(child.stdout);
  assert.ok(result, `per-step snapshot probe emitted no result record\n${child.stdout}`);
  assert.deepEqual(result.validationErrors, []);
  assert.equal(result.samples.length, 20);
  assert.equal(result.samples.at(-1)?.t_s, 0.08);
  assert.ok(result.samples.some((sample) =>
    sample.topologyTransition.candidateAuthority.epoch === 0),
  "the cadence did not exercise a dormant candidate snapshot");
});

test("Dawn keeps the full-tank leaf-8 candidate transaction coherent through 0.240 s", {
  skip: !process.env.WEBGPU_NODE_MODULE
    && "set WEBGPU_NODE_MODULE for the focused Losasso candidate gate",
  timeout: 180_000,
}, () => {
  const cleanEnvironment = Object.fromEntries(
    Object.entries(process.env).filter(([key]) => !key.startsWith("FLUID_")),
  );
  const root = fileURLToPath(new URL("..", import.meta.url));
  const child = spawnSync(process.execPath,
    ["--import", "tsx", "tools/probe-dam-surface-shape.ts"], {
      cwd: root,
      encoding: "utf8",
      timeout: 150_000,
      killSignal: "SIGKILL",
      maxBuffer: 32 * 1024 * 1024,
      env: {
        ...cleanEnvironment,
        NODE_ENV: process.env.NODE_ENV ?? "test",
        FLUID_WEBGPU_BACKEND: process.env.FLUID_WEBGPU_BACKEND ?? "metal",
        ...(process.env.FLUID_WEBGPU_ADAPTER
          ? { FLUID_WEBGPU_ADAPTER: process.env.FLUID_WEBGPU_ADAPTER } : {}),
        // Consume and re-arm the UI-equivalent step receipt at every fixed
        // advance, including the historical epoch-26 rejection.
        FLUID_SAMPLE_TIMES_S: Array.from({ length: 60 }, (_, index) =>
          ((index + 1) * 0.004).toFixed(3)).join(","),
        FLUID_MAX_DT: "0.004",
        FLUID_TRANSACTION_ONLY: "1",
        FLUID_SURFACE_COMPACT: "1",
        FLUID_TOPOLOGY_TRANSITION_DIAGNOSTICS: "1",
        FLUID_REFINEMENT_REGION_FLOOR: "8",
        FLUID_REFINEMENT_REGION_SCOPE: "full",
      },
    });
  assert.equal(child.status, 0,
    `leaf-8 candidate probe failed\nstdout:\n${child.stdout}\nstderr:\n${child.stderr}`);
  const result = probeResult(child.stdout);
  assert.ok(result, `leaf-8 candidate probe emitted no result record\n${child.stdout}`);
  assert.deepEqual(result.validationErrors, []);
  assert.equal(result.samples.length, 60);
  const sample = result.samples.at(-1)!;
  const authority = sample.candidateAuthority!;
  const graph = sample.candidateGraph!;
  const mass = sample.massControl!;
  const migration = sample.velocityMigration!;
  assert.deepEqual(authority.slice(3, 5), [1, 0]);
  assert.deepEqual(graph.slice(3, 5), [graph[0], 0]);
  assert.equal(graph[6], graph[5]);
  assert.deepEqual([mass[1], mass[7], mass[12]], [authority[0], 1, 0]);
  assert.deepEqual([migration[0], migration[1], migration[4], migration[5]],
    [authority[0], authority[2], 0, authority[0]]);
  assert.equal((migration[2] ?? 0) + (migration[3] ?? 0), migration[1]);
  assert.equal(migration[6], 4, "bounded topology-local completion rounds");
  assert.ok((migration[7] ?? Infinity) <= (migration[3] ?? 0),
    "fallback face count must remain visible and bounded");
  const epoch26 = result.samples.find((candidate) => candidate.candidateAuthority?.[0] === 26);
  assert.ok(epoch26, "UI-equivalent per-step lane did not exercise candidate epoch 26");
  assert.deepEqual(epoch26.candidateAuthority!.slice(3, 5), [1, 0]);
  assert.deepEqual(epoch26.candidateGraph!.slice(3, 5), [26, 0]);
  assert.deepEqual([epoch26.massControl![1], epoch26.massControl![7],
    epoch26.massControl![12]], [26, 1, 0]);
  assert.deepEqual([epoch26.velocityMigration![0], epoch26.velocityMigration![4],
    epoch26.velocityMigration![5], epoch26.velocityMigration![6]], [26, 0, 26, 4]);
  assert.equal((epoch26.velocityMigration![2] ?? 0) + (epoch26.velocityMigration![3] ?? 0),
    epoch26.velocityMigration![1]);
});
