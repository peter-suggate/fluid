/**
 * The resolved environment a power-dam benchmark actually ran in.
 *
 * `docs/STRUCTURED_CUTOVER_LEDGER.md` C7 records the failure this exists to
 * prevent: two artifact sets for the same lane disagreed by 3.5x on ms/advance
 * and by 26x on MGPCG dispatches, and *neither* recorded what it ran with, so
 * the difference could not be attributed to a regression, a diagnostic load, or
 * a different step count. A number nobody can reproduce is not a baseline.
 *
 * So every artifact now carries three things:
 *
 * 1. `resolved` — every `FLUID_*`/`WEBGPU_*` variable the child process
 *    received. The whole thing, sorted, no editorialising.
 * 2. `inherited` — the subset that came from the caller's shell rather than the
 *    harness or the lane table. This is the one that catches a stray export.
 * 3. `contaminants` — knobs whose value is known to move the wall, resolved to
 *    something other than their measurement-clean setting. A run with a
 *    non-empty list is a diagnostic capture, not a baseline, and `clean` says so
 *    in one boolean.
 *
 * `comparisonKey` covers the scene and solver configuration, so two artifacts
 * with different keys were never measuring the same thing. It deliberately
 * excludes wall-affecting *diagnostics*: those belong to `contaminants`, where
 * they are visible rather than folded into an opaque digest.
 */

import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { arch, cpus, platform, totalmem } from "node:os";

/** Variables that describe the run rather than the machine. */
const RUN_VARIABLE = /^(?:FLUID_|WEBGPU_)/;

/**
 * Knobs whose resolved value changes the measured wall, listing every resolved
 * state a comparable throughput run may be in. `null` means "unset", which is
 * usually — but not always — the same as off: `FLUID_WEBGPU_DAWN_FEATURES`
 * unset means full Dawn validation, so only the explicit `skip_validation`
 * counts as clean there.
 *
 * `FLUID_GPU_COMMAND_AUDIT` is deliberately absent: `benchmark-power-dam.ts`
 * turns it on unconditionally, so it is part of the definition of the lane's
 * number rather than a contaminant of it. Every entry here is something a
 * caller can turn on and then forget about.
 */
const MEASUREMENT_CLEAN: Readonly<Record<string, readonly (string | null)[]>> = Object.freeze({
  // +26.8% on the large lane: a per-step queue fence removes host/GPU overlap.
  // See docs/POWER_LIQUIDS_ULTIMATE_M1MAX.md A3 and the tripwire memo.
  FLUID_TRIPWIRES: ["1"],
  FLUID_GPU_PASS_TIMESTAMPS: ["0", null],
  FLUID_GPU_FINE_TIMESTAMPS: ["0", null],
  // Reads ~19% high in aggregate; the ranking is honest, the ms/advance is not.
  FLUID_GPU_ISOLATE_PASS_LABELS: ["0", null],
  FLUID_GPU_ISOLATE_PASS_ENCODERS: ["0", null],
  FLUID_ALGORITHM_DIAGNOSTICS: ["0", null],
  FLUID_PERFORMANCE_TRACES: ["0", null],
  // Readback + host validation every step of the envelope window.
  FLUID_STABILITY_ENVELOPE: ["0", null],
  // A checkpoint is a fenced readback. `0` disables them.
  FLUID_CHECKPOINT_EVERY_S: ["0", null],
  FLUID_REGRESSION_ARTIFACT: ["0", null],
  FLUID_FIELD_STATS: ["0", null],
  FLUID_SPARSE_STATS: ["0", null],
  FLUID_RASTER_CHECKPOINTS: ["0", null],
  FLUID_POWER_GENERATION_AUDIT: ["0", null],
  FLUID_POWER_GENERATION_AUDIT_LOG: ["0", null],
  FLUID_POWER_STAGE_AUDIT: ["0", null],
  FLUID_SYMMETRY_STAGE_AUDIT: ["0", null],
  FLUID_FINE_TOPOLOGY_TRACE: ["0", null],
  FLUID_SPGRID_TOUCHED_RADIX_TRIPWIRE: ["0", null],
  // The completion-fence cadence: lowering it removes host/GPU overlap the same
  // way `FLUID_TRIPWIRES=failfast` does. Default 30.
  FLUID_AWAIT_EVERY_STEPS: ["30", null],
  // The short name wins over the lane table's pinned factor, so a stale export
  // silently re-discretizes every subsequent run. Highest-risk shell variable
  // in the tree; the lane authors FLUID_OCTREE_GLOBAL_FINE_FACTOR instead.
  FLUID_FINE_FACTOR: [null],
  // Quality-invalid probes downgrade tripwires to warnings; the run finishes,
  // but it is no longer measuring the physics the lane gates on.
  FLUID_TRIPWIRE_ALLOW: [null],
  FLUID_QUALITY_INVALID_PROBE: ["0", null],
  // Dawn validation is off by default in the benchmark harness; turning it on
  // (FLUID_BENCHMARK_VALIDATE=1) adds a per-command validation pass. Unset is
  // NOT clean here: unset means Dawn validates.
  FLUID_WEBGPU_DAWN_FEATURES: ["skip_validation"],
});

/**
 * Variables that define *what was simulated* and with which solver
 * configuration. Two runs whose values here differ are not comparable at all,
 * as opposed to comparable-but-contaminated.
 */
const COMPARISON_VARIABLE = /^FLUID_(?:SCENE|LANE|METHOD|QUALITY|TARGET_S|MAX_DT|ORACLE_STEPS|EXPECT_EXACT_STEPS|EXPECT_GRID|MAXIMUM_LEAF_SIZE|OCTREE_|SPGRID_|FINE_|STRUCTURED_|COARSE_|HYBRID_|GPU_ISOLATE_PASS_LABEL_PREFIXES$|WEBGPU_ADAPTER$|WEBGPU_BACKEND$)/;

export interface PowerDamHostEnvironment {
  readonly node: string;
  readonly platform: string;
  readonly arch: string;
  readonly cpu: string;
  readonly cores: number;
  readonly totalMemory_bytes: number;
}

export interface PowerDamGitEnvironment {
  readonly head: string | null;
  readonly branch: string | null;
  /** Null when git is unavailable, not `false` — absence is not cleanliness. */
  readonly dirty: boolean | null;
  readonly dirtyPaths: readonly string[];
}

export interface PowerDamRunEnvironment {
  readonly schemaVersion: 1;
  readonly lane: string;
  readonly argv: readonly string[];
  readonly adapter: string;
  readonly backend: string;
  readonly host: PowerDamHostEnvironment;
  readonly git: PowerDamGitEnvironment;
  /** Every run variable the child received, sorted by key. */
  readonly resolved: Readonly<Record<string, string>>;
  /** Run variables the shell supplied that the harness did not author. */
  readonly inherited: Readonly<Record<string, string>>;
  /** Run variables the shell supplied that the harness overrode, shell value first. */
  readonly overridden: Readonly<Record<string, { readonly shell: string; readonly resolved: string }>>;
  /** Wall-affecting knobs resolved away from their measurement-clean value. */
  readonly contaminants: readonly {
    readonly name: string;
    readonly resolved: string | null;
    readonly clean: readonly (string | null)[];
  }[];
  /** True when no contaminant is present: this number may be used as a baseline. */
  readonly clean: boolean;
  /** Digest over scene + solver configuration. Different key ⇒ never comparable. */
  readonly comparisonKey: string;
}

const sortedRunVariables = (
  source: Readonly<Record<string, string | undefined>>,
): Record<string, string> => {
  const entries: [string, string][] = [];
  for (const key of Object.keys(source).sort()) {
    if (!RUN_VARIABLE.test(key)) continue;
    const value = source[key];
    if (value === undefined) continue;
    entries.push([key, value]);
  }
  return Object.fromEntries(entries);
};

const gitEnvironment = (repositoryRoot: string): PowerDamGitEnvironment => {
  const git = (...argv: string[]): string | null => {
    try {
      return execFileSync("git", argv, {
        cwd: repositoryRoot, encoding: "utf8", stdio: ["ignore", "pipe", "ignore"],
      }).trim();
    } catch { return null; }
  };
  const head = git("rev-parse", "HEAD");
  const branch = git("rev-parse", "--abbrev-ref", "HEAD");
  const status = git("status", "--porcelain");
  return Object.freeze({
    head,
    branch,
    dirty: status === null ? null : status.length > 0,
    dirtyPaths: Object.freeze(status === null || status.length === 0 ? []
      : status.split("\n").map((line) => line.slice(3).trim()).sort()),
  });
};

/**
 * Resolve the environment record for one benchmark run.
 *
 * `resolved` is the environment handed to the child; `authored` is the overlay
 * the harness and lane table wrote on top of the shell. Passing them separately
 * is what makes `inherited` computable — a single merged map cannot tell a
 * deliberate lane setting from a variable somebody exported an hour ago.
 */
export function powerDamRunEnvironment(input: {
  readonly lane: string;
  readonly argv: readonly string[];
  readonly resolved: Readonly<Record<string, string | undefined>>;
  readonly authored: Readonly<Record<string, string>>;
  readonly shell?: Readonly<Record<string, string | undefined>>;
  readonly repositoryRoot: string;
  readonly host?: PowerDamHostEnvironment;
  readonly git?: PowerDamGitEnvironment;
}): PowerDamRunEnvironment {
  const resolved = sortedRunVariables(input.resolved);
  const shell = sortedRunVariables(input.shell ?? process.env);
  const inherited: Record<string, string> = {};
  const overridden: Record<string, { shell: string; resolved: string }> = {};
  for (const [key, value] of Object.entries(shell)) {
    if (!(key in input.authored)) {
      inherited[key] = value;
    } else if (input.authored[key] !== value) {
      overridden[key] = { shell: value, resolved: input.authored[key] };
    }
  }
  const contaminants: {
    name: string; resolved: string | null; clean: readonly (string | null)[];
  }[] = [];
  for (const name of Object.keys(MEASUREMENT_CLEAN).sort()) {
    const clean = MEASUREMENT_CLEAN[name];
    const actual = resolved[name];
    const normalized = actual === undefined || actual === "" ? null : actual;
    if (clean.includes(normalized)) continue;
    contaminants.push({ name, resolved: normalized, clean });
  }
  // An unrecognised run variable from the shell cannot be classified, so it is
  // reported as a contaminant on provenance grounds. This is the C7 catch-all:
  // the failure there was not a knob we knew about, it was not knowing.
  for (const name of Object.keys(inherited).sort()) {
    if (name in MEASUREMENT_CLEAN) continue;
    contaminants.push({ name, resolved: inherited[name], clean: [] });
  }
  const comparison = Object.entries(resolved)
    .filter(([key]) => COMPARISON_VARIABLE.test(key))
    .map(([key, value]) => `${key}=${value}`)
    .join("\n");
  const cpuModel = cpus()[0]?.model ?? "unknown";
  return Object.freeze({
    schemaVersion: 1 as const,
    lane: input.lane,
    argv: Object.freeze([...input.argv]),
    adapter: resolved.FLUID_WEBGPU_ADAPTER ?? "unknown",
    backend: resolved.FLUID_WEBGPU_BACKEND ?? "unknown",
    host: input.host ?? Object.freeze({
      node: process.version, platform: platform(), arch: arch(),
      cpu: cpuModel, cores: cpus().length, totalMemory_bytes: totalmem(),
    }),
    git: input.git ?? gitEnvironment(input.repositoryRoot),
    resolved: Object.freeze(resolved),
    inherited: Object.freeze(inherited),
    overridden: Object.freeze(overridden),
    contaminants: Object.freeze(contaminants),
    clean: contaminants.length === 0,
    comparisonKey: createHash("sha256").update(comparison).digest("hex").slice(0, 16),
  });
}

export interface PowerDamEnvironmentDifference {
  readonly name: string;
  readonly left: string | null;
  readonly right: string | null;
}

/**
 * Every run variable on which two captures disagree. `compare:octree-regression`
 * and any future A/B reader should print this before printing a delta: a wall
 * difference explained by an environment difference is not a result.
 */
export function powerDamEnvironmentDifferences(
  left: PowerDamRunEnvironment, right: PowerDamRunEnvironment,
): readonly PowerDamEnvironmentDifference[] {
  const names = [...new Set([
    ...Object.keys(left.resolved), ...Object.keys(right.resolved),
  ])].sort();
  const differences: PowerDamEnvironmentDifference[] = [];
  for (const name of names) {
    const a = left.resolved[name] ?? null;
    const b = right.resolved[name] ?? null;
    if (a !== b) differences.push({ name, left: a, right: b });
  }
  return Object.freeze(differences);
}
