/** Remove generated profiling artifacts and Instruments' private scratch.
 *
 * Safety contract:
 *  - only children of this checkout's `artifacts/` directory are removed;
 *  - private cleanup is limited to `instruments*.ktrace` in TMPDIR and
 *    `xrtmp__*` in InstrumentsCLI's sibling cache, plus manifest paths that
 *    resolve to one of those same roots;
 *  - an active xctrace process or Dawn's machine-wide lock aborts the cleanup.
 */
import { execFileSync } from "node:child_process";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  realpathSync,
  rmSync,
  statSync,
} from "node:fs";
import { basename, dirname, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { readWebGPUExclusiveLockHolder } from "./webgpu-smoke-isolation";

// fileURLToPath preserves the directory URL's trailing slash while dirname()
// does not. Normalise once so the exact-root safety comparison is meaningful.
const repositoryRoot = resolve(fileURLToPath(new URL("..", import.meta.url)));
const artifactRoot = resolve(repositoryRoot, "artifacts");

const isInside = (parent: string, candidate: string): boolean => {
  const rel = relative(parent, candidate);
  return rel !== "" && !rel.startsWith("..") && !rel.startsWith("/");
};

const directoryEntries = (path: string): string[] => existsSync(path)
  ? readdirSync(path).map((name) => resolve(path, name)) : [];

const manifestScratchPaths = (root: string): string[] => {
  const found: string[] = [];
  const visit = (directory: string): void => {
    for (const path of directoryEntries(directory)) {
      const metadata = statSync(path);
      if (metadata.isDirectory()) visit(path);
      else if (basename(path) === "temp-info.json") {
        try {
          const manifest = JSON.parse(readFileSync(path, "utf8")) as {
            created?: readonly { path?: unknown }[];
          };
          for (const entry of manifest.created ?? []) {
            if (typeof entry.path === "string") found.push(entry.path);
          }
        } catch (error) {
          throw new Error(`cannot parse ${path}: ${error instanceof Error ? error.message : error}`);
        }
      }
    }
  };
  if (existsSync(root)) visit(root);
  return found;
};

export const cleanupRootsForTemporaryDirectory = (temporaryDirectory: string): {
  temporary: string; instrumentsCache: string;
} => {
  const temporary = realpathSync(temporaryDirectory.replace(/\/$/, ""));
  const cache = resolve(temporary, "../C/com.apple.dt.InstrumentsCLI/path_manager");
  return { temporary, instrumentsCache: existsSync(cache) ? realpathSync(cache) : cache };
};

export const assertSafeScratchPath = (
  path: string,
  roots: ReturnType<typeof cleanupRootsForTemporaryDirectory>,
): void => {
  const parent = dirname(path);
  const name = basename(path);
  const validKtrace = realpathSync(parent) === roots.temporary
    && /^instruments.*\.ktrace$/.test(name);
  const validReductionStore = existsSync(roots.instrumentsCache)
    && realpathSync(parent) === roots.instrumentsCache && /^xrtmp__/.test(name);
  if (!validKtrace && !validReductionStore) {
    throw new Error(`refusing private cleanup outside Instruments scratch roots: ${path}`);
  }
};

const approximateBytes = (path: string): number => {
  try {
    return Number(execFileSync("du", ["-sk", path], { encoding: "utf8" }).trim().split(/\s+/)[0])
      * 1024;
  } catch { return 0; }
};

const activeXctrace = (): string | undefined => {
  try {
    const output = execFileSync("pgrep", ["-lf", "xctrace"], { encoding: "utf8" }).trim();
    return output || undefined;
  } catch { return undefined; }
};

const main = async (): Promise<void> => {
  const processEvidence = activeXctrace();
  if (processEvidence) throw new Error(`refusing cleanup while xctrace is active:\n${processEvidence}`);
  const lock = await readWebGPUExclusiveLockHolder();
  if (lock?.alive) throw new Error(`refusing cleanup while Dawn holds the GPU lock: ${lock.description}`);
  if (basename(artifactRoot) !== "artifacts" || dirname(artifactRoot) !== repositoryRoot) {
    throw new Error(`refusing unexpected artifact root ${artifactRoot}`);
  }

  const temporary = process.env.TMPDIR;
  if (!temporary) throw new Error("TMPDIR is unavailable; refusing unbounded private cleanup");
  const scratchRoots = cleanupRootsForTemporaryDirectory(temporary);
  const privatePaths = new Set(manifestScratchPaths(artifactRoot));
  for (const path of directoryEntries(scratchRoots.temporary)) {
    if (/^instruments.*\.ktrace$/.test(basename(path))) privatePaths.add(path);
  }
  for (const path of directoryEntries(scratchRoots.instrumentsCache)) {
    if (/^xrtmp__/.test(basename(path))) privatePaths.add(path);
  }

  let removedPaths = 0;
  let reclaimedBytes = 0;
  for (const path of privatePaths) {
    if (!existsSync(path)) continue;
    assertSafeScratchPath(path, scratchRoots);
    reclaimedBytes += approximateBytes(path);
    rmSync(path, { recursive: true, force: true, maxRetries: 3 });
    removedPaths += 1;
  }

  mkdirSync(artifactRoot, { recursive: true });
  for (const path of directoryEntries(artifactRoot)) {
    if (!isInside(artifactRoot, path)) throw new Error(`refusing artifact path ${path}`);
    reclaimedBytes += approximateBytes(path);
    rmSync(path, { recursive: true, force: true, maxRetries: 3 });
    removedPaths += 1;
  }
  console.log(`removed ${removedPaths} artifact/scratch paths; reclaimed approximately ${
    (reclaimedBytes / 1024 ** 3).toFixed(2)} GiB`);
};

const isMain = process.argv[1] !== undefined
  && import.meta.url === new URL(`file://${process.argv[1]}`).href;
if (isMain) await main().catch((error: unknown) => {
  console.error(`cleanup aborted: ${error instanceof Error ? error.message : String(error)}`);
  process.exitCode = 1;
});
