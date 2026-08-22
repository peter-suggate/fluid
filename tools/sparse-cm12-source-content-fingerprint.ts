import { execFileSync } from "node:child_process";
import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

export const SPARSE_CM12_SOURCE_CONTENT_FINGERPRINT_KIND =
  "sparse-cm12-source-content";
export const SPARSE_CM12_SOURCE_CONTENT_FINGERPRINT_VERSION = 1;

export interface SparseCM12SourceContentEntry {
  readonly path: string;
  readonly bytes: Uint8Array;
}

export interface SparseCM12SourceContentFingerprint {
  readonly kind: typeof SPARSE_CM12_SOURCE_CONTENT_FINGERPRINT_KIND;
  readonly version: typeof SPARSE_CM12_SOURCE_CONTENT_FINGERPRINT_VERSION;
  readonly algorithm: "sha256";
  readonly scope: "git-listed-lib-tools-and-build-manifests";
  readonly fileCount: number;
  readonly sha256: string;
}

/** Hash exact source bytes with their repository-relative paths. */
export function fingerprintSparseCM12SourceEntries(
  entries: readonly SparseCM12SourceContentEntry[],
): SparseCM12SourceContentFingerprint {
  const hash = createHash("sha256");
  hash.update("sparse-cm12-source-content-v1\0");
  const ordered = [...entries].sort((left, right) => left.path.localeCompare(right.path));
  for (const entry of ordered) {
    const path = Buffer.from(entry.path, "utf8");
    hash.update(`${path.byteLength}\0`);
    hash.update(path);
    hash.update(`${entry.bytes.byteLength}\0`);
    hash.update(entry.bytes);
  }
  return {
    kind: SPARSE_CM12_SOURCE_CONTENT_FINGERPRINT_KIND,
    version: SPARSE_CM12_SOURCE_CONTENT_FINGERPRINT_VERSION,
    algorithm: "sha256",
    scope: "git-listed-lib-tools-and-build-manifests",
    fileCount: ordered.length,
    sha256: hash.digest("hex"),
  };
}

/** Fingerprint tracked plus untracked, non-ignored source inputs in the working tree. */
export async function fingerprintSparseCM12RepositorySources(
  repositoryRoot: string,
): Promise<SparseCM12SourceContentFingerprint> {
  const listing = execFileSync("git", [
    "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--",
    "lib", "tools", "package.json", "package-lock.json", "tsconfig.json",
    "next.config.mjs", "next-env.d.ts",
  ], { cwd: repositoryRoot, encoding: "utf8" });
  const paths = listing.split("\0").filter((path) => path.length > 0);
  const entries: SparseCM12SourceContentEntry[] = [];
  for (const path of paths) {
    try {
      entries.push({ path, bytes: await readFile(resolve(repositoryRoot, path)) });
    } catch (error) {
      // `git ls-files --cached` includes tracked files deleted in an uncommitted
      // production cut. Their absence is already represented by omission from
      // the exact path+byte set; do not make performance capture impossible in
      // a dirty, intentionally pruned working tree.
      if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
    }
  }
  return fingerprintSparseCM12SourceEntries(entries);
}
