import { spawnSync } from "node:child_process";
import {
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  rmSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const projectDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const buildDir = join(projectDir, "dist");
const serverEntry = join(buildDir, "server", "index.js");
const hostingConfig = join(projectDir, ".openai", "hosting.json");
const migrationsDir = join(projectDir, "drizzle");
const outputDir = join(projectDir, ".openai", "deploy");
const archive = join(outputDir, "fluid-lab-site.tgz");

function requireFile(path, description) {
  if (!existsSync(path)) {
    throw new Error(`Missing ${description}: ${path}`);
  }
}

requireFile(serverEntry, "Sites server build");
requireFile(hostingConfig, "Sites hosting configuration");

mkdirSync(outputDir, { recursive: true });
rmSync(archive, { force: true });

const stage = mkdtempSync(join(tmpdir(), "fluid-sites-package-"));

try {
  const stagedBuild = join(stage, "dist");
  const stagedHostingDir = join(stagedBuild, ".openai");

  cpSync(buildDir, stagedBuild, { recursive: true });
  mkdirSync(stagedHostingDir, { recursive: true });
  cpSync(hostingConfig, join(stagedHostingDir, "hosting.json"));

  if (existsSync(migrationsDir)) {
    cpSync(migrationsDir, join(stagedHostingDir, "drizzle"), {
      recursive: true,
    });
  }

  const packed = spawnSync(
    "tar",
    ["-C", stage, "-czf", archive, "dist"],
    { encoding: "utf8" },
  );

  if (packed.status !== 0) {
    throw new Error(packed.stderr.trim() || "tar failed to create the archive");
  }

  const listed = spawnSync("tar", ["-tzf", archive], { encoding: "utf8" });
  if (listed.status !== 0) {
    throw new Error(listed.stderr.trim() || "tar failed to inspect the archive");
  }

  const entries = new Set(listed.stdout.trim().split(/\r?\n/));
  for (const required of [
    "dist/server/index.js",
    "dist/.openai/hosting.json",
  ]) {
    if (!entries.has(required)) {
      throw new Error(`Deployment archive is missing ${required}`);
    }
  }

  console.log(`Sites deployment archive ready: ${archive}`);
  console.log("Ask Codex to publish the prepared archive to ChatGPT Sites.");
} finally {
  rmSync(stage, { recursive: true, force: true });
}
