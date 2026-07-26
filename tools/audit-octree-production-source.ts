#!/usr/bin/env node
import { readFileSync, readdirSync, statSync } from "node:fs";
import { basename, extname, join, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import {
  auditOctreeProductionSource,
  type OctreeProductionSourceRole,
  type OctreeSourceViolation,
} from "../lib/webgpu-octree-work-accounting";

/** Presentation and one-shot inspection paths are not part of the recurring
 * simulation graph. Everything else with an octree production prefix is
 * scanned; the list is intentionally an exclusion list so new files enter the
 * gate automatically. */
const NON_RECURRING_SOURCE_BASENAMES = new Set([
  "webgpu-octree-sparse-bricks.ts",
  "webgpu-octree-technique-overlay.ts",
  "webgpu-octree-voxel-inspection.ts",
  "webgpu-octree-work-accounting.ts",
]);

export interface OctreeProductionRepositoryAudit {
  readonly scannedSources: readonly string[];
  readonly violations: readonly OctreeSourceViolation[];
}

export interface EmbeddedWgslSource {
  readonly source: string;
  readonly body: string;
  readonly bodyOffset: number;
  readonly bodyLine: number;
  readonly templateStart: number;
  readonly templateEnd: number;
}

function lineAt(source: string, offset: number): number {
  let line = 1;
  for (let index = 0; index < offset; index += 1) {
    if (source.charCodeAt(index) === 10) line += 1;
  }
  return line;
}

/** Discover WGSL template literals by syntax, not exported constant names.
 * This makes every new embedded shader enter the source gate automatically. */
export function extractEmbeddedWgslSources(
  sourceName: string,
  source: string,
): readonly EmbeddedWgslSource[] {
  const shaders: EmbeddedWgslSource[] = [];
  for (let cursor = 0; cursor < source.length;) {
    const templateStart = source.indexOf("`", cursor);
    if (templateStart < 0) break;
    let templateEnd = templateStart + 1;
    while (templateEnd < source.length) {
      if (source[templateEnd] === "\\") { templateEnd += 2; continue; }
      if (source[templateEnd] === "`") break;
      templateEnd += 1;
    }
    if (templateEnd >= source.length) break;
    const bodyOffset = templateStart + 1;
    const body = source.slice(bodyOffset, templateEnd);
    if (/(?:@(?:compute|vertex|fragment|group|binding)\b|\bstruct\s+[A-Za-z_]\w*\s*\{)/.test(body)) {
      shaders.push(Object.freeze({
        source: `${sourceName}#wgsl-${shaders.length + 1}`,
        body,
        bodyOffset,
        bodyLine: lineAt(source, bodyOffset),
        templateStart,
        templateEnd: templateEnd + 1,
      }));
    }
    cursor = templateEnd + 1;
  }
  return Object.freeze(shaders);
}

function maskEmbeddedShaders(source: string, shaders: readonly EmbeddedWgslSource[]): string {
  const characters = source.split("");
  for (const shader of shaders) {
    for (let index = shader.templateStart; index < shader.templateEnd; index += 1) {
      if (characters[index] !== "\n") characters[index] = " ";
    }
  }
  return characters.join("");
}

function walk(directory: string): string[] {
  const files: string[] = [];
  for (const entry of readdirSync(directory).sort()) {
    const path = join(directory, entry);
    const stats = statSync(path);
    if (stats.isDirectory()) files.push(...walk(path));
    else files.push(path);
  }
  return files;
}

export function discoverOctreeProductionSources(repositoryRoot: string): readonly string[] {
  const libraryRoot = join(resolve(repositoryRoot), "lib");
  return walk(libraryRoot).filter((path) => {
    const name = basename(path);
    if (NON_RECURRING_SOURCE_BASENAMES.has(name)) return false;
    const extension = extname(name);
    if (extension !== ".ts" && extension !== ".wgsl") return false;
    return name === "webgpu-octree.ts"
      || name.startsWith("webgpu-octree-")
      || name.startsWith("octree-");
  });
}

export function auditOctreeProductionRepository(
  repositoryRoot: string,
  sources: readonly string[] = discoverOctreeProductionSources(repositoryRoot),
): OctreeProductionRepositoryAudit {
  const root = resolve(repositoryRoot);
  const scannedSources: string[] = [];
  const violations: OctreeSourceViolation[] = [];
  for (const absolutePath of [...sources].sort()) {
    const sourceName = relative(root, absolutePath);
    const role: OctreeProductionSourceRole = extname(absolutePath) === ".wgsl" ? "shader" : "host";
    const source = readFileSync(absolutePath, "utf8");
    scannedSources.push(sourceName);
    if (role === "shader") {
      violations.push(...auditOctreeProductionSource(sourceName, source, role));
      continue;
    }
    const embeddedShaders = extractEmbeddedWgslSources(sourceName, source);
    violations.push(...auditOctreeProductionSource(
      sourceName, maskEmbeddedShaders(source, embeddedShaders), "host",
    ));
    for (const shader of embeddedShaders) {
      violations.push(...auditOctreeProductionSource(shader.source, shader.body, "shader")
        .map((violation) => ({
          ...violation,
          source: sourceName,
          line: shader.bodyLine + violation.line - 1,
        })));
    }
  }
  return Object.freeze({
    scannedSources: Object.freeze(scannedSources),
    violations: Object.freeze(violations.sort(
      (a, b) => a.source.localeCompare(b.source) || a.line - b.line || a.rule.localeCompare(b.rule),
    )),
  });
}

function formatViolation(violation: OctreeSourceViolation): string {
  return `${violation.source}:${violation.line}: ${violation.rule}: ${violation.excerpt}`;
}

const invokedPath = process.argv[1] ? resolve(process.argv[1]) : "";
if (invokedPath === fileURLToPath(import.meta.url)) {
  const repositoryRoot = resolve(process.argv[2] ?? process.cwd());
  const audit = auditOctreeProductionRepository(repositoryRoot);
  if (audit.violations.length === 0) {
    console.log(`octree production source audit: ${audit.scannedSources.length} sources clean`);
  } else {
    console.error(
      `octree production source audit: ${audit.violations.length} violation(s) in `
      + `${audit.scannedSources.length} sources`,
    );
    for (const violation of audit.violations) console.error(formatViolation(violation));
    process.exitCode = 1;
  }
}
