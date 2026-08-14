/**
 * Compatibility re-export: the audit ABI moved to
 * `lib/methods/octree-shared/structured-authority-audit.ts`.
 *
 * The record layout is the format of buffers the shared engine's step-snapshot
 * ring copies and decodes, so it may not sit inside a method package. This
 * path survives only because `lib/harness/webgpu-smoke-structured-audit.ts`
 * still re-exports through it; repoint that import and delete this file.
 */
export * from "../octree-shared/structured-authority-audit";
