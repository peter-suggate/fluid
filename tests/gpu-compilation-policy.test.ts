import assert from "node:assert/strict";
import { readFile, readdir } from "node:fs/promises";
import path from "node:path";
import test from "node:test";
import ts from "typescript";

async function sourceFiles(directory: string): Promise<string[]> {
  const entries = await readdir(directory, { withFileTypes: true });
  const nested = await Promise.all(entries.map(async (entry) => {
    const target = path.join(directory, entry.name);
    if (entry.isDirectory()) return sourceFiles(target);
    return entry.isFile() && /\.(?:ts|tsx)$/.test(entry.name) ? [target] : [];
  }));
  return nested.flat();
}

test("production code cannot call immediate WebGPU pipeline compilation", async () => {
  const root = path.resolve(import.meta.dirname, "../lib");
  const violations: string[] = [];
  for (const file of await sourceFiles(root)) {
    const source = await readFile(file, "utf8");
    source.split("\n").forEach((line, index) => {
      if (/\.create(?:Compute|Render)Pipeline\s*\(/.test(line)) {
        violations.push(`${path.relative(root, file)}:${index + 1}`);
      }
    });
  }
  assert.deepEqual(violations, [],
    "All production pipelines must go through async compilation; immediate API calls are forbidden");
});

test("production constructors cannot start shader or pipeline compilation", async () => {
  const root = path.resolve(import.meta.dirname, "../lib");
  const compilationCalls = new Set([
    "createShaderModule",
    "createComputePipeline",
    "createComputePipelineAsync",
    "createRenderPipeline",
    "createRenderPipelineAsync",
  ]);
  const violations: string[] = [];

  for (const file of await sourceFiles(root)) {
    const source = await readFile(file, "utf8");
    const parsed = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true,
      file.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS);
    const visit = (node: ts.Node, insideConstructor = false): void => {
      const constructor = insideConstructor || ts.isConstructorDeclaration(node);
      if (constructor && ts.isCallExpression(node)
        && ts.isPropertyAccessExpression(node.expression)
        && compilationCalls.has(node.expression.name.text)) {
        const line = parsed.getLineAndCharacterOfPosition(node.getStart(parsed)).line + 1;
        violations.push(`${path.relative(root, file)}:${line}:${node.expression.name.text}`);
      }
      ts.forEachChild(node, (child) => visit(child, constructor));
    };
    visit(parsed);
  }

  assert.deepEqual(violations, [],
    "Constructors must only allocate lightweight state; compilation belongs to GPUCompilationManager tasks");
});
