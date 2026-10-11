/** Diagnostic only: buildDistance owns every output vertex exactly once.
 * Crossing/cache banks are complete before this consumer runs, so its bins
 * accesses require no read-modify-write operation or inter-invocation ordering.
 * Other entry points keep the original atomic storage declaration. */
import assert from "node:assert/strict";
import { pruneWGSLToEntry } from "./prune-wgsl-entry";

export function plainMetricStorageWGSL(source: string): string {
  // Instrumented counters are deliberately absent from timed consumers.
  for (let start; (start = source.indexOf("if(nbNewtonStatistics){")) >= 0;) {
    let end = start + "if(nbNewtonStatistics){".length, depth = 1;
    for (; end < source.length && depth; end++) { if (source[end] === "{") depth++; else if (source[end] === "}") depth--; }
    assert.equal(depth, 0); source = source.slice(0, start) + source.slice(end);
  }
  let code = pruneWGSLToEntry(source, "buildDistance").code;
  const declaration = "var<storage,read_write> bins:array<atomic<u32>>;";
  assert.equal(code.split(declaration).length, 2);
  code = code.replace(declaration, "var<storage,read_write> bins:array<u32>;");
  code = code.replace(/atomicLoad\(&bins\[([^\]]+)\]\)/g, "bins[$1]");
  code = code.replace(/atomicStore\(&bins\[([^\]]+)\],([^;]+)\);/g, "bins[$1]=$2;");
  assert.ok(!/atomic\w+\(&bins\[/.test(code), "Metric consumer must have only ordinary bins reads and uniquely owned writes");
  return code;
}
