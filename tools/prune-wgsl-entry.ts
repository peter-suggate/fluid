/** Diagnostic WGSL specialization. Reachable function text is copied verbatim;
 * only unreachable functions and unreferenced workgroup declarations go away.
 * Retain all other module declarations, types, overrides and resource bindings.
 */
import assert from "node:assert/strict";
function uncomment(source: string): string {
  const text = source.split(""); let block = 0, line = false;
  for (let i = 0; i < source.length; i++) {
    const a = source[i], b = source[i + 1];
    if (line) { if (a === "\n") line = false; else text[i] = " "; continue; }
    if (block) {
      if (a === "/" && b === "*") { block++; text[i] = text[++i] = " "; }
      else if (a === "*" && b === "/") { block--; text[i] = text[++i] = " "; }
      else if (a !== "\n") text[i] = " ";
      continue;
    }
    if (a === "/" && b === "/") { line = true; text[i] = text[++i] = " "; }
    else if (a === "/" && b === "*") { block = 1; text[i] = text[++i] = " "; }
  }
  assert.equal(block, 0, "WGSL block comments must be balanced"); return text.join("");
}
const names = (text: string) => new Set(text.match(/\b[A-Za-z_]\w*\b/g) ?? []);
export function pruneWGSLToEntry(source: string, entryPoint: string): { code: string; functionsBefore: number; functionsAfter: number; workgroupDeclarationsRemoved: number } {
  const clean = uncomment(source), functions = new Map<string, { start: number; end: number; body: string }>();
  const declarations = /\bfn\s+([A-Za-z_]\w*)\s*\(/g;
  for (let match; (match = declarations.exec(clean));) {
    const name = match[1]!; assert.ok(!functions.has(name), `Duplicate WGSL function ${name}`);
    const prefix = clean.slice(0, match.index).match(/(?:@[A-Za-z_]\w*(?:\([^)]*\))?\s*)+$/);
    const start = prefix ? match.index - prefix[0].length : match.index;
    const body = clean.indexOf("{", declarations.lastIndex); assert.ok(body >= 0);
    let depth = 1, end = body + 1;
    for (; end < clean.length && depth; end++) { if (clean[end] === "{") depth++; else if (clean[end] === "}") depth--; }
    assert.equal(depth, 0, `WGSL function ${name} must have balanced braces`);
    functions.set(name, { start, end, body: clean.slice(body + 1, end - 1) }); declarations.lastIndex = end;
  }
  assert.ok(functions.has(entryPoint), `WGSL entry ${entryPoint} is missing`);
  const reachable = new Set([entryPoint]);
  for (const name of reachable) for (const dependency of names(functions.get(name)!.body)) if (functions.has(dependency)) reachable.add(dependency);
  const edits: { start: number; end: number }[] = [];
  for (const [name, fn] of functions) if (!reachable.has(name)) edits.push(fn);
  const retained = [...functions].filter(([name]) => reachable.has(name)).map(([, fn]) => fn.body).join("\n");
  const retainedNames = names(retained); let workgroupDeclarationsRemoved = 0;
  let outside = clean;
  for (const fn of [...functions.values()].sort((a, b) => b.start - a.start)) outside = outside.slice(0, fn.start) + " ".repeat(fn.end - fn.start) + outside.slice(fn.end);
  const counts = new Map<string, number>();
  for (const token of outside.match(/\b[A-Za-z_]\w*\b/g) ?? []) counts.set(token, (counts.get(token) ?? 0) + 1);
  const workgroups = /\bvar\s*<\s*workgroup\s*>\s*([A-Za-z_]\w*)\s*:/g;
  for (let match; (match = workgroups.exec(clean));) {
    const end = clean.indexOf(";", workgroups.lastIndex) + 1; assert.ok(end > 0);
    // Conservatively keep a workgroup variable used anywhere outside a
    // function (e.g. by another declaration), in addition to live bodies.
    if (!retainedNames.has(match[1]!) && (counts.get(match[1]!) ?? 0) <= 1) { edits.push({ start: match.index, end }); workgroupDeclarationsRemoved++; }
  }
  let code = source;
  for (const edit of edits.sort((a, b) => b.start - a.start)) code = code.slice(0, edit.start) + code.slice(edit.end);
  // Guards against accidental overlap or lexical removal of a live body.
  for (const name of reachable) { const fn = functions.get(name)!; assert.ok(code.includes(source.slice(fn.start, fn.end)), `Live WGSL function ${name} changed`); }
  return { code, functionsBefore: functions.size, functionsAfter: reachable.size, workgroupDeclarationsRemoved };
}
