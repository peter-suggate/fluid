/** Deterministic top-level WGSL function reachability pruning. */

interface WGSLFunctionRange {
  readonly name: string;
  readonly start: number;
  readonly end: number;
  readonly body: string;
}
function maskWGSLComments(source: string): string {
  let output = "";
  let index = 0;
  while (index < source.length) {
    if (source[index] === "/" && source[index + 1] === "/") {
      const end = source.indexOf("\n", index + 2);
      const stop = end < 0 ? source.length : end;
      output += " ".repeat(stop - index);
      index = stop;
      continue;
    }
    if (source[index] === "/" && source[index + 1] === "*") {
      const end = source.indexOf("*/", index + 2);
      if (end < 0) throw new RangeError("WGSL contains an unterminated block comment");
      const stop = end + 2;
      output += source.slice(index, stop).replace(/[^\n]/g, " ");
      index = stop;
      continue;
    }
    output += source[index];
    index += 1;
  }
  return output;
}

function functionRanges(source: string): readonly WGSLFunctionRange[] {
  const masked = maskWGSLComments(source);
  const declaration = /((?:@[A-Za-z_]\w*(?:\s*\([^)]*\))?\s*)*)\bfn\s+([A-Za-z_]\w*)\s*\(/g;
  const ranges: WGSLFunctionRange[] = [];
  for (let match = declaration.exec(masked); match; match = declaration.exec(masked)) {
    const open = masked.indexOf("{", declaration.lastIndex);
    if (open < 0) throw new RangeError(`WGSL function ${match[2]} has no body`);
    let depth = 1;
    let cursor = open + 1;
    for (; cursor < masked.length && depth > 0; cursor += 1) {
      if (masked[cursor] === "{") depth += 1;
      else if (masked[cursor] === "}") depth -= 1;
    }
    if (depth !== 0) throw new RangeError(`WGSL function ${match[2]} has an unterminated body`);
    ranges.push({ name: match[2], start: match.index, end: cursor, body: masked.slice(open + 1, cursor - 1) });
    declaration.lastIndex = cursor;
  }
  return ranges;
}

/**
 * Returns one valid WGSL module containing the selected compute entry point,
 * every transitively called user function, and unchanged module declarations.
 * Function source is never rewritten; unreachable top-level function ranges
 * are replaced by newlines so diagnostics retain approximately useful lines.
 */
export function isolateWGSLEntryPointModule(source: string, entryPoint: string): string {
  if (!/^[A-Za-z_]\w*$/.test(entryPoint)) throw new RangeError("WGSL entry point name is invalid");
  const ranges = functionRanges(source);
  const byName = new Map(ranges.map((range) => [range.name, range]));
  if (!byName.has(entryPoint)) throw new RangeError(`WGSL entry point ${entryPoint} was not found`);
  const reachable = new Set<string>();
  const pending = [entryPoint];
  while (pending.length > 0) {
    const name = pending.pop()!;
    if (reachable.has(name)) continue;
    reachable.add(name);
    const body = byName.get(name)!.body;
    for (const candidate of byName.keys()) {
      if (!reachable.has(candidate) && new RegExp(`\\b${candidate}\\s*\\(`).test(body)) pending.push(candidate);
    }
  }
  let output = "";
  let cursor = 0;
  for (const range of ranges) {
    output += source.slice(cursor, range.start);
    const text = source.slice(range.start, range.end);
    output += reachable.has(range.name) ? text : text.replace(/[^\n]/g, " ");
    cursor = range.end;
  }
  return output + source.slice(cursor);
}
