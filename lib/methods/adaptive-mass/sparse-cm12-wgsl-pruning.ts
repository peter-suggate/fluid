/**
 * Retain the WGSL declarations and only the function call graph needed by a
 * small entry-point family. Metal otherwise compiles every function in the
 * monolithic CM12 module even when a pipeline names one presentation kernel.
 *
 * The parse index belongs to this source owner. A resident construction reuses
 * it across entry-point families; no global cache retains source generations.
 * Each call has independent reachability sets and emits original source slices.
 */
export function createSparseCM12WGSLPruner(source: string): (roots: readonly string[]) => string {
  type FunctionSpan = { name: string; start: number; end: number; body: string };
  type GlobalSpan = { name: string; start: number; end: number; text: string };
  // Mask comments while preserving offsets/newlines. Generated WGSL is often
  // deliberately compact (several declarations per line), so line anchoring
  // misses real globals while an unmasked regex mistakes prose for syntax.
  const syntaxCharacters = source.split("");
  for (let index = 0; index < syntaxCharacters.length;) {
    if (source[index] === "/" && source[index + 1] === "/") {
      while (index < syntaxCharacters.length && source[index] !== "\n") {
        syntaxCharacters[index++] = " ";
      }
    } else if (source[index] === "/" && source[index + 1] === "*") {
      syntaxCharacters[index++] = " ";syntaxCharacters[index++] = " ";
      while (index < syntaxCharacters.length
        && !(source[index] === "*" && source[index + 1] === "/")) {
        if (source[index] !== "\n") syntaxCharacters[index] = " ";
        index += 1;
      }
      if (index < syntaxCharacters.length) {
        syntaxCharacters[index++] = " ";syntaxCharacters[index++] = " ";
      }
    } else index += 1;
  }
  const syntaxSource = syntaxCharacters.join("");
  const spans: FunctionSpan[] = [];
  const declaration = /(?:@\w+(?:\([^)]*\))?\s*)*fn\s+([A-Za-z_]\w*)\s*\(/g;
  for (let match = declaration.exec(syntaxSource); match;
    match = declaration.exec(syntaxSource)) {
    const open = syntaxSource.indexOf("{", declaration.lastIndex);
    if (open < 0) throw new Error(`WGSL function ${match[1]} has no body`);
    let depth = 0, end = open;
    for (; end < syntaxSource.length; end += 1) {
      const character = syntaxSource[end]!;
      if (character === "{") depth += 1;
      else if (character === "}" && --depth === 0) { end += 1; break; }
    }
    if (depth !== 0) throw new Error(`WGSL function ${match[1]} has an unclosed body`);
    spans.push({ name: match[1]!, start: match.index, end,
      body: syntaxSource.slice(open, end) });
    declaration.lastIndex = end;
  }
  const byName = new Map(spans.map((span) => [span.name, span]));
  const insideFunction = (offset: number) => spans.some((span) =>
    offset >= span.start && offset < span.end);
  const globals: GlobalSpan[] = [];
  const addSimpleGlobals = (pattern: RegExp) => {
    for (let match = pattern.exec(syntaxSource); match;
      match = pattern.exec(syntaxSource)) {
      if (insideFunction(match.index)) continue;
      globals.push({ name: match[1]!, start: match.index,
        end: pattern.lastIndex, text: match[0] });
    }
  };
  // WGSL has no executable global initializers. These declaration forms are
  // therefore sufficient to close the lexical dependency graph of a sliced
  // entry point. Keeping every declaration had left each tiny pipeline with
  // the monolith's complete binding and workgroup-memory topology.
  addSimpleGlobals(/(?:@\w+(?:\([^)]*\))?\s*)*\bvar(?:<[^>]+>)?\s*([A-Za-z_]\w*)[^;]*;/g);
  addSimpleGlobals(/\b(?:const|override|alias)\s+([A-Za-z_]\w*)[^;]*;/g);
  const structPattern = /\bstruct\s+([A-Za-z_]\w*)\s*\{/g;
  for (let match = structPattern.exec(syntaxSource); match;
    match = structPattern.exec(syntaxSource)) {
    if (insideFunction(match.index)) continue;
    const open = syntaxSource.indexOf("{", match.index);
    let depth = 0, end = open;
    for (; end < syntaxSource.length; end += 1) {
      if (syntaxSource[end] === "{") depth += 1;
      else if (syntaxSource[end] === "}" && --depth === 0) {
        end += 1;
        if (syntaxSource[end] === ";") end += 1;
        break;
      }
    }
    globals.push({ name: match[1]!, start: match.index, end,
      text: source.slice(match.index, end) });
    structPattern.lastIndex = end;
  }
  const globalByName = new Map(globals.map((span) => [span.name, span]));
  return (roots: readonly string[]): string => {
    const retained = new Set<string>();
    const pending = roots.filter((root) => byName.has(root));
    while (pending.length > 0) {
      const name = pending.pop()!;
      if (retained.has(name)) continue;
      retained.add(name);
      const body = byName.get(name)!.body;
      for (const call of body.matchAll(/\b([A-Za-z_]\w*)\s*\(/g)) {
        const dependency = call[1]!;
        if (byName.has(dependency) && !retained.has(dependency)) pending.push(dependency);
      }
    }
    const requiredGlobals = new Set<string>();
    const globalPending: string[] = [];
    const enqueueIdentifiers = (text: string) => {
      for (const token of text.matchAll(/\b([A-Za-z_]\w*)\b/g)) {
        const name = token[1]!;
        if (globalByName.has(name) && !requiredGlobals.has(name)) globalPending.push(name);
      }
    };
    for (const name of retained) enqueueIdentifiers(
      source.slice(byName.get(name)!.start, byName.get(name)!.end));
    while (globalPending.length > 0) {
      const name = globalPending.pop()!;
      if (requiredGlobals.has(name)) continue;
      requiredGlobals.add(name);
      enqueueIdentifiers(globalByName.get(name)!.text);
    }
    const removable = [
      ...spans.filter((span) => !retained.has(span.name)),
      ...globals.filter((span) => !requiredGlobals.has(span.name)),
    ].sort((left, right) => left.start - right.start);
    let result = "", cursor = 0;
    for (const span of removable) {
      if (span.start < cursor) continue;
      result += source.slice(cursor, span.start);
      cursor = span.end;
    }
    return result + source.slice(cursor);
  };
}

/** Compatibility entry point for callers that need only one slice. */
export function sparseCM12WGSLForEntryPoints(source: string, roots: readonly string[]): string {
  return createSparseCM12WGSLPruner(source)(roots);
}
