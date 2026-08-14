/**
 * Mechanical module mover for the method-decoupling refactor.
 *
 * Moves files with `git mv` and rewrites every relative/`@/` import specifier in
 * the repository so that it still resolves. Import edges are resolved against
 * the *old* layout, mapped through the move plan, and re-emitted relative to the
 * importer's *new* location.
 *
 * Usage: node --import tsx tools/move-modules.ts plan.json [--dry]
 *   plan.json: { "moves": { "lib/foo.ts": "lib/core/foo.ts", ... } }
 */
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, readdirSync, renameSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, relative, resolve } from "node:path";

const REPO = resolve(new URL("..", import.meta.url).pathname);

type Plan = { moves: Record<string, string> };

const SPEC_RE = /(\bfrom\s*|\bimport\s*\(\s*|\bimport\s+|\brequire\s*\(\s*|\bnew\s+URL\s*\(\s*)(["'])([^"']+)\2/g;

function tracked(absolutePath: string): boolean {
  try {
    execFileSync("git", ["ls-files", "--error-unmatch", "--", relative(REPO, absolutePath)],
      { cwd: REPO, stdio: "ignore" });
    return true;
  } catch {
    return false;
  }
}

function listSources(dir: string, out: string[] = []): string[] {
  if (!existsSync(dir)) return out;
  for (const name of readdirSync(dir)) {
    if (name === "node_modules" || name === ".git" || name === ".next" || name === "dist" || name === "build") continue;
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) listSources(p, out);
    else if (/\.(ts|tsx|mts|mjs)$/.test(name)) out.push(p);
  }
  return out;
}

function resolveSpec(fromFileAbs: string, spec: string): string | null {
  let base: string;
  if (spec.startsWith(".")) base = resolve(dirname(fromFileAbs), spec);
  else if (spec.startsWith("@/")) base = resolve(REPO, spec.slice(2));
  else return null;
  const candidates = [base, `${base}.ts`, `${base}.tsx`, join(base, "index.ts"), join(base, "index.tsx")];
  for (const c of candidates) if (existsSync(c) && statSync(c).isFile()) return c;
  // The target may already have been moved in this run; fall back to the literal.
  return base;
}

function stripExt(p: string): string {
  return p.replace(/\.(tsx?|mts)$/, "");
}

function toRelative(fromNewAbs: string, targetNewAbs: string, original: string): string {
  const keepExt = /\.(wgsl|json|css|svg|png|js|mjs)$/.test(original) || original.endsWith(".ts") || original.endsWith(".tsx");
  let target = keepExt ? targetNewAbs : stripExt(targetNewAbs);
  if (original.endsWith(".ts") || original.endsWith(".tsx")) target = targetNewAbs;
  else target = stripExt(targetNewAbs);
  let r = relative(dirname(fromNewAbs), target);
  if (!r.startsWith(".")) r = `./${r}`;
  return r;
}

function main() {
  const args = process.argv.slice(2);
  const dry = args.includes("--dry");
  const planPath = args.find((a) => !a.startsWith("--"));
  if (!planPath) throw new Error("usage: move-modules.ts plan.json [--dry]");
  const plan: Plan = JSON.parse(readFileSync(planPath, "utf8"));

  const moves = new Map<string, string>();
  for (const [from, to] of Object.entries(plan.moves)) {
    moves.set(resolve(REPO, from), resolve(REPO, to));
  }

  // 1. Record every import edge under the OLD layout.
  const files = listSources(REPO);
  const edges = new Map<string, Array<{ raw: string; target: string }>>();
  for (const f of files) {
    const src = readFileSync(f, "utf8");
    const list: Array<{ raw: string; target: string }> = [];
    SPEC_RE.lastIndex = 0;
    let m: RegExpExecArray | null;
    while ((m = SPEC_RE.exec(src))) {
      const spec = m[3];
      if (!spec.startsWith(".") && !spec.startsWith("@/")) continue;
      const t = resolveSpec(f, spec);
      if (t) list.push({ raw: spec, target: t });
    }
    if (list.length) edges.set(f, list);
  }

  // 2. Move the files.
  if (!dry) {
    for (const [from, to] of moves) {
      if (!existsSync(from)) {
        console.warn(`skip missing ${relative(REPO, from)}`);
        continue;
      }
      mkdirSync(dirname(to), { recursive: true });
      // `git mv` refuses a file git has never seen, and a refactor in progress
      // always has some: modules created a few steps earlier and not yet
      // committed. Those still have to travel, so fall back to a plain rename
      // and let the eventual `git add` record them at their new home.
      if (tracked(from)) execFileSync("git", ["mv", "-f", from, to], { cwd: REPO });
      else renameSync(from, to);
    }
  }

  // 3. Rewrite specifiers.
  let rewritten = 0;
  let touched = 0;
  for (const [oldFile, list] of edges) {
    const newFile = moves.get(oldFile) ?? oldFile;
    let src = "";
    try {
      src = readFileSync(dry ? oldFile : newFile, "utf8");
    } catch {
      continue;
    }
    let changed = false;
    const replacements = new Map<string, string>();
    for (const { raw, target } of list) {
      const newTarget = moves.get(target) ?? target;
      if (newTarget === target && newFile === oldFile) continue;
      const next = toRelative(newFile, newTarget, raw);
      if (next !== raw) replacements.set(raw, next);
    }
    if (!replacements.size) continue;
    SPEC_RE.lastIndex = 0;
    src = src.replace(SPEC_RE, (whole, pre, quote, spec) => {
      const next = replacements.get(spec);
      if (!next) return whole;
      changed = true;
      rewritten += 1;
      return `${pre}${quote}${next}${quote}`;
    });
    if (changed) {
      touched += 1;
      if (!dry) writeFileSync(newFile, src);
    }
  }
  console.log(`moved ${moves.size} files; rewrote ${rewritten} specifiers in ${touched} files${dry ? " (dry)" : ""}`);
}

main();
