import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";
import nextTs from "eslint-config-next/typescript";

export default defineConfig([
  ...nextVitals,
  ...nextTs,
  {
    /**
     * Catch the temporal dead zone before it reaches a browser — in the one
     * place whose whole idiom invites it.
     *
     * The generators under `lib/voxel-scenery/` build module-scope tables at
     * import time, by calling helpers that read module-scope constants: a
     * layout freezes its entity list on load, and each entry's extent comes
     * from a shared budget. Move that budget below the table and the table is
     * built while the budget is still in its dead zone — `Cannot access X
     * before initialization`, at import, nothing rendered, and a stack that
     * points at the innocent helper rather than at the ordering. That happened.
     *
     * TypeScript will not catch it: the reference is perfectly well typed and
     * only the *evaluation order* is wrong. So it is a lint rule or it is a
     * runtime surprise.
     *
     * Scoped rather than global on purpose. Enabled across the repo it reports
     * about ninety hits, and nearly all of them are the safe shape — a function
     * that textually precedes a constant it only reads once called. Those are
     * not bugs, and a rule that cries ninety times to catch the ninety-first is
     * a rule that gets switched off. These modules are new, they are clean
     * under it today, and they are the ones being written fastest.
     */
    files: ["lib/voxel-scenery/**/*.ts"],
    rules: {
      "no-use-before-define": ["error", { functions: false, classes: false, variables: true, allowNamedExports: true }],
    },
  },
  globalIgnores([".next/**", "dist/**", ".wrangler/**", ".claude/**", "next-env.d.ts"])
]);
