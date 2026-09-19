// Test-only preload: flips the shared `pressureCycleBudget` param default to
// "fixed" in memory so the existing Dawn suites can be run on the off arm
// without editing a repository file another session is also working in.
import { uniformMethod } from "/Users/petersuggate/code/me/fluid/lib/methods/uniform/method";
const param = uniformMethod.params.find(p => p.key === "pressureCycleBudget");
if (!param) throw new Error("pressureCycleBudget param not found");
(param as unknown as { default: string }).default = "fixed";
console.log("[force-fixed-budget] pressureCycleBudget default -> fixed");
