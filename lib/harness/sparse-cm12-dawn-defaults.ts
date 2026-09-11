import { resolveMethodValues } from "../core/method-contract";
import { adaptiveMassMethod, adaptiveMassSolverOptions } from "../methods/adaptive-volume/method";

/** The same balanced defaults used by the product, without scene-profile overrides. */
export function sparseCM12DawnDefaultValues() {
  return resolveMethodValues(adaptiveMassMethod, "balanced", {});
}

/** Low-level topology fixtures need options, but must use the same numerical policy. */
export function sparseCM12DawnDefaultOptions() {
  return adaptiveMassSolverOptions(sparseCM12DawnDefaultValues());
}
