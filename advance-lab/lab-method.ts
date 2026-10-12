export type LabMethod = "uniform-narrow-band-flip" | "adaptive-volume";
/** Explicit old transport links keep their original engine. Links to the
 * retired Uniform Geometric lab open narrow-band FLIP, which replaced it. */
export function labMethodFromSearch(search: string): LabMethod {
  const query = new URLSearchParams(search);
  return query.get("method") === "adaptive-volume" ||
    (!query.has("method") && query.has("transport"))
    ? "adaptive-volume"
    : "uniform-narrow-band-flip";
}
