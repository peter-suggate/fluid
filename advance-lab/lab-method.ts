export type LabMethod = "uniform-volume" | "adaptive-volume";
/** Explicit old transport links keep their original engine. */
export function labMethodFromSearch(search: string): LabMethod {
  const query = new URLSearchParams(search);
  return query.get("method") === "adaptive-volume" ||
    (!query.has("method") && query.has("transport"))
    ? "adaptive-volume"
    : "uniform-volume";
}
