/** Keep large scene documents in the fragment, which is never sent over HTTP. */
const STATE_FRAGMENT = "#fluid-state=";
const QUERY_BUDGET = 4096;

type QueryLocation = Pick<Location, "search" | "hash">;

function fragmentState(hash: string): { query: string; hash: string } | undefined {
  if (!hash.startsWith(STATE_FRAGMENT)) return;
  try {
    const value: unknown = JSON.parse(decodeURIComponent(hash.slice(STATE_FRAGMENT.length)));
    if (value && typeof value === "object" && "query" in value && "hash" in value
      && typeof value.query === "string" && typeof value.hash === "string") {
      return { query: value.query, hash: value.hash };
    }
  } catch { /* An unrelated or malformed fragment is not scene state. */ }
}

/** Read both parts of a shared link. Explicit query parameters take precedence. */
export function readLocationQuery(location: QueryLocation = window.location): string {
  const state = fragmentState(location.hash);
  if (!state) return location.search;
  const query = new URLSearchParams(state.query);
  const explicit = new URLSearchParams(location.search);
  for (const key of explicit.keys()) query.delete(key);
  for (const [key, value] of explicit) query.append(key, value);
  return query.size ? `?${query}` : "";
}

/** Keep ordinary links readable; spill the largest fields when the query grows. */
export function queryLocationUrl(pathname: string, search: string, hash = ""): string {
  const query = new URLSearchParams(search);
  const anchor = fragmentState(hash)?.hash ?? hash;
  let encoded = query.toString();
  if (encoded.length <= QUERY_BUDGET) return `${pathname}${encoded ? `?${encoded}` : ""}${anchor}`;

  const overflow = new URLSearchParams();
  const keys = [...new Set(query.keys())].sort((a, b) => {
    const size = (key: string) => new URLSearchParams(query.getAll(key).map(value => [key, value])).toString().length;
    return size(b) - size(a);
  });
  for (const key of keys) {
    for (const value of query.getAll(key)) overflow.append(key, value);
    query.delete(key);
    encoded = query.toString();
    if (encoded.length <= QUERY_BUDGET) break;
  }
  const fragment = STATE_FRAGMENT + encodeURIComponent(JSON.stringify({ query: overflow.toString(), hash: anchor }));
  return `${pathname}${encoded ? `?${encoded}` : ""}${fragment}`;
}
