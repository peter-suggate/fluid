/** Feature-owned codecs; hosts compose them without knowing individual query keys. */
export interface QueryValue<T> {
  readonly key: string;
  readonly initial: T;
  read(raw: string | null): T;
  write(value: T): string;
}
export interface QueryCodec<State> {
  readonly keys: readonly string[];
  read(query: URLSearchParams): State;
  write(query: URLSearchParams, state: State): void;
}
export function booleanQuery(key: string, initial: boolean): QueryValue<boolean> {
  return { key, initial, read: raw => raw === "1" ? true : raw === "0" ? false : initial, write: value => value ? "1" : "0" };
}
export function choiceQuery<T extends string>(key: string, initial: T, choices: readonly T[]): QueryValue<T> {
  return { key, initial, read: raw => choices.includes(raw as T) ? raw as T : initial, write: value => {
    if (!choices.includes(value)) throw new Error(`Invalid ${key}: ${value}`);
    return value;
  } };
}
export function numberQuery(key: string, initial: number, min = -Infinity, max = Infinity): QueryValue<number> {
  return { key, initial, read: raw => {
    if (raw === null || raw.trim() === "") return initial;
    const value = Number(raw);
    return Number.isFinite(value) && value >= min && value <= max ? value : initial;
  }, write: value => {
    if (!Number.isFinite(value) || value < min || value > max) throw new Error(`Invalid ${key}: ${value}`);
    return String(value);
  } };
}
export function queryRecord<State extends object>(fields: { readonly [K in keyof State]: QueryValue<State[K]> }): QueryCodec<State> {
  const entries = Object.entries(fields) as [keyof State, QueryValue<State[keyof State]>][];
  const keys = entries.map(([, field]) => field.key);
  if (new Set(keys).size !== keys.length) throw new Error("Duplicate feature query key");
  return { keys,
    read: query => Object.fromEntries(entries.map(([name, field]) => [name, field.read(query.get(field.key))])) as State,
    write: (query, state) => {
      // Validate all values before changing even the first key.
      const encoded = entries.map(([name, field]) => [field, field.write(state[name])] as const);
      for (const [field, value] of encoded) {
        query.delete(field.key);
        if (value !== field.write(field.initial)) query.set(field.key, value);
      }
    },
  };
}
export function combineQueryCodecs<State extends object>(codecs: readonly QueryCodec<any>[]): QueryCodec<State> {
  const keys = codecs.flatMap(codec => codec.keys);
  if (new Set(keys).size !== keys.length) throw new Error("Features claim the same query key");
  return { keys, read: query => Object.assign({}, ...codecs.map(codec => codec.read(query))),
    write: (query, state) => {
      const candidate = new URLSearchParams(query);
      for (const codec of codecs) codec.write(candidate, state);
      for (const key of keys) { query.delete(key); if (candidate.has(key)) query.set(key, candidate.get(key)!); }
    },
  };
}
