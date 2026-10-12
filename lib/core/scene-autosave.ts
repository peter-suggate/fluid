import type { SceneDescription } from "./model";
import type { MethodProfile } from "./method-contract";
import { savedSceneCard } from "./scene-cards";
import type { SceneCard } from "./scene-definition";
import {
  deleteSceneFromLibrary,
  isSceneAutosaveEntry,
  loadSceneFromLibrary,
  SCENE_AUTOSAVE_ENTRY_ID,
  SCENE_LIBRARY_STORAGE_KEY,
  sceneLibraryEntry,
  type SceneLibraryEntry,
  type SceneLibraryStorage,
} from "./scene-library";
import { findSceneDefinition } from "./scenes";
import { resolveSession, type PaneSession } from "./session/session";
import { useShellStore } from "./stores/shell-store";

/**
 * The working document, kept so that closing the tab is not destructive.
 *
 * Everything a person authors between two explicit saves used to exist only in
 * a store, which made the library a wall between them and what they were doing:
 * *Continue* could offer nothing but scenes they had already thought to name.
 *
 * It is the same artifact as a saved scene — it reopens through the same card
 * and the same `parseScene` — but it is not kept where one is. A saved scene is
 * one string in `localStorage`, written when the reader asks. The working
 * document is written a second after every edit, and as one string that is the
 * whole document serialized on the thread the edit was made on, into a store
 * that refuses it outright once a sculpted scene passes a few megabytes.
 *
 * So it is kept in IndexedDB as parts. The document is immutable and an edit
 * shares everything it did not touch, which means a part whose value is the
 * very object that was saved last time is already saved: only the parts on the
 * path to the edit are written, and what the thread pays is the size of the
 * edit rather than the size of the scene.
 */

/** How long the edits must stop before the document is written. */
export const SCENE_AUTOSAVE_DEBOUNCE_MS = 1_000;

/** What an autosave is called when its origin scene is not this build's. */
export const SCENE_AUTOSAVE_NAME = "Untitled scene";

export const SCENE_AUTOSAVE_DATABASE = "fluid-lab.scene-autosave";
/** What a closing tab leaves in `localStorage`: the part of a save IndexedDB had not confirmed. */
export const SCENE_AUTOSAVE_JOURNAL_KEY = "fluid-lab.scene-autosave.journal";

const PARTS = "parts";
const HEAD = "head";
/** A value this light is written inside its parent's record rather than as a part of its own. */
const INLINE_WEIGHT = 64;
/** Where a chunk of an array closes: about eighty brush strokes, a twentieth of a millisecond to clone. */
const CHUNK_WEIGHT = 1_024;
/** How long one turn of a save may hold the thread before it yields to the transaction it filled. */
const SLICE_MS = 1.5;

export interface SceneAutosaveDocument {
  readonly scene: SceneDescription;
  readonly presetId: string;
  readonly methodProfile?: MethodProfile;
}

/**
 * The name to file the working document under.
 *
 * The scene it was opened from, when this build still has it: *Continue —
 * Garden pond* names the thing the reader was looking at, which a generic
 * "Untitled scene" does not. A starter or a scene from an older build has no
 * definition to borrow a name from.
 */
export function sceneAutosaveName(presetId: string): string {
  return findSceneDefinition(presetId)?.name ?? SCENE_AUTOSAVE_NAME;
}

/** `indexedDB` when the platform has it, else undefined. */
export function browserSceneAutosaveDatabase(): IDBFactory | undefined {
  try {
    return typeof indexedDB === "undefined" ? undefined : indexedDB;
  } catch {
    // As with `localStorage`, some privacy settings throw on the access itself.
    return undefined;
  }
}

/**
 * One stored record.
 *
 * `v` is a whole value. `o` is an object, each field either held (0) or the
 * key of the part that holds it (1), in the document's own field order. `a` is
 * an array as the keys of its chunks, and `c` one chunk: its elements, with a
 * placeholder wherever `p` names the part that element is kept in.
 */
type PartRecord =
  | { readonly t: "v"; readonly v: unknown }
  | { readonly t: "o"; readonly f: readonly (readonly [name: string, split: 0 | 1, held: unknown])[] }
  | { readonly t: "a"; readonly c: readonly string[] }
  | { readonly t: "c"; readonly v: readonly unknown[]; readonly p: readonly (readonly [index: number, key: string])[] };

/** A record, and the document value it was written from: meeting that value again is meeting this part. */
interface Part {
  readonly key: string;
  /** For a chunk, the elements it covers. */
  readonly value: unknown;
  readonly record: PartRecord;
  /** The parts this record names, in the order it names them. */
  readonly children: readonly Part[];
  readonly weight: number;
  /** Set once the transaction that wrote it has completed; until then a closing tab journals it. */
  stored: boolean;
}

interface AutosaveHead {
  readonly root: string;
  readonly name: string;
  readonly presetId: string;
  readonly savedAt_ms: number;
  readonly methodProfile?: MethodProfile;
  readonly writer: string;
}

type Keep = (record: PartRecord, value: unknown, children: readonly Part[], weight?: number) => Part;

const NO_PARTS: readonly Part[] = [];

function isRecord(value: object): value is Record<string, unknown> {
  const prototype: unknown = Object.getPrototypeOf(value);
  return prototype === Object.prototype || prototype === null;
}

let weighed = 0;
function weigh(value: unknown): boolean {
  weighed += 1;
  if (typeof value === "string") weighed += value.length >> 4;
  else if (typeof value === "object" && value !== null) {
    if (Array.isArray(value)) {
      for (let index = 0; index < value.length; index += 1) if (!weigh(value[index])) return false;
    } else if (isRecord(value)) {
      for (const name in value) if (!weigh(value[name])) return false;
    } else {
      // Not document JSON. It is cloned whole, as its own part.
      weighed += INLINE_WEIGHT;
    }
  }
  return weighed <= INLINE_WEIGHT;
}

/** Roughly how many values this is, counted no further than it takes to call it heavy. */
function weightOf(value: unknown): number {
  weighed = 0;
  weigh(value);
  return weighed;
}

/**
 * The part for `value`, written unless `saved` already is it.
 *
 * A generator because the first save of a session is the whole document, and
 * the caller must be able to stop between any two records: each `yield` is one
 * record handed to the transaction.
 */
function* savePart(value: unknown, saved: Part | undefined, keep: Keep): Generator<undefined, Part> {
  if (saved !== undefined && saved.value === value) return saved;
  if (Array.isArray(value)) return yield* saveArray(value, saved?.record.t === "a" ? saved : undefined, keep);
  if (typeof value === "object" && value !== null && isRecord(value)) {
    return yield* saveObject(value, saved?.record.t === "o" ? saved : undefined, keep);
  }
  const part = keep({ t: "v", v: value }, value, NO_PARTS);
  yield;
  return part;
}

function* saveObject(value: Record<string, unknown>, saved: Part | undefined, keep: Keep): Generator<undefined, Part> {
  let prior: Map<string, Part> | undefined;
  if (saved?.record.t === "o") {
    prior = new Map();
    let child = 0;
    for (const [name, split] of saved.record.f) if (split === 1) prior.set(name, saved.children[child++]!);
  }
  const fields: [string, 0 | 1, unknown][] = [];
  const children: Part[] = [];
  for (const name of Object.keys(value)) {
    const held = value[name];
    if (weightOf(held) <= INLINE_WEIGHT) { fields.push([name, 0, held]); continue; }
    // The same field of the last save, so an edit deep inside it is found by walking down rather than rewritten whole.
    const child = yield* savePart(held, prior?.get(name), keep);
    fields.push([name, 1, child.key]);
    children.push(child);
  }
  const part = keep({ t: "o", f: fields }, value, children);
  yield;
  return part;
}

function* saveArray(value: readonly unknown[], saved: Part | undefined, keep: Keep): Generator<undefined, Part> {
  const chunks: Part[] = [];
  let at = 0;
  if (saved) {
    // An append — a stroke onto the patch list — leaves every chunk before the
    // last exactly as it was saved. These are pointer comparisons.
    for (let index = 0; index < saved.children.length; index += 1) {
      const chunk = saved.children[index]!, elements = chunk.value as readonly unknown[];
      if (at + elements.length > value.length) break;
      let same = true;
      for (let element = 0; element < elements.length; element += 1) {
        if (value[at + element] !== elements[element]) { same = false; break; }
      }
      if (!same) break;
      // A last chunk with room in it is reopened, or every stroke would trail a chunk of its own.
      if (index === saved.children.length - 1 && chunk.weight < CHUNK_WEIGHT && at + elements.length < value.length) break;
      chunks.push(chunk);
      at += elements.length;
    }
  }
  let same: Map<unknown, Part> | undefined, placed: Map<number, Part> | undefined;
  /** The heavy element last saved as this very value, or failing that at this index, to walk into. */
  const prior = (element: unknown, index: number): Part | undefined => {
    if (!saved) return undefined;
    if (!same || !placed) {
      same = new Map(); placed = new Map();
      let offset = 0;
      for (const chunk of saved.children) {
        if (chunk.record.t !== "c") continue;
        for (let child = 0; child < chunk.record.p.length; child += 1) {
          const part = chunk.children[child]!;
          same.set(part.value, part);
          placed.set(offset + chunk.record.p[child]![0], part);
        }
        offset += (chunk.value as readonly unknown[]).length;
      }
    }
    return same.get(element) ?? placed.get(index);
  };
  while (at < value.length) {
    const start = at, held: unknown[] = [], split: [number, string][] = [], children: Part[] = [];
    let weight = 0;
    while (at < value.length && weight < CHUNK_WEIGHT) {
      const element = value[at], own = weightOf(element);
      if (own <= INLINE_WEIGHT) { held.push(element); weight += own; }
      else {
        const child = yield* savePart(element, prior(element, at), keep);
        held.push(null); split.push([at - start, child.key]); children.push(child);
        weight += 1;
      }
      at += 1;
    }
    chunks.push(keep({ t: "c", v: held, p: split }, value.slice(start, at), children, weight));
    yield;
  }
  const part = keep({ t: "a", c: chunks.map((chunk) => chunk.key) }, value, chunks);
  yield;
  return part;
}

function partKeys(part: Part, into: Set<string>): void {
  into.add(part.key);
  for (const child of part.children) partKeys(child, into);
}

/** A stored part was written after everything it names, so nothing under one is unstored. */
function unstoredParts(part: Part, into: [string, PartRecord][]): void {
  if (part.stored) return;
  into.push([part.key, part.record]);
  for (const child of part.children) unstoredParts(child, into);
}

function assemble(records: ReadonlyMap<string, unknown>, key: string): unknown {
  const record = records.get(key) as PartRecord | undefined;
  if (!record) throw new Error(`Autosave part ${key} is missing.`);
  if (record.t === "v") return record.v;
  if (record.t === "o") {
    const value: Record<string, unknown> = {};
    for (const [name, split, held] of record.f) value[name] = split === 1 ? assemble(records, held as string) : held;
    return value;
  }
  if (record.t !== "a") throw new Error(`Autosave part ${key} is not a value.`);
  const value: unknown[] = [];
  for (const chunkKey of record.c) {
    const chunk = records.get(chunkKey) as PartRecord | undefined;
    if (chunk?.t !== "c") throw new Error(`Autosave part ${chunkKey} is missing.`);
    const from = value.length;
    for (const element of chunk.v) value.push(element);
    for (const [index, child] of chunk.p) value[from + index] = assemble(records, child);
  }
  return value;
}

function isHead(value: unknown): value is AutosaveHead {
  if (!value || typeof value !== "object") return false;
  const head = value as Partial<AutosaveHead>;
  return typeof head.root === "string" && typeof head.savedAt_ms === "number";
}

function openParts(
  factory: IDBFactory | undefined,
  opened: (database: IDBDatabase) => void,
  refused: (error: Error) => void,
): void {
  if (!factory) { refused(new Error("This browser has no IndexedDB to keep the working scene in.")); return; }
  let request: IDBOpenDBRequest;
  try {
    request = factory.open(SCENE_AUTOSAVE_DATABASE, 1);
  } catch (error) {
    refused(error instanceof Error ? error : new Error(String(error)));
    return;
  }
  request.onupgradeneeded = () => { request.result.createObjectStore(PARTS); };
  request.onsuccess = () => opened(request.result);
  request.onerror = () => refused(request.error ?? new Error("IndexedDB refused to open."));
}

function readJournal(storage: SceneLibraryStorage | undefined): { head: AutosaveHead; parts: [string, unknown][] } | undefined {
  try {
    const raw = storage?.getItem(SCENE_AUTOSAVE_JOURNAL_KEY);
    if (!raw) return undefined;
    const journal = JSON.parse(raw) as { head?: unknown; parts?: unknown };
    return isHead(journal.head) && Array.isArray(journal.parts)
      ? { head: journal.head, parts: journal.parts as [string, unknown][] } : undefined;
  } catch {
    return undefined;
  }
}

export interface SceneAutosaveStores {
  /** Where the working document's parts are kept. */
  readonly database?: IDBFactory;
  /** `localStorage`: the journal a closing tab leaves, and the entry earlier builds wrote the whole autosave into. */
  readonly storage?: SceneLibraryStorage;
}

/**
 * The working document as a library entry, or nothing.
 *
 * A closing tab's journal is laid over the stored parts when it is the newer of
 * the two: it holds exactly the records its transaction had not confirmed. A
 * save whose parts cannot all be found is passed over rather than half opened.
 */
export function readStoredSceneAutosave(stores: SceneAutosaveStores): Promise<SceneLibraryEntry | undefined> {
  return new Promise((resolve) => {
    const journal = readJournal(stores.storage);
    const settle = (records: Map<string, unknown>) => {
      const stored = records.get(HEAD);
      const candidates: { head: AutosaveHead; records: ReadonlyMap<string, unknown> }[] = [];
      if (journal) candidates.push({ head: journal.head, records: new Map([...records, ...journal.parts]) });
      if (isHead(stored)) candidates.push({ head: stored, records });
      candidates.sort((left, right) => right.head.savedAt_ms - left.head.savedAt_ms);
      for (const { head, records: parts } of candidates) {
        try {
          const entry = sceneLibraryEntry({
            id: SCENE_AUTOSAVE_ENTRY_ID, name: head.name, savedAt_ms: head.savedAt_ms, presetId: head.presetId,
            scene: JSON.stringify(assemble(parts, head.root)),
            ...(head.methodProfile === undefined ? {} : { methodProfile: head.methodProfile }),
          });
          if (entry) { resolve(entry); return; }
        } catch {
          continue;
        }
      }
      resolve(undefined);
    };
    openParts(stores.database, (database) => {
      const records = new Map<string, unknown>();
      try {
        const transaction = database.transaction(PARTS, "readonly"), store = transaction.objectStore(PARTS);
        const keys = store.getAllKeys(), values = store.getAll();
        transaction.oncomplete = () => {
          keys.result.forEach((key, index) => records.set(String(key), values.result[index]));
          database.close();
          settle(records);
        };
        transaction.onabort = () => { database.close(); settle(new Map()); };
      } catch {
        settle(new Map());
      }
    }, () => settle(new Map()));
  });
}

/**
 * The library's rows with the working document among them.
 *
 * An entry an earlier build wrote into the library itself stays readable until
 * a save here retires it; the stored one replaces it unless it is the older.
 */
export function withStoredSceneAutosave(
  entries: readonly SceneLibraryEntry[],
  stored: SceneLibraryEntry | undefined,
): readonly SceneLibraryEntry[] {
  if (!stored) return entries;
  const legacy = entries.find(isSceneAutosaveEntry);
  if (legacy && legacy.savedAt_ms > stored.savedAt_ms) return entries;
  return [stored, ...entries.filter((entry) => !isSceneAutosaveEntry(entry))];
}

export function readSceneAutosave(entries: readonly SceneLibraryEntry[]): SceneLibraryEntry | undefined {
  return entries.find(isSceneAutosaveEntry);
}

export interface SceneResume {
  readonly entry: SceneLibraryEntry;
  readonly card: SceneCard;
  /** Where the reader left off, rather than something they chose to keep. */
  readonly autosaved: boolean;
}

/**
 * What *Continue* offers, or nothing.
 *
 * The working document wins over the newest explicit save even when the save is
 * more recent, because the question the hero answers is "where was I", not
 * "what did I keep". A candidate is only offered once its document has been
 * read: a stored entry is validated on load, so an autosave written by an older
 * schema must degrade to no Continue at all rather than to a hero button that
 * throws — and the reader's own newest save is a better answer than nothing.
 */
export function sceneResume(entries: readonly SceneLibraryEntry[]): SceneResume | undefined {
  const autosave = readSceneAutosave(entries);
  const explicit = entries.filter((entry) => !isSceneAutosaveEntry(entry))
    .sort((left, right) => right.savedAt_ms - left.savedAt_ms);
  for (const entry of autosave ? [autosave, ...explicit] : explicit) {
    try {
      loadSceneFromLibrary(entry);
    } catch {
      continue;
    }
    return { entry, card: savedSceneCard(entry), autosaved: isSceneAutosaveEntry(entry) };
  }
  return undefined;
}

export interface SceneAutosaveOptions extends SceneAutosaveStores {
  readonly delay_ms?: number;
  readonly now?: () => number;
  /** Milliseconds, for the budget of one turn of a save. */
  readonly clock?: () => number;
  /** Told once, when the store first refuses a write. Nothing is written after it. */
  readonly failed?: (error: Error) => void;
}

export interface SceneAutosave {
  /** Note a document change. The write happens once the edits stop. */
  request(working: SceneAutosaveDocument): void;
  /** Write anything outstanding now, for a tab that is going away. */
  flush(): void;
  cancel(): void;
}

interface Pass {
  readonly working: SceneAutosaveDocument;
  readonly steps: Generator<undefined, Part>;
  /** Set once the steps have run out and only their transaction is awaited. */
  root?: Part;
}

/**
 * A debounced, incremental writer.
 *
 * A gesture reaches the document many times — the tank resizer patches per
 * pointer-move — so nothing is written until the edits stop. What is written
 * then is the parts that are not the ones last written, a slice at a time, and
 * last of all a head naming the root: the head and the removal of every part it
 * no longer names are one transaction, so the store holds the previous document
 * whole until the moment it holds this one whole.
 *
 * A refused write ends autosaving for the session and says so once. Retrying
 * is what the store this replaces did, a second after every edit, against a
 * quota that was never going to move.
 */
export function createSceneAutosave(options: SceneAutosaveOptions = {}): SceneAutosave {
  const delay_ms = options.delay_ms ?? SCENE_AUTOSAVE_DEBOUNCE_MS;
  const now = options.now ?? Date.now;
  const clock = options.clock ?? (() => performance.now());
  const storage = options.storage;
  // Keys are never reused, so a part is immutable for as long as it exists and
  // an interrupted save cannot have damaged the one the head still names.
  const writer = `${now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  let minted = 0;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let asked: { scene: SceneDescription; presetId: string; profile: string } | undefined;
  let pending: SceneAutosaveDocument | undefined;
  let pass: Pass | undefined;
  /** The root of the last pass to run out, which the next is compared against. */
  let basis: Part | undefined;
  let uncommitted: { root: Part; head: AutosaveHead } | undefined;
  let commits = 0;
  let database: IDBDatabase | undefined;
  let opening = false;
  let failed = false;
  let lost = 0;
  let retired = false;
  let journaled = false;
  let sink: IDBObjectStore | undefined;
  let written: Part[] = [];

  const fail = (cause: unknown) => {
    if (failed) return;
    failed = true;
    if (timer !== undefined) clearTimeout(timer);
    timer = undefined;
    pending = undefined;
    pass = undefined;
    const error = cause instanceof Error ? cause : new Error(String(cause ?? "The store refused the write."));
    if (options.failed) options.failed(error);
    else console.error("Scene autosave has stopped; nothing further will be saved this session.", error);
  };

  const keep: Keep = (record, value, children, weight = 0) => {
    const part: Part = { key: `${writer}:${(minted += 1).toString(36)}`, value, record, children, weight, stored: false };
    // The clone of the record happens here, on this thread, which is why a slice is timed around it.
    if (sink) { sink.put(record, part.key); written.push(part); }
    return part;
  };

  const headFor = (root: Part, working: SceneAutosaveDocument): AutosaveHead => ({
    root: root.key, name: sceneAutosaveName(working.presetId), presetId: working.presetId, savedAt_ms: now(), writer,
    ...(working.methodProfile === undefined ? {} : { methodProfile: working.methodProfile }),
  });

  const clearJournal = () => {
    try { storage?.setItem(SCENE_AUTOSAVE_JOURNAL_KEY, ""); } catch { /* A journal that cannot be cleared is older than the head and loses to it. */ }
    journaled = false;
  };

  /** Once a session, after its first save has landed: what earlier builds and earlier tabs left in `localStorage`. */
  const retire = () => {
    retired = true;
    try {
      if (storage?.getItem(SCENE_AUTOSAVE_JOURNAL_KEY)) clearJournal();
      if (storage?.getItem(SCENE_LIBRARY_STORAGE_KEY)?.includes(`"id":"${SCENE_AUTOSAVE_ENTRY_ID}"`)) {
        deleteSceneFromLibrary(storage, SCENE_AUTOSAVE_ENTRY_ID);
      }
    } catch { /* Storage that cannot be read holds nothing to retire. */ }
  };

  const commit = (root: Part, head: AutosaveHead, working: SceneAutosaveDocument) => {
    if (!database) return;
    const keys = new Set<string>();
    partKeys(root, keys);
    let missing = false;
    let transaction: IDBTransaction;
    try {
      transaction = database.transaction(PARTS, "readwrite");
      const store = transaction.objectStore(PARTS), present = store.getAllKeys();
      present.onsuccess = () => {
        const found = new Set(present.result.map(String));
        // Another tab's save removes every part that is not its own. A head
        // naming a part that is gone would be an unreadable document.
        for (const key of keys) if (!found.has(key)) { missing = true; transaction.abort(); return; }
        store.put(head, HEAD);
        for (const key of found) if (key !== HEAD && !keys.has(key)) store.delete(key);
      };
    } catch (error) {
      fail(error);
      return;
    }
    commits += 1;
    uncommitted = { root, head };
    transaction.oncomplete = () => {
      commits -= 1;
      lost = 0;
      if (uncommitted?.root === root) {
        uncommitted = undefined;
        if (journaled) clearJournal();
      }
      if (!retired) retire();
      advance();
    };
    transaction.onabort = () => {
      commits -= 1;
      if (!missing) { fail(transaction.error ?? new Error("The autosave commit was aborted.")); return; }
      if (basis !== root) return;
      lost += 1;
      if (lost > 1) { fail(new Error("Another tab keeps replacing the autosaved scene.")); return; }
      // Once: write the whole document again under this tab's keys.
      basis = undefined;
      uncommitted = undefined;
      pending ??= working;
      begin();
    };
  };

  const slice = () => {
    const current = pass;
    if (!current || !database) return;
    const batch: Part[] = [];
    let transaction: IDBTransaction;
    try {
      transaction = database.transaction(PARTS, "readwrite");
      sink = transaction.objectStore(PARTS);
      written = batch;
      const start = clock();
      for (;;) {
        const step = current.steps.next();
        if (step.done) { current.root = step.value; break; }
        if (clock() - start >= SLICE_MS) break;
      }
    } catch (error) {
      sink = undefined;
      fail(error);
      return;
    }
    sink = undefined;
    // The next slice waits for this one to be on disk: that is the yield, and
    // it is what stops a head from being committed over a write that failed.
    transaction.oncomplete = () => {
      for (const part of batch) part.stored = true;
      // A flush may have run this pass out without waiting.
      if (pass !== current) return;
      if (current.root === undefined) { slice(); return; }
      pass = undefined;
      basis = current.root;
      commit(current.root, headFor(current.root, current.working), current.working);
    };
    transaction.onabort = () => fail(transaction.error ?? new Error("An autosave write was aborted."));
  };

  const open = () => {
    if (opening || database) return;
    opening = true;
    openParts(options.database, (opened) => {
      database = opened;
      opened.onversionchange = () => opened.close();
      advance();
    }, fail);
  };

  function begin(): void {
    if (failed || pass || commits > 0 || !pending) return;
    if (!database) { open(); return; }
    const working = pending;
    pending = undefined;
    pass = { working, steps: savePart(working.scene, basis, keep) };
    slice();
  }

  function advance(): void {
    if (timer === undefined) begin();
  }

  const journal = (root: Part, head: AutosaveHead) => {
    if (!storage) return;
    const parts: [string, PartRecord][] = [];
    unstoredParts(root, parts);
    try {
      storage.setItem(SCENE_AUTOSAVE_JOURNAL_KEY, JSON.stringify({ head, parts }));
      journaled = true;
    } catch {
      // More than `localStorage` will take. The head alone still opens the
      // document if the transaction carrying the parts outlives the tab.
      try {
        storage.setItem(SCENE_AUTOSAVE_JOURNAL_KEY, JSON.stringify({ head, parts: [] }));
        journaled = true;
      } catch { /* Then the transaction is all there is. */ }
    }
  };

  return {
    request(working) {
      if (failed) return;
      // Selection, panels and the shell all touch stores that reach this. The
      // document is immutable, so one that has not changed is the same object.
      const profile = JSON.stringify(working.methodProfile ?? null);
      if (asked?.scene === working.scene && asked.presetId === working.presetId && asked.profile === profile) return;
      asked = { scene: working.scene, presetId: working.presetId, profile };
      pending = working;
      open();
      if (timer !== undefined) clearTimeout(timer);
      timer = setTimeout(() => { timer = undefined; begin(); }, delay_ms);
    },
    /**
     * Everything outstanding, without yielding: the pass in flight is run out
     * and the document waiting behind it after it, into one transaction.
     *
     * A tab that is closing may not live to see that transaction complete, so
     * what it carries is also left in `localStorage` — which is synchronous,
     * and for one edit is a few hundred bytes — for the next read to lay over
     * the store.
     */
    flush() {
      if (timer !== undefined) clearTimeout(timer);
      timer = undefined;
      if (failed) return;
      if (!pass && !pending) {
        if (uncommitted) journal(uncommitted.root, uncommitted.head);
        return;
      }
      const batch: Part[] = [];
      let transaction: IDBTransaction | undefined;
      let root: Part | undefined, working: SceneAutosaveDocument | undefined;
      const finish = (steps: Generator<undefined, Part>): Part => {
        for (;;) { const step = steps.next(); if (step.done) return step.value; }
      };
      try {
        if (database) {
          transaction = database.transaction(PARTS, "readwrite");
          sink = transaction.objectStore(PARTS);
        }
        written = batch;
        if (pass) { root = pass.root ?? finish(pass.steps); working = pass.working; pass = undefined; }
        if (pending) { root = finish(savePart(pending.scene, root ?? basis, keep)); working = pending; pending = undefined; }
      } catch (error) {
        sink = undefined;
        fail(error);
        return;
      }
      sink = undefined;
      if (!root || !working) return;
      const head = headFor(root, working);
      journal(root, head);
      if (!database || !transaction) {
        // Not open yet: the journal is the save, and the parts are written when it opens.
        pending = working;
        return;
      }
      const parts = transaction;
      parts.oncomplete = () => { for (const part of batch) part.stored = true; };
      parts.onabort = () => fail(parts.error ?? new Error("An autosave write was aborted."));
      // Committed explicitly so it does not wait on a callback this tab may never run.
      if (typeof parts.commit === "function") parts.commit();
      basis = root;
      commit(root, head, working);
    },
    cancel() {
      if (timer !== undefined) clearTimeout(timer);
      timer = undefined;
      pending = undefined;
    },
  };
}

/**
 * Autosave for as long as the app is up. Returns the stop function.
 *
 * Subscribed to the stores rather than threaded through the controller, as
 * `startQueryStateSync` is, because every path that changes the document
 * already ends in the session's scene store and none of them should have to
 * remember this. `studioEntered` is the gate: on a cold load the store holds the default
 * preset that nobody chose, and offering that as *Continue* would be a lie.
 * Entering the studio is itself a trigger, so a scene opened and left alone is
 * still where the reader is.
 */
export function startSceneAutosave(
  options: SceneAutosaveOptions = {},
  // Pane A only, by design: pane B is a diff and never a library entry unless
  // it is promoted. WP3 is where a promotion hands this the session it saves.
  session: PaneSession = resolveSession(),
): () => void {
  const autosave = createSceneAutosave({
    failed: (error) => {
      console.error("Scene autosave has stopped; nothing further will be saved this session.", error);
      session.runtime.getState().setNotice(
        `Autosave has stopped — this browser refused the write (${error.message}). Save the scene by name or export it to keep your work.`,
        "warn",
      );
    },
    ...options,
  });
  const schedule = () => {
    if (!useShellStore.getState().studioEntered) return;
    const { scene, presetId } = session.scene.getState();
    const { methodId, quality, overrides } = session.method.getState();
    autosave.request({ scene, presetId, methodProfile: {
      methodId, quality, overrides: { ...(overrides[methodId] ?? {}) },
    } });
  };
  // A closing tab is exactly the case this exists for, and it never runs an
  // unmount; `visibilitychange` is the last callback a browser reliably gives.
  const onHidden = () => { if (document.visibilityState === "hidden") autosave.flush(); };
  const stopScene = session.scene.subscribe(schedule);
  const stopMethod = session.method.subscribe(schedule);
  const stopShell = useShellStore.subscribe(schedule);
  if (typeof document !== "undefined") document.addEventListener("visibilitychange", onHidden);
  schedule();

  return () => {
    stopScene();
    stopMethod();
    stopShell();
    if (typeof document !== "undefined") document.removeEventListener("visibilitychange", onHidden);
    autosave.flush();
  };
}
