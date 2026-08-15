/**
 * Stream-parse `xctrace export` XML tables into NDJSON rows.
 *
 * Instruments emits an id/ref-compressed dialect: a value appears once carrying
 * `id="N"`, and every later occurrence is `<tag ref="N"/>`. Composite columns
 * (`formatted-label`, `tagged-backtrace`) nest child elements that themselves
 * define ids consumed by *later* rows, so every element has to be registered
 * even though only the depth-0 children of `<row>` map to schema columns.
 *
 * Traces are gigabytes, so this parses incrementally off a read stream and
 * never holds the document in memory. Only the id table grows, which is
 * inherent to the format.
 */
import { createReadStream } from "node:fs";
import { createInterface } from "node:readline";

export interface TraceRow {
  readonly [column: string]: string | readonly string[] | null | undefined;
  /** Leaf-first call stack, present only when `stacks` is requested. */
  readonly frames?: readonly string[];
}

const TAG = /<([a-zA-Z0-9_-]+)((?:\s+[a-zA-Z0-9_:-]+="[^"]*")*)\s*(\/?)>|<\/([a-zA-Z0-9_-]+)>/g;
const ATTR = /([a-zA-Z0-9_:-]+)="([^"]*)"/g;

const decodeEntities = (value: string): string => (value.includes("&")
  ? value.replace(/&lt;/g, "<").replace(/&gt;/g, ">").replace(/&quot;/g, "\"")
    .replace(/&#39;/g, "'").replace(/&apos;/g, "'").replace(/&amp;/g, "&")
  : value);

const parseAttributes = (raw: string): Record<string, string> => {
  const result: Record<string, string> = {};
  if (!raw) return result;
  ATTR.lastIndex = 0;
  let match: RegExpExecArray | null;
  // Attribute values are entity-encoded too; `fmt` in particular carries the
  // display text of composite columns, where a literal "&" separates the
  // encoders Instruments merged into one interval.
  while ((match = ATTR.exec(raw)) !== null) result[match[1]] = decodeEntities(match[2]);
  return result;
};

interface Frame {
  readonly kind?: "schema" | "mnemonic" | "row" | "other";
  readonly id?: string;
  readonly fmt?: string;
  readonly isColumn?: boolean;
  readonly counted?: boolean;
  readonly backtraceFrames?: string[];
}

export interface ParseTableOptions {
  /** Materialise `tagged-backtrace` frame lists as `row.frames`. */
  readonly stacks?: boolean;
  /** Restrict emitted columns; every column is kept when omitted. */
  readonly columns?: readonly string[];
}

/** Yield one object per `<row>` of an exported Instruments table. */
export async function* parseTraceTable(
  source: string | AsyncIterable<string>,
  options: ParseTableOptions = {},
): AsyncGenerator<TraceRow> {
  const wanted = options.columns ? new Set(options.columns) : undefined;
  const values = new Map<string, { fmt?: string; text: string }>();
  const frameNames = new Map<string, string>();
  const backtraceFrames = new Map<string, readonly string[]>();
  const columns: string[] = [];
  const stack: Frame[] = [];

  let sawSchema = false;
  let inRow = false;
  let rowDepth = 0;
  let columnIndex = 0;
  let row: Record<string, unknown> = {};
  let text = "";
  let buffer = "";

  const assign = (value: string | null): void => {
    const name = columns[columnIndex];
    columnIndex += 1;
    if (name === undefined) return;
    if (wanted && !wanted.has(name)) return;
    row[name] = value;
  };

  const stream = typeof source === "string"
    ? createReadStream(source, { encoding: "utf8", highWaterMark: 1 << 21 })
    : source;
  for await (const chunk of stream as AsyncIterable<string>) {
    buffer += chunk;
    // Hold back a trailing partial tag so a split never corrupts a match.
    const lastOpen = buffer.lastIndexOf("<");
    const complete = lastOpen === -1 || buffer.indexOf(">", lastOpen) !== -1;
    const safeEnd = complete ? buffer.length : lastOpen;
    const work = buffer.slice(0, safeEnd);
    buffer = buffer.slice(safeEnd);

    TAG.lastIndex = 0;
    let match: RegExpExecArray | null;
    let cursor = 0;
    while ((match = TAG.exec(work)) !== null) {
      const [full, openName, rawAttributes, selfClose, closeName] = match;
      if (stack.length > 0) text += work.slice(cursor, match.index);
      cursor = match.index + full.length;

      if (closeName !== undefined) {
        const frame = stack.pop();
        const body = decodeEntities(text);
        text = "";
        if (frame === undefined) continue;
        if (frame.kind === "mnemonic") { columns.push(body); continue; }
        if (frame.kind === "row") {
          inRow = false;
          yield row as TraceRow;
          continue;
        }
        if (frame.id !== undefined) values.set(frame.id, { fmt: frame.fmt, text: body });
        if (frame.backtraceFrames !== undefined) {
          if (frame.id !== undefined) backtraceFrames.set(frame.id, frame.backtraceFrames);
          row.frames = frame.backtraceFrames;
        }
        if (frame.isColumn) assign(frame.fmt !== undefined ? frame.fmt : body);
        if (frame.counted) rowDepth -= 1;
        continue;
      }

      const attributes = parseAttributes(rawAttributes);

      if (openName === "schema" && !selfClose) {
        // Only the first schema block of a table export describes its columns.
        stack.push({ kind: sawSchema ? "other" : "schema" });
        sawSchema = true;
        text = "";
        continue;
      }
      if (openName === "mnemonic" && !selfClose && !inRow) {
        stack.push({ kind: "mnemonic" });
        text = "";
        continue;
      }
      if (openName === "row" && !selfClose) {
        stack.push({ kind: "row" });
        inRow = true;
        rowDepth = 0;
        columnIndex = 0;
        row = {};
        text = "";
        continue;
      }

      if (options.stacks && openName === "frame") {
        const name = attributes.name !== undefined ? decodeEntities(attributes.name)
          : (attributes.ref !== undefined ? frameNames.get(attributes.ref) : undefined);
        if (attributes.id !== undefined && name !== undefined) {
          frameNames.set(attributes.id, name);
        }
        if (inRow && name !== undefined) {
          for (let index = stack.length - 1; index >= 0; index -= 1) {
            const collector = stack[index]?.backtraceFrames;
            if (collector !== undefined) {
              collector.push(name);
              break;
            }
          }
        }
        if (!selfClose) {
          stack.push({ counted: inRow });
          if (inRow) rowDepth += 1;
          text = "";
        }
        continue;
      }

      const isColumn = inRow && rowDepth === 0;

      if (openName === "sentinel") {
        if (isColumn) assign(null);
        continue;
      }
      if (attributes.ref !== undefined) {
        if (isColumn) {
          const hit = values.get(attributes.ref);
          assign(hit === undefined ? null : (hit.fmt !== undefined ? hit.fmt : hit.text));
          if (options.stacks && openName === "tagged-backtrace") {
            const frames = backtraceFrames.get(attributes.ref);
            if (frames !== undefined) row.frames = frames;
          }
        }
        continue;
      }
      if (selfClose) {
        if (attributes.id !== undefined) {
          values.set(attributes.id, { fmt: attributes.fmt, text: "" });
        }
        if (isColumn) assign(attributes.fmt !== undefined ? attributes.fmt : null);
        continue;
      }

      stack.push({
        id: attributes.id, fmt: attributes.fmt, isColumn, counted: inRow,
        backtraceFrames: options.stacks && openName === "tagged-backtrace" ? [] : undefined,
      });
      if (inRow) rowDepth += 1;
      text = "";
    }
    if (stack.length > 0) text += work.slice(cursor);
  }
}

/** Read an NDJSON file produced by this module back into rows. */
export async function* readTraceRows(path: string): AsyncGenerator<TraceRow> {
  const lines = createInterface({ input: createReadStream(path, { encoding: "utf8" }) });
  for await (const line of lines) {
    if (line.length > 0) yield JSON.parse(line) as TraceRow;
  }
}

// ---- Value coercion --------------------------------------------------------
// Instruments formats every value for display. These recover the numbers.

const DURATION_SCALE: Record<string, number> = {
  ns: 1e-3, "µs": 1, us: 1, ms: 1e3, s: 1e6, min: 60e6,
};

/** Parse a formatted Instruments duration ("834.96 µs", "1.05 s") to µs. */
export const durationMicroseconds = (value: string | null | undefined): number => {
  if (!value) return 0;
  const match = /^([\d.,]+)\s*(ns|µs|us|ms|s|min)$/.exec(value.trim());
  if (!match) return 0;
  const scale = DURATION_SCALE[match[2]];
  return scale === undefined ? 0 : parseFloat(match[1].replace(/,/g, "")) * scale;
};

/** Parse a formatted Instruments timestamp ("00:01.266.605") to µs. */
export const timestampMicroseconds = (value: string | null | undefined): number => {
  if (!value) return 0;
  const match = /^(?:(\d+):)?(\d+):(\d+)\.(\d+)\.(\d+)$/.exec(value.trim())
    ?? /^(\d+):(\d+)\.(\d+)\.(\d+)$/.exec(value.trim());
  if (!match) return 0;
  const parts = match.slice(1).filter((part) => part !== undefined) as string[];
  const [minutes, seconds, milli, micro] = parts.length === 5
    ? [Number(parts[0]) * 60 + Number(parts[1]), Number(parts[2]), Number(parts[3]), Number(parts[4])]
    : [Number(parts[0]), Number(parts[1]), Number(parts[2]), Number(parts[3])];
  return (minutes * 60 + seconds) * 1e6 + milli * 1e3 + micro;
};

/** Parse a formatted integer ("612,305,231"). */
export const formattedInteger = (value: string | null | undefined): number => (value
  ? Number(value.replace(/[,\s]/g, "")) : 0);

// ---- CLI -------------------------------------------------------------------

const isMain = process.argv[1] !== undefined
  && import.meta.url === new URL(`file://${process.argv[1]}`).href;

if (isMain) {
  const path = process.argv[2];
  if (!path) {
    console.error("usage: xctrace-trace-tables.ts <exported.xml> [--stacks] [--columns=a,b]");
    process.exit(2);
  }
  const columnsFlag = process.argv.find((argument) => argument.startsWith("--columns="));
  const batch: string[] = [];
  for await (const row of parseTraceTable(path, {
    stacks: process.argv.includes("--stacks"),
    columns: columnsFlag?.slice("--columns=".length).split(","),
  })) {
    batch.push(JSON.stringify(row));
    if (batch.length >= 4096) {
      process.stdout.write(`${batch.join("\n")}\n`);
      batch.length = 0;
    }
  }
  if (batch.length > 0) process.stdout.write(`${batch.join("\n")}\n`);
}
