"use client";

import { useRef, useState, type ChangeEvent, type KeyboardEvent } from "react";
import { clampNumber, printNumber } from "./number";

/**
 * The two edit gestures every control in the app is built from.
 *
 * A drag and a typed entry each have one right answer to "when does the value
 * land", and the app had four: the studio's range committed on release, the
 * pipeline's did too but on any key-up, the toolstrip's reported every
 * pointer-move and committed on release *or* blur, and the studio's number
 * field committed every keystroke that parsed. Each value a control commits can
 * rebuild a solver, re-seed a scene or take an undo snapshot, so the answer is
 * made once, here.
 */

/** Keys that move a range input and so end a keyboard gesture on release. */
const STEP_KEYS = new Set(["ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown"]);

/**
 * A drag that commits once, on release.
 *
 * The track follows the thumb from a local draft; the owner hears `onChange`
 * once per gesture — pointer release, the key-up that ends a keyboard step, or
 * the blur of a drag interrupted by the row closing under it (which must still
 * commit, or the next edit's undo snapshot would be this one's).
 *
 * `onInput` is for an owner that previews per pointer-move — a scene editor
 * redrawing a gizmo — and still wants the single commit at the end. The commit
 * then compares against the value the gesture *started* from, because the
 * owner's value has been following the thumb all along.
 *
 * `onGestureEnd` hears the end of every gesture that moved the thumb, changed
 * or not — for an owner that opened something on the first `onInput` (an undo
 * bracket around live patches) and must close it however the gesture ends,
 * including a drag that came back to where it started and so commits nothing.
 */
export function useSliderGesture({ value, min, max, onChange, onInput, onGestureEnd }: {
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
  onInput?: (value: number) => void;
  onGestureEnd?: () => void;
}) {
  const [draft, setDraft] = useState<{ origin: number; base: number; value: number } | null>(null);
  const shown = draft && (draft.base === value || onInput) ? draft.value : value;
  const commit = (raw: number) => {
    if (!draft) return;
    const next = clampNumber(raw, min, max);
    setDraft(null);
    if (next !== draft.origin) onChange(next);
    onGestureEnd?.();
  };
  return {
    shown,
    dragging: draft !== null,
    inputProps: {
      value: shown,
      onChange: (event: ChangeEvent<HTMLInputElement>) => {
        const next = Number(event.currentTarget.value);
        setDraft((current) => ({ origin: current?.origin ?? value, base: value, value: next }));
        onInput?.(next);
      },
      onPointerUp: (event: { currentTarget: HTMLInputElement }) => commit(Number(event.currentTarget.value)),
      // A cancelled pointer abandons a commit-on-release drag. A previewing one
      // has already moved its owner, so it lands where it stopped instead.
      onPointerCancel: (event: { currentTarget: HTMLInputElement }) => {
        if (onInput) commit(Number(event.currentTarget.value));
        else if (draft) { setDraft(null); onGestureEnd?.(); }
      },
      onBlur: (event: { currentTarget: HTMLInputElement }) => commit(Number(event.currentTarget.value)),
      onKeyUp: (event: KeyboardEvent<HTMLInputElement>) => {
        if (STEP_KEYS.has(event.key)) commit(Number(event.currentTarget.value));
      },
    },
  };
}

/**
 * A typed number that commits on Enter or on leaving the field.
 *
 * The raw text is held while editing: "-", "0." and "" are not finite numbers,
 * and snapping a controlled value back on every keystroke swallows the leading
 * minus sign. Escape abandons the edit. The commit compares against the value
 * as *printed*, so a field entered and left untouched never reads its own
 * rounding back as an edit — these commits are often structural.
 */
export function useNumberEntry({ value, step, digits, min, max, onCommit }: {
  value: number;
  step?: number;
  digits?: number;
  min?: number;
  max?: number;
  onCommit: (value: number) => void;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  // Enter and Escape both end the edit by leaving the field, so the blur is the
  // one place an edit lands; this says the blur is an abandonment.
  const abandoning = useRef(false);
  const printed = printNumber(value, { step, digits });
  const commit = (raw: string) => {
    setDraft(null);
    const typed = Number(raw);
    if (raw.trim() === "" || !Number.isFinite(typed)) return;
    const next = clampNumber(typed, min, max);
    if (next !== Number(printed)) onCommit(next);
  };
  return {
    editing: draft !== null,
    inputProps: {
      type: "number" as const,
      value: draft ?? printed,
      step, min, max,
      onChange: (event: ChangeEvent<HTMLInputElement>) => setDraft(event.currentTarget.value),
      onBlur: (event: { currentTarget: HTMLInputElement }) => {
        if (abandoning.current) { abandoning.current = false; setDraft(null); return; }
        if (draft !== null) commit(event.currentTarget.value);
      },
      onKeyDown: (event: KeyboardEvent<HTMLInputElement>) => {
        if (event.key === "Enter") event.currentTarget.blur();
        else if (event.key === "Escape") { abandoning.current = true; event.currentTarget.blur(); }
      },
    },
  };
}
