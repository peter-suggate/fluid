"use client";

import type { ReactNode } from "react";

/**
 * What a rail key draws: its glyph, its digit in the corner where it has one,
 * and its name.
 *
 * The name stands outside the rail's card, over the scene — right of the
 * sculpt rail, left of the view rail — so the card stays a column of marks and
 * the names read as captions to it, haloed against whatever the scene puts
 * behind them. A reader learns the rail by reading it rather than by hovering
 * each mark in turn; the tip, and the line a lit key opens, take the name's
 * place while they stand.
 */
export function RailKeyFace({ glyph, label, shortcut }: { glyph: ReactNode; label: string; shortcut?: string }) {
  return <>
    {glyph}
    {shortcut && <kbd aria-hidden>{shortcut}</kbd>}
    <span className="tool-rail-label" aria-hidden>{label}</span>
  </>;
}
