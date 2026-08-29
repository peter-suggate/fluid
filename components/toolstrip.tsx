"use client";

import {
  createContext,
  useContext,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { useAnchoredFlyout } from "./anchored-flyout";

/**
 * Which section of the strip currently has a row open.
 *
 * The strip is assembled from independent pieces — the field views, the solve's
 * own rows, the selected object's options, a sculpt's dials — and each used to
 * hold its own open row. Nothing stopped two of them being open at once, so the
 * tank's SETUP card and its CONTAINER card overlapped in the same corner, each
 * hanging off a different row of one column. One column, one thing open: the
 * section that opens something claims the strip, and the others close.
 *
 * A section token rather than a row id, so a section keeps saying *which* of
 * its rows is open in whatever shape suits it — a string, or the field rows'
 * tagged detail — and only has to hear that it is no longer the one.
 */
const ToolstripSectionContext = createContext<{
  readonly open: string | undefined;
  readonly claim: (section: string | undefined) => void;
}>({ open: undefined, claim: () => {} });

/**
 * Opening and closing for one section of the strip, against the shared claim.
 *
 * Returns whether this section still holds the strip; a section that has lost
 * it must close whatever it had open, which is what the effect at the call site
 * is for — `onLost` runs when another section takes over.
 */
export function useToolstripSection(section: string, onLost: () => void) {
  const { open, claim } = useContext(ToolstripSectionContext);
  const held = open === section;
  const lost = useRef(onLost);
  // Kept current in an effect rather than during render: the closure captures
  // this render's state setters, and the only thing that may read it is the
  // effect below, which runs after this one.
  useEffect(() => {
    lost.current = onLost;
  });
  useEffect(() => {
    if (!held) lost.current();
  }, [held]);
  return {
    /** Called when this section opens a row, or closes the one it had. */
    claim: (opened: boolean) => claim(opened ? section : undefined),
  };
}

/**
 * A column of controls hung off a corner of the thing they belong to.
 *
 * The editor's options used to live in cards: a chip you clicked to expand a
 * stacked panel of every setting the thing had. That is the wrong instrument
 * for a workbench whose subject is a moving image — a card is a permanent
 * rectangle over the water, and a panel that has to be opened first is a panel
 * whose contents nobody knows exist. This is the replacement, and the argument
 * for its shape is the same one every time:
 *
 * - **Hung off the subject's own corner**, so what the controls are about is
 *   said by where they are rather than by a title bar.
 * - **No panel behind them.** The rows are the chrome. A background would be a
 *   rectangle over the image to hold marks that are already legible on it, and
 *   the only thing that gets one is the row that is *on*.
 * - **Everything visible at rest.** The list is the disclosure; there is no
 *   expand step between the reader and the options.
 * - **One row opens at a time**, to the right — away from the scene. A control
 *   on the inboard side covers the water at exactly the moment the reader is
 *   adjusting how it looks.
 * - **The tip is the panel's own**, because the browser's arrives about a
 *   second late, and a strip of short marks whose only names are behind a delay
 *   is a strip a reader gives up on before it answers.
 *
 * Rows come in two flavours and both are here rather than in two components,
 * because they have to line up in one column: a **glyph** row for something
 * with an inherent picture (a field view is a picture of the water), and a
 * **tag** row for something without one (there is no honest pictogram for
 * viscosity, and a column of guessed ones is a hover-to-decode puzzle). A tag
 * row also carries its current value, so the column reads as a list of facts
 * before it is touched.
 */
export function Toolstrip({
  leftFraction,
  topFraction,
  ariaLabel,
  narrow,
  testId,
  children,
}: {
  leftFraction: number;
  topFraction: number;
  ariaLabel: string;
  /**
   * A column sized for one object rather than for the scene.
   *
   * The tag width is a single decision for the whole column — every row shares
   * it so the values line up — so it is set here rather than per row. An
   * object's strip hangs off the object, often near the middle of the frame
   * with water on both sides, and its rows are short: a tag and a number.
   * The container's stands at the edge of the image carrying sentences like
   * "Re-author at", and narrowing it would only ellipsize them.
   */
  narrow?: boolean;
  testId?: string;
  children: ReactNode;
}) {
  // Anchored against the shell's measured box, so orbiting until this corner
  // nears an edge slides the column rather than clipping it. `originY: 0` hangs
  // its top edge off the corner, which is the corner it is about.
  const { ref, style } = useAnchoredFlyout<HTMLDivElement>({ leftFraction, topFraction, originY: 0 });
  const [open, setOpen] = useState<string | undefined>(undefined);
  const section = useMemo(() => ({ open, claim: setOpen }), [open]);
  return <ToolstripSectionContext.Provider value={section}><div
    ref={ref}
    className={`toolstrip${narrow ? " is-narrow" : ""}`}
    data-testid={testId}
    style={style}
    role="group"
    aria-label={ariaLabel}
  >{children}</div></ToolstripSectionContext.Provider>;
}

/** What the column is about, when the subject's own outline does not say it. */
export function ToolstripTitle({ children }: { children: ReactNode }) {
  return <div className="toolstrip-title">{children}</div>;
}

/**
 * A hairline between two kinds of row.
 *
 * Used where one column carries two subjects — the water's views above, the
 * tank's own settings below — since without it the reader has to infer the
 * seam from the tags.
 */
export function ToolstripRule() {
  return <div className="toolstrip-rule" role="separator" />;
}

/**
 * One option: the mark that identifies it, and whatever it expands into.
 *
 * `children` is the row's open state and nothing else — a segmented strip, a
 * scrub, or a whole pane for a list. It is rendered to the right of the mark,
 * so the control sits beside the thing it belongs to rather than at the foot of
 * a column of things it does not, and the row's tip measures itself against the
 * whole row so it opens clear of that control instead of over it.
 */
export function ToolstripRow({
  icon,
  tag,
  value,
  name,
  hint,
  active,
  disabled,
  testId,
  onClick,
  children,
}: {
  /** The glyph, for a row whose subject is a picture. Mutually exclusive with `tag`. */
  icon?: ReactNode;
  /** The short name, for a row whose subject is not a picture. */
  tag?: string;
  /** What the option currently is, beside its tag. */
  value?: ReactNode;
  /** The full name, on the tip. */
  name: string;
  /** One line under the name saying what it does or costs. */
  hint?: string;
  active?: boolean;
  disabled?: boolean;
  testId?: string;
  onClick?: () => void;
  children?: ReactNode;
}) {
  const className = `toolstrip-key${icon ? " is-glyph" : ""}${active ? " active" : ""}`;
  // A tip that repeats the tag and adds nothing is worse than no tip: it is a
  // box that opens over the scene to tell the reader what they can already
  // read. So a row whose tag is already its whole name, with nothing further to
  // say, simply does not have one.
  const tip = hint !== undefined || name !== tag;
  const body = <>
    {icon}
    {tag !== undefined && <b className="toolstrip-tag">{tag}</b>}
    {value !== undefined && <span className="toolstrip-value">{value}</span>}
    {tip && <span className="toolstrip-tip">
      <strong>{name}</strong>
      {hint !== undefined && <small>{hint}</small>}
    </span>}
  </>;
  // An open row is one whose control is on screen, and its tip has to stand down
  // for it: the tip opens into exactly the space the control occupies, and a
  // label painted over the slider it names is worse than no label at all.
  return <div className={`toolstrip-row${children ? " is-open" : ""}`}>
    {/* A row whose control is always shown has nothing left for its key to do,
        and a button that does nothing is worse than a label: it invites the
        click it will not answer. Such a row keeps the key's anatomy — the tag,
        the value, the tip — as a label, and the control beside it is the whole
        interaction. */}
    {onClick === undefined
      ? <span className={`${className} is-static`} data-testid={testId}>{body}</span>
      : <button
        type="button"
        className={className}
        aria-pressed={active}
        disabled={disabled}
        data-testid={testId}
        onClick={onClick}
      >{body}</button>}
    {children}
  </div>;
}

/**
 * A row that *does* something rather than reporting what something is set to.
 *
 * The column's two authored flavours are both statements — a glyph row is a
 * picture of a view, a tag row is a name and its current value — and neither
 * fits a verb. A verb has no value to show and no open state to hold; it has a
 * name, a picture, and a consequence. Written as a tag row it would be a
 * setting with a permanently blank answer; written as a glyph row it would be a
 * lone pictogram whose only label arrives on hover, which is the wrong bargain
 * for something irreversible. So: mark *and* name, on one full-width row, and
 * `tone` so the destructive one can say so in the same red the ring's wedge
 * uses — one verb, one colour, wherever the reader meets it.
 */
export function ToolstripActionRow({
  icon,
  label,
  name,
  hint,
  tone,
  disabled,
  testId,
  onClick,
}: {
  /** The verb's picture, drawn from the editor's one icon vocabulary. */
  icon: ReactNode;
  /** The verb, in the column's tag style. */
  label: string;
  /** The full phrase — the verb and what it acts on — on the tip. */
  name: string;
  /** One line saying what it does, and how it can be undone. */
  hint?: string;
  /** Binds `--tone`; `danger` is the red every destructive mark shares. */
  tone?: "danger";
  disabled?: boolean;
  testId?: string;
  onClick: () => void;
}) {
  return <div className="toolstrip-row is-verb">
    <button
      type="button"
      className={`toolstrip-key is-verb${tone ? ` tone-${tone}` : ""}`}
      disabled={disabled}
      data-testid={testId}
      onClick={onClick}
    >
      {icon}
      <b className="toolstrip-tag">{label}</b>
      <span className="toolstrip-tip">
        <strong>{name}</strong>
        {hint !== undefined && <small>{hint}</small>}
      </span>
    </button>
  </div>;
}

/**
 * The row that is only a way out: everything the column left out.
 *
 * Rendered as a row so it lands in the same column at the same left edge, and
 * marked so it can be the one thing in the strip drawn at less than full
 * strength — it is not an option, it is the door.
 */
export function ToolstripMoreRow({
  name,
  hint,
  active,
  testId,
  onClick,
  children,
}: {
  name: string;
  hint?: string;
  active?: boolean;
  testId?: string;
  onClick: () => void;
  children?: ReactNode;
}) {
  return <div className={`toolstrip-row is-more${children ? " is-open" : ""}`}>
    <button
      type="button"
      className={`toolstrip-key is-glyph${active ? " active" : ""}`}
      aria-pressed={active}
      data-testid={testId}
      onClick={onClick}
    >
      <i aria-hidden="true">⋯</i>
      <span className="toolstrip-tip">
        <strong>{name}</strong>
        {hint !== undefined && <small>{hint}</small>}
      </span>
    </button>
    {children}
  </div>;
}

/**
 * Where a pane has to go to stay on screen.
 *
 * A pane is the one part of the strip that is out of flow, and so the one part
 * the strip's own anchoring cannot see. Two corrections, both measured here so
 * every kind of pane gets them:
 *
 * `lift` is how far above its row it sits. A pane hangs off a row of a column
 * that can be twenty rows long, so the row that opens it is often near the foot
 * of the window while the card is 280px tall; rendered where it is authored,
 * its controls would be below the fold. Lifting beats capping — there is nearly
 * always room above, and a card that scrolls internally when the screen could
 * have shown all of it is the worse answer. It is applied as a transform
 * deliberately: a transform takes no part in layout, so measuring the pane
 * cannot move it and the observer cannot chase itself.
 *
 * `side` is the flip. Right is the authored side for the same reason the rest
 * of the strip opens right — away from the scene — and it gives way only when
 * the strip is near the window's right edge and the left genuinely has room.
 */
function usePaneFit(content?: string) {
  const ref = useRef<HTMLDivElement>(null);
  const [placement, setPlacement] = useState<{
    lift: number;
    side: "left" | "right";
    offset: number | undefined;
  }>({ lift: 0, side: "right", offset: undefined });
  useLayoutEffect(() => {
    const pane = ref.current;
    if (pane === null) return;
    const fit = () => {
      const row = pane.offsetParent;
      if (!(row instanceof HTMLElement)) return;
      const margin = 12;
      const gap = 6;
      const box = row.getBoundingClientRect();
      // Cleared past the whole column, not past the row that opened it. A pane
      // measured from its own row starts 24px out when a glyph row opens it and
      // lands on top of the 158px rows below — and even a tag row's pane will
      // sit over the widest open control in the column. The strip's right edge
      // is the one edge every pane has to clear.
      const strip = row.closest(".toolstrip");
      const column = strip instanceof HTMLElement ? strip.getBoundingClientRect() : box;
      const width = pane.offsetWidth;
      const highest = Math.max(margin, window.innerHeight - margin - pane.offsetHeight);
      const flipped = column.right + gap + width > window.innerWidth - margin
        && column.left - gap - width >= margin;
      setPlacement({
        lift: Math.max(0, box.top - highest),
        side: flipped ? "left" : "right",
        // Expressed against the row, since that is what the pane is positioned
        // in; `left` and `right` are the same distance read from either end.
        offset: flipped ? box.right - column.left + gap : column.right + gap - box.left,
      });
    };
    fit();
    // The pane's own size is what decides both, and it changes when the thing
    // inside it does — a tab switched, a list filtered, a details block opened.
    const observer = new ResizeObserver(fit);
    observer.observe(pane);
    window.addEventListener("resize", fit);
    return () => {
      observer.disconnect();
      window.removeEventListener("resize", fit);
    };
    // `content` re-measures on a swap the observer does not report: replacing a
    // 300px face with a 430px one resizes the pane, but the notification for it
    // arrives in the frame the transform is already written from the old height
    // — so the tall face hangs off the bottom of the window until something else
    // moves. Naming what is inside makes the swap re-run the measurement.
  }, [content]);
  return {
    ref,
    side: placement.side,
    style: {
      ...(placement.lift === 0 ? {} : { transform: `translateY(${-placement.lift}px)` }),
      ...(placement.offset === undefined ? {}
        : placement.side === "left" ? { right: placement.offset }
          : { left: placement.offset }),
    },
  };
}

function paneClassName(side: "left" | "right", extra?: string) {
  return `toolstrip-pane${side === "left" ? " is-left" : ""}${extra ? ` ${extra}` : ""}`;
}

/**
 * A row's open state, when what it opens is more than a control.
 *
 * The card the strip refuses to be, granted to the one row that is currently
 * asking for it: a list of a dozen views, a solve's convergence curve, a
 * cluster of an object's settings. It is bordered because it is content rather
 * than chrome, and it is inside the row so it hangs off the thing that opened
 * it and disappears with it.
 */
export function ToolstripPane({
  label,
  onClose,
  children,
}: {
  label: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const { ref, side, style } = usePaneFit();
  return <div ref={ref} className={paneClassName(side)} aria-label={label} style={style}>
    <div className="toolstrip-pane-head">
      <h4>{label}</h4>
      <button type="button" title="Close" aria-label="Close" onClick={onClose}>×</button>
    </div>
    {children}
  </div>;
}

/** One face of a tabbed pane. `content` is built only when the tab is shown. */
export interface ToolstripTab {
  readonly id: string;
  readonly label: string;
  readonly content: ReactNode;
}

/**
 * The strip's one full panel: several subjects behind a row of names.
 *
 * Everything else on the strip is a row that opens exactly one thing, which is
 * right for a setting and wrong for the door at the foot of the column. What is
 * behind that door is not one more setting — it is the solver's construction,
 * the scene's own switches, and the verbs that act on the selected object:
 * unrelated subjects, each a screenful, none of which can be read as a row.
 * Stacked in one scrolling list they were a drawer nobody could find anything
 * in, so they are tabs, with the most-visited one first.
 *
 * It carries no card. A pane a row opens in passing is content and gets a
 * border; this is where the reader works, it is the largest thing on the
 * screen after the water, and a 384px frosted rectangle standing over the scene
 * is the panel this whole strip was built to replace. So it is the strip's own
 * material at the strip's own scale: the tabs are its head, the controls carry
 * their own small plates the way the strip's chips do, and the image shows
 * between them. The row that opened it is lit and closes it, like every other
 * row in the column — there is nothing between the click and the settings.
 */
export function ToolstripTabbedPane({
  label,
  tabs,
}: {
  /** The panel as a whole, for assistive technology. The tabs name themselves. */
  label: string;
  tabs: readonly ToolstripTab[];
}) {
  const [active, setActive] = useState<string | undefined>(undefined);
  const shown = tabs.find((tab) => tab.id === active) ?? tabs[0];
  const { ref, side, style } = usePaneFit(shown?.id);
  if (shown === undefined) return null;
  return <div ref={ref} className={paneClassName(side, "is-panel")} aria-label={label} style={style}>
    <div className="toolstrip-tabs" role="tablist" aria-label={label}>
      {tabs.map((tab) => <button
        key={tab.id}
        type="button"
        role="tab"
        className={tab.id === shown.id ? "active" : ""}
        aria-selected={tab.id === shown.id}
        data-testid={`toolstrip-tab-${tab.id}`}
        onClick={() => setActive(tab.id)}
      >{tab.label}</button>)}
    </div>
    {/* Keyed by tab, so a panel switched to another subject starts that subject
        fresh rather than inheriting the last one's scroll and open details. */}
    <div className="toolstrip-tabpanel" role="tabpanel" key={shown.id}>{shown.content}</div>
  </div>;
}

/**
 * A scrub sized for a strip row.
 *
 * Written here rather than reached for from `controls` because the studio's
 * `RangeControl` is a form field with a label above it and a 32px floor under
 * it; this is a 12px line beside a mark. The value is shown as text rather than
 * as a second output row, since the row it lives in is one line tall.
 */
export function ToolstripScrub({
  min,
  max,
  step,
  value,
  ariaLabel,
  readout,
  onChange,
  onCommit,
}: {
  min: number;
  max: number;
  step: number;
  value: number;
  ariaLabel: string;
  readout?: string;
  onChange: (value: number) => void;
  onCommit?: (value: number) => void;
}) {
  return <label className="toolstrip-scrub">
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value}
      aria-label={ariaLabel}
      onChange={(event) => onChange(Number(event.currentTarget.value))}
      // Release, key-up and blur all close the gesture: a drag that ended
      // because the row was deselected mid-pointer must still commit, or the
      // next edit's undo snapshot would be this one's.
      onPointerUp={onCommit && ((event) => onCommit(Number(event.currentTarget.value)))}
      onKeyUp={onCommit && ((event) => onCommit(Number(event.currentTarget.value)))}
      onBlur={onCommit && ((event) => onCommit(Number(event.currentTarget.value)))}
    />
    {readout !== undefined && <output>{readout}</output>}
  </label>;
}

/**
 * Exact entry for a quantity with no two ends to slide between.
 *
 * A position or an extent has no authored minimum and maximum, and a scrub
 * without both is a control that cannot say where it is. Those get a number
 * instead — committed on Enter or on leaving the field rather than per
 * keystroke, because every commit here is a history entry and a re-seed.
 */
export function ToolstripNumber({
  value,
  step,
  min,
  max,
  unit,
  ariaLabel,
  onCommit,
}: {
  value: number;
  step: number;
  min?: number;
  max?: number;
  unit?: string;
  ariaLabel: string;
  onCommit: (value: number) => void;
}) {
  const commit = (raw: string) => {
    const next = Number(raw);
    if (Number.isFinite(next) && next !== value) onCommit(next);
  };
  return <label className="toolstrip-number">
    <input
      type="number"
      // Keyed off the value so a commit that the document clamped or refused
      // shows the document's answer rather than the reader's typing.
      key={value}
      defaultValue={value}
      step={step}
      min={min}
      max={max}
      aria-label={ariaLabel}
      onBlur={(event) => commit(event.currentTarget.value)}
      onKeyDown={(event) => {
        if (event.key === "Enter") commit(event.currentTarget.value);
      }}
    />
    {unit !== undefined && <span>{unit}</span>}
  </label>;
}

/** A row's enumeration, small enough to live on one line beside its tag. */
export function ToolstripChoice({
  ariaLabel,
  value,
  options,
  onChange,
}: {
  ariaLabel: string;
  value: string;
  options: readonly {
    readonly value: string;
    readonly label: string;
    readonly title?: string;
    readonly disabled?: boolean;
  }[];
  onChange: (value: string) => void;
}) {
  return <div className="toolstrip-choice" role="group" aria-label={ariaLabel}>
    {options.map((option) => <button
      key={option.value}
      type="button"
      className={option.value === value ? "active" : ""}
      aria-pressed={option.value === value}
      disabled={option.disabled}
      title={option.title}
      onClick={() => { if (option.value !== value) onChange(option.value); }}
    >{option.label}</button>)}
  </div>;
}
