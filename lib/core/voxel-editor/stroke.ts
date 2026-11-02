import type { EditorRay } from "../editor-entity";
import type { ToolUpdate } from "./plugin";

/**
 * Where a stroke is, one phase at a time:
 *
 *     dragging ──up──► releasing ──advance──► hovering ──press──► ending ──► concluded
 *        │                 └──────no further phase──────────────────┘
 *        └───────────── cancel, Escape, lost capture, blur ─────────┘
 *
 * - `releasing`: the pointer came up and the gesture has not yet said whether it continues.
 * - `hovering`: a later phase follows the bare pointer; the next press commits it.
 * - `ending`: no more samples are taken; the queue drains and the stroke concludes.
 * - `concluded`: handed to the edit queue or abandoned: the pointer is no longer this stroke's.
 *
 * Ending is reachable from every phase before `concluded`, and a release that
 * answers after the stroke was ended does not take it back to `hovering`.
 */
export type StrokePhase = "dragging" | "releasing" | "hovering" | "ending" | "concluded";

/** The tool transaction a stroke drives. See `beginToolTransaction`. */
export interface StrokeTransaction {
  update(sample: EditorRay): Promise<ToolUpdate | undefined>;
  advance(): Promise<boolean>;
  finish(cancelled: boolean): Promise<void>;
}

/** What a stroke asks of the viewport it is drawn in. */
export interface StrokeHost {
  /** Put this stroke's outline up; null takes it down. */
  outline(update: ToolUpdate | null): void;
  notice(error: unknown): void;
  /** Run once on the next frame; the result cancels it. */
  frame(run: () => void): () => void;
  /** The stroke has let go of the pointer; its edit is still being accepted. */
  released(): void;
  /** The edit has been answered, accepted or not. */
  settled(): void;
}

/**
 * One stroke of an armed voxel tool: the samples it takes, the release that
 * asks the gesture whether it continues, and the end that hands it to the edit
 * queue. Samples resolve within the turn, so a stroke that ends has let go of
 * the pointer before the next press can arrive.
 */
export class VoxelStroke {
  private current: StrokePhase = "dragging";
  private queued?: EditorRay;
  private abandoned = false;
  /** A pump is running; the next one waits for it rather than interleaving. */
  private busy = false;
  private cancelFrame?: () => void;

  constructor(
    readonly pointerId: number,
    private readonly transaction: StrokeTransaction,
    /** A tool that authors only on release: its outline is the proposal, and an end mid-drag discards it. */
    readonly releaseOnly: boolean,
    private readonly host: StrokeHost,
  ) {}

  get phase(): StrokePhase { return this.current; }
  get cancelled(): boolean { return this.abandoned; }

  /** The press: its first sample is taken at once rather than on the next frame. */
  begin(ray: EditorRay): void {
    this.queued = ray;
    void this.pump();
  }

  /** A pointer move. While hovering any pointer is the stroke's; otherwise only its own, until it ends. */
  sample(ray: EditorRay, pointerId: number): void {
    if (this.current === "hovering"
      || (pointerId === this.pointerId && (this.current === "dragging" || this.current === "releasing"))) {
      this.queued = ray;
      this.schedule();
    }
  }

  /** The pointer came up, or was taken away: a cancelled pointer ends the stroke; a released one first asks the gesture whether it continues. */
  release(ray: EditorRay, cancelled: boolean): void {
    if (this.current !== "dragging") return;
    this.abandoned ||= cancelled;
    this.queued = this.abandoned ? undefined : ray;
    this.current = this.abandoned ? "ending" : "releasing";
    void this.pump();
  }

  /** The primary press that ends a hovering phase commits what it shows. */
  commit(ray: EditorRay): void {
    if (this.current !== "hovering") return;
    this.queued = ray;
    this.current = "ending";
    void this.pump();
  }

  /**
   * End it from wherever it is. Cancelled, nothing more is authored and the
   * stroke rolls back; `discard` also drops a waiting sample without rolling
   * back, which is what a runtime that can no longer validate one needs.
   */
  end(cancelled: boolean, discard = cancelled): void {
    if (this.current === "concluded") return;
    this.abandoned ||= cancelled;
    this.current = "ending";
    if (discard) this.queued = undefined;
    void this.pump();
  }

  /**
   * The viewport went away mid-stroke. Nothing more is sampled, and the
   * transaction still finishes — serialisation waits for any outstanding
   * preflight even after the canvas is gone and frames stop firing.
   */
  abandon(): void {
    if (this.current === "concluded") return;
    this.cancelFrame?.();
    this.cancelFrame = undefined;
    this.queued = undefined;
    this.current = "concluded";
    void this.transaction.finish(this.abandoned || this.releaseOnly).catch(error => this.host.notice(error));
  }

  /** Free the pointer for the next press now, and let the edit be accepted behind it. */
  private async conclude(): Promise<void> {
    this.current = "concluded";
    this.cancelFrame?.();
    this.cancelFrame = undefined;
    this.host.released();
    try { await this.transaction.finish(this.abandoned); }
    catch (error) { this.host.notice(error); }
    this.host.settled();
  }

  /** Run what the stroke has waiting: its latest sample, then the release, then the end. */
  private async pump(): Promise<void> {
    if (this.busy || this.current === "concluded") return;
    this.busy = true;
    try {
      const sample = this.queued;
      this.queued = undefined;
      if (sample && !this.abandoned) {
        const result = await this.transaction.update(sample);
        if (!this.abandoned && (result || this.releaseOnly)) this.host.outline(result ?? null);
      }
      if (this.current === "releasing" && !this.queued) {
        const continues = await this.transaction.advance();
        // Escape or a lost runtime may have ended it while the gesture answered.
        if (this.phase === "releasing") this.current = continues ? "hovering" : "ending";
      }
    } catch (error) {
      if (this.releaseOnly) this.host.outline(null);
      // Author tools retain their last proposal. Live validation failures
      // roll the stroke back; a failed rollback keeps its accepted prefix undoable.
      this.host.notice(error);
    }
    this.busy = false;
    if (this.current === "ending" && !this.queued) await this.conclude();
    else if (this.current === "ending" || this.current === "releasing") await this.pump();
    else if (this.queued) this.schedule();
  }

  /** Samples of a drag are taken once a frame, the latest winning; a press, a release and an end are not made to wait for one. */
  private schedule(): void {
    if (this.cancelFrame || this.current === "concluded") return;
    this.cancelFrame = this.host.frame(() => { this.cancelFrame = undefined; void this.pump(); });
  }
}
