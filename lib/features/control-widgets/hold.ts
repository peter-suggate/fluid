/**
 * A held control's one release: the pointer coming up, a cancel, a lost
 * capture, the window losing focus or the control unmounting mid-press,
 * whichever comes first. The press's `release` runs once however many of those
 * arrive, and the focus listener lives only while something is held.
 */
export function controlHold(focus: Pick<EventTarget, "addEventListener" | "removeEventListener">):
  { readonly grab: (release: () => void) => void; readonly letGo: () => void } {
  let held: (() => void) | undefined;
  const letGo = () => {
    const release = held;
    if (!release) return;
    held = undefined;
    focus.removeEventListener("blur", letGo);
    release();
  };
  const grab = (release: () => void) => {
    if (held) return;
    held = release;
    focus.addEventListener("blur", letGo);
  };
  return { grab, letGo };
}
