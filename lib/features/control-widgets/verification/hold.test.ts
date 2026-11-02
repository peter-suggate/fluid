import assert from "node:assert/strict";
import test from "node:test";
import { controlHold } from "../hold";

function counted() {
  const focus = new EventTarget();
  let releases = 0;
  const hold = controlHold(focus);
  const grab = () => hold.grab(() => { releases++; });
  return { focus, hold: { grab, letGo: hold.letGo }, releases: () => releases };
}

test("a held control releases once however many ends arrive", () => {
  const { hold, releases } = counted();
  hold.grab();
  hold.letGo(); // pointerup
  hold.letGo(); // the lost capture that follows it
  hold.letGo(); // unmount
  assert.equal(releases(), 1);
});

test("losing window focus mid-press releases, and the listener goes with the press", () => {
  const { focus, hold, releases } = counted();
  focus.dispatchEvent(new Event("blur"));
  assert.equal(releases(), 0, "nothing held, nothing to release");
  hold.grab();
  focus.dispatchEvent(new Event("blur"));
  assert.equal(releases(), 1);
  hold.letGo();
  focus.dispatchEvent(new Event("blur"));
  assert.equal(releases(), 1, "the press already ended");
});

test("a second grab while held does not stack, and each press gets its own release", () => {
  const { focus, hold, releases } = counted();
  hold.grab();
  hold.grab();
  focus.dispatchEvent(new Event("blur"));
  assert.equal(releases(), 1);
  hold.grab();
  hold.letGo();
  assert.equal(releases(), 2);
});

test("an unmount with nothing held sends no release", () => {
  const { hold, releases } = counted();
  hold.letGo();
  assert.equal(releases(), 0);
});
