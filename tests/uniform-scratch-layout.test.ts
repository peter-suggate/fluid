import assert from "node:assert/strict";
import test from "node:test";
import { UniformScratchLayout } from "../lib/methods/uniform/uniform-scratch-arena";

test("scratch planning rejects malformed dimensions and byte ranges before allocation", () => {
  assert.throws(()=>new UniformScratchLayout([0,4,4],0),RangeError);
  assert.throws(()=>new UniformScratchLayout([4,1.5,4],0),RangeError);
  assert.throws(()=>new UniformScratchLayout([4,4,4],-4),RangeError);
  assert.throws(()=>new UniformScratchLayout([4,4,4],3),RangeError);
});
