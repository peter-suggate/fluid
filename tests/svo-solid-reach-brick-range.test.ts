import assert from "node:assert/strict";
import test from "node:test";
import { svoSolidReachBrickRange } from "../lib/svo/webgpu-svo-brick-selection";

test("GPU solid broad-phase ranges match every CPU brick overlap, including exact boundaries", () => {
  for(const origin of [-3.2,0,.1]) for(const edge of [.05,.2,1]) for(const margin of [0,.04330127018922193,.1]) {
    for(const minimum of [-.3,0,.1,.10000000000000002,.9]) {
      const maximum=minimum+.125;
      const [first,last]=svoSolidReachBrickRange(minimum,maximum,origin,edge,margin);
      for(let coordinate=first-2;coordinate<=last+2;coordinate++) {
        const lo=origin+coordinate*edge,hi=lo+edge;
        const expected=!(maximum<lo-margin||minimum>hi+margin);
        assert.equal(coordinate>=first&&coordinate<=last,expected,
          `broad-phase support at ${origin}/${edge}/${margin}/${minimum}/${coordinate}`);
      }
    }
  }
});
