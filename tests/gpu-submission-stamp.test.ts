import test from 'node:test';
import assert from 'node:assert/strict';
import { GPUSubmissionStamp, GPU_SUBMISSION_STAMP_LIMIT } from '../lib/core/gpu-submission-stamp';

test('submission stamps are nonzero, fresh per submission, and refuse a stale or zero readback', () => {
  const stamp = new GPUSubmissionStamp('Probe exchange'), a = stamp.next(), b = stamp.next();
  assert.equal(a, 1); assert.equal(b, 2);
  stamp.check(b, b);
  assert.throws(() => stamp.check(a, b), /^Error: Probe exchange: stale GPU readback \(stamp 1, expected 2\)$/);
  assert.throws(() => stamp.check(0, b), /stale GPU readback \(stamp 0, expected 2\)/, 'an aborted submission reads back the cleared zero');
  assert.throws(() => stamp.check(0, 0), /invalid expected GPU submission stamp 0/);
  const wrap = new GPUSubmissionStamp('Wrap');
  (wrap as unknown as { issued: number }).issued = GPU_SUBMISSION_STAMP_LIMIT - 1;
  assert.equal(wrap.next(), GPU_SUBMISSION_STAMP_LIMIT); assert.equal(wrap.next(), 1, 'the stamp skips zero when it wraps');
});
