import assert from "node:assert/strict";
import test from "node:test";
import { ClothStagePasses, clothStageOf } from "./stage-timing";

test("cloth pipelines map to their stages by label", () => {
  assert.equal(clothStageOf("Cloth beginStep"), "setup");
  assert.equal(clothStageOf("ogc frameBegin"), "setup");
  assert.equal(clothStageOf("Cloth nwGradient"), "assemble");
  assert.equal(clothStageOf("Cloth nwLinBuild"), "assemble");
  assert.equal(clothStageOf("Cloth pcgStep"), "pcg");
  assert.equal(clothStageOf("Cloth pcgSolve"), "pcg");
  assert.equal(clothStageOf("ogc filterPlanes"), "filter");
  assert.equal(clothStageOf("ogc filter"), "filter");
  assert.equal(clothStageOf("ogc verify"), "detect");
  assert.equal(clothStageOf("ogc decide"), "detect");
  assert.equal(clothStageOf("Cloth nwAccept"), "line-search");
  assert.equal(clothStageOf("Cloth nwApply"), "line-search");
  assert.equal(clothStageOf("Cloth finishStep"), "finish");
  assert.equal(clothStageOf("ogc contactState"), "finish");
  assert.equal(clothStageOf("Cloth somethingNew"), "other");
  assert.equal(clothStageOf("Water pressure"), "other");
});

test("a timed cloth pass opens one pass per stage change and keeps its bind groups", () => {
  const log: string[] = [];
  const pass = (label: string) => ({
    setPipeline: (p: { label: string }) => log.push(`${label} pipeline ${p.label}`),
    setBindGroup: (i: number, g: { label: string }, offsets: ArrayLike<number>) => log.push(`${label} group ${i} ${g.label} [${Array.from(offsets).join(",")}]`),
    dispatchWorkgroups: (x: number) => log.push(`${label} dispatch ${x}`),
    end: () => log.push(`${label} end`),
  });
  const encoder = { beginComputePass: (d: { label: string }) => { log.push(`begin ${d.label}`); return pass(d.label); } } as unknown as GPUCommandEncoder;
  const split = new ClothStagePasses(encoder).pass;
  const pipeline = (label: string) => ({ label }) as GPUComputePipeline, group = { label: "g" } as unknown as GPUBindGroup;
  split.setPipeline(pipeline("Cloth nwFaces")); split.setBindGroup(0, group, [256]); split.dispatchWorkgroups(4);
  split.setPipeline(pipeline("Cloth nwGradient")); split.dispatchWorkgroups(2);
  split.setPipeline(pipeline("Cloth pcgStep")); split.dispatchWorkgroups(1);
  split.end();
  assert.deepEqual(log, [
    "begin cloth:assemble", "cloth:assemble pipeline Cloth nwFaces", "cloth:assemble group 0 g [256]", "cloth:assemble dispatch 4",
    "cloth:assemble pipeline Cloth nwGradient", "cloth:assemble dispatch 2",
    "cloth:assemble end", "begin cloth:pcg", "cloth:pcg group 0 g [256]", "cloth:pcg pipeline Cloth pcgStep", "cloth:pcg dispatch 1",
    "cloth:pcg end",
  ]);
});
