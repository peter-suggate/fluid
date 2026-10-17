import assert from "node:assert/strict";
import { createServer } from "node:http";
import { gzipSync } from "node:zlib";
import test from "node:test";
import { fetchMeshVolume } from "./mesh-volume-loader";

test("mesh HTTP loading accepts file gzip and automatically decoded HTTP gzip", async () => {
  const header = { version: 1, units: "metres", upAxis: "Y", asset: "transport-fixture",
    origin: [0, 0, 0], dimensions: [8, 8, 8], cellSize: .00625, bandWidth: .025,
    brickCount: 0, brickCells: 8, nodeStrideBytes: 16, materials: [] };
  const json = Buffer.from(JSON.stringify(header)), prefix = Buffer.alloc(8);
  prefix.write("FVX1"); prefix.writeUInt32LE(json.length, 4);
  const raw = Buffer.concat([prefix, json]), gzip = gzipSync(raw);
  const server = createServer((req, res) => {
    if (req.url === "/missing.fvx.gz") { res.writeHead(404); res.end(); return; }
    if (req.url === "/invalid.fvx.gz") { res.end("not a volume"); return; }
    res.setHeader("Content-Type", "application/octet-stream");
    if (req.url === "/http.fvx.gz") res.setHeader("Content-Encoding", "gzip");
    res.end(req.url === "/raw.fvx.gz" ? raw : gzip);
  });
  await new Promise<void>(resolve => server.listen(0, "127.0.0.1", resolve));
  const address = server.address(); assert.ok(address && typeof address !== "string");
  const root = `http://127.0.0.1:${address.port}`;
  try {
    for (const file of ["http", "file", "raw"]) {
      assert.deepEqual((await fetchMeshVolume(`${root}/${file}.fvx.gz`)).header, header);
    }
    await assert.rejects(fetchMeshVolume(`${root}/missing.fvx.gz`), /missing\.fvx\.gz: HTTP 404/);
    await assert.rejects(fetchMeshVolume(`${root}/invalid.fvx.gz`), /invalid\.fvx\.gz: Invalid mesh volume magic/);
  } finally {
    await new Promise<void>((resolve, reject) => server.close(error => error ? reject(error) : resolve()));
  }
});

test("mesh network failures identify the asset and preserve the cause", async () => {
  const cause = new TypeError("Failed to fetch");
  await assert.rejects(fetchMeshVolume("/assets/fence.fvx.gz", async () => { throw cause; }), error => {
    assert.ok(error instanceof Error);
    assert.equal(error.message, "Loading mesh asset /assets/fence.fvx.gz: Failed to fetch");
    assert.equal(error.cause, cause);
    return true;
  });
});
