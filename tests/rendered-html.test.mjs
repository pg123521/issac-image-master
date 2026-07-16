import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the offline manual-selection lens", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  const html = await response.text();
  assert.match(html, /<title>Isaac Item Lens<\/title>/i);
  assert.match(html, /上传截图以识别物品/);
  assert.match(html, /图片和模型推理仅在本机处理/);
  assert.doesNotMatch(html, /自动检测|目标检测|127\.0\.0\.1/);
});

test("ships the browser model, index, PWA files, and complete object library", async () => {
  const metadata = JSON.parse(await readFile(new URL("../public/models/item-vectors.json", import.meta.url), "utf8"));
  assert.equal(metadata.rows, 1020);
  assert.equal(metadata.dimensions, 512);
  assert.ok(metadata.objects.some((object) => object.id === "trinket-10073"));
  await Promise.all([
    access(new URL("../public/models/mobileclip-image-encoder.onnx", import.meta.url)),
    access(new URL("../public/models/item-vectors.f16", import.meta.url)),
    access(new URL("../public/manifest.webmanifest", import.meta.url)),
    access(new URL("../public/sw.js", import.meta.url)),
    access(new URL("../public/items/icons/item-070.png", import.meta.url)),
  ]);
});

test("browser client uses local ONNX inference and contains no detector endpoints", async () => {
  const source = await readFile(new URL("../app/IsaacLens.tsx", import.meta.url), "utf8");
  assert.match(source, /onnxruntime-web/);
  assert.match(source, /InferenceSession\.create/);
  assert.match(source, /executionProviders: \["wasm"\]/);
  assert.match(source, /单指拖动 · 双指缩放 · 点击选取/);
  assert.match(source, /首次运行需要下载约 46MB/);
  assert.match(source, /fetchBinaryWithProgress/);
  assert.match(source, /XMLHttpRequest/);
  assert.match(source, /candidateDisplayLimit/);
  assert.match(source, /candidateDisplayLimitV2/);
  assert.match(source, /item-detail-dialog/);
  assert.doesNotMatch(source, /\/detect|predict-batch|127\.0\.0\.1/);
});
