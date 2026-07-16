import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html" },
    }),
    {
      ASSETS: {
        fetch: async () => new Response("Not found", { status: 404 }),
      },
    },
    {
      waitUntil() {},
      passThroughOnException() {},
    },
  );
}

test("server-renders Isaac Item Lens", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Isaac Item Lens<\/title>/i);
  assert.match(html, /离线道具识别/);
  assert.match(html, /上传截图/);
  assert.match(html, /empty-upload-button/);
  assert.match(html, /手动选取/);
  assert.doesNotMatch(html, /自动标注/);
  assert.match(html, /自动检测房间道具/);
  assert.doesNotMatch(html, /选中区域/);
  assert.doesNotMatch(html, /相似道具/);
  assert.doesNotMatch(html, /候选区域/);
  assert.doesNotMatch(html, /943(?:<!-- -->)? objects/);
  assert.doesNotMatch(html, /MobileCLIP2-S0/);
  assert.doesNotMatch(html, /置信度/);
  assert.match(html, /aria-label="上传后的游戏截图"/);
});

test("ships the complete offline object library and active index", async () => {
  const objects = JSON.parse(
    await readFile(new URL("../data/objects.en.json", import.meta.url), "utf8"),
  );
  const counts = Object.groupBy(objects, (object) => object.kind);

  assert.equal(objects.length, 943);
  assert.equal(counts.item.length, 717);
  assert.equal(counts.trinket.length, 121);
  assert.equal(counts.card.length, 105);
  assert.ok(objects.some((object) => object.id === "item-070" && object.nameEn === "Growth Hormones"));

  await Promise.all([
    access(new URL("../models/mobileclip-partial-v1.pt", import.meta.url)),
    access(new URL("../models/mobileclip-object-partial-index-v1.pt", import.meta.url)),
    access(new URL("../public/items/icons/item-070.png", import.meta.url)),
    access(new URL("../public/objects/icons/trinket-10147.png", import.meta.url)),
  ]);
});

test("includes user-facing detection progress and removable annotations", async () => {
  const source = await readFile(new URL("../app/IsaacLens.tsx", import.meta.url), "utf8");

  assert.match(source, /role="progressbar"/);
  assert.match(source, /正在查找道具/);
  assert.match(source, /手动选取/);
  assert.match(source, /未自动检测到道具，请手动选取/);
  assert.match(source, /查看自动检测结果/);
  assert.match(source, /item-detail-dialog/);
  assert.match(source, /删除道具详情窗口/);
  assert.match(source, /检测框大小：/);
  assert.match(source, /换一张图/);
  assert.match(source, /点击图片中物品进行手动选取/);
  assert.match(source, /selectedRegion && <aside/);
  assert.match(source, /选中区域/);
  assert.match(source, /相似道具/);
  assert.match(source, /超级小/);
  assert.match(source, /超级大/);
  assert.doesNotMatch(source, /manualBoxSize\}px/);
  assert.match(source, /deleteButtonGeometry/);
  assert.match(source, /isDeleteButtonHit/);
  assert.match(source, /删除 \$\{region\.label\}/);
  assert.doesNotMatch(source, />[^<]*置信度[^<]*</);
});
