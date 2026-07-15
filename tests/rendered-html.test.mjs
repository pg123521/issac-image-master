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
  assert.match(html, /943(?:<!-- -->)? objects/);
  assert.match(html, /MobileCLIP2-S0/);
  assert.match(html, /上传截图/);
  assert.match(html, /Top K 相似对象/);
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
    access(new URL("../models/mobileclip-object-icon-index-v1.pt", import.meta.url)),
    access(new URL("../public/items/icons/item-070.png", import.meta.url)),
    access(new URL("../public/objects/icons/trinket-10147.png", import.meta.url)),
  ]);
});
