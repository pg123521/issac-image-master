"use client";

import { ChangeEvent, MouseEvent, useEffect, useMemo, useRef, useState } from "react";
import itemsData from "../data/items.zh-CN.json";
import objectsData from "../data/objects.en.json";

type IsaacObject = {
  id: string;
  gameId: number;
  kind: "item" | "trinket" | "card" | string;
  nameZh: string;
  nameEn: string;
  pickup: string;
  effects: string[];
  type: string;
  pools: string[];
  tags: string[];
  description: string;
  iconPath: string;
  sourceName: string;
  sourceUrl: string;
  iconFeature?: {
    r: number;
    g: number;
    b: number;
    aspect: number;
    hash: string;
    descriptor: number[];
  };
};

type Feature = {
  r: number;
  g: number;
  b: number;
  edge: number;
  aspect: number;
};

type Region = {
  id: string;
  label: string;
  x: number;
  y: number;
  w: number;
  h: number;
  score: number;
  feature: Feature;
  hash: string;
  descriptor: number[];
  imageUrl: string;
  modelImageUrl: string;
  modelBox: {
    x: number;
    y: number;
    w: number;
    h: number;
  };
  matchBox: {
    x: number;
    y: number;
    w: number;
    h: number;
  };
};

type Match = {
  item: IsaacObject;
  similarity: number;
  source: "model" | "fallback";
};

type DetectionResponse = {
  model?: string;
  device?: string;
  boxes?: Array<{ x: number; y: number; w: number; h: number; score: number }>;
};

type BatchVerificationResponse = {
  results?: Array<{
    device?: string;
    model?: string;
    topK?: Array<{ itemId: string; score?: number; confidence?: number }>;
  }>;
};

const DETECTOR_CANDIDATE_THRESHOLD = 0.25;
const EMBEDDING_STRONG_THRESHOLD = 0.76;
const EMBEDDING_WEAK_THRESHOLD = 0.70;
const EMBEDDING_WEAK_MARGIN = 0.10;
const EMBEDDING_WEAK_DETECTOR_THRESHOLD = 0.70;

const fallbackItems = itemsData as IsaacObject[];
const objects = objectsData as IsaacObject[];

export function IsaacLens() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const uploadIdRef = useRef(0);
  const [regions, setRegions] = useState<Region[]>([]);
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const [selectedItemId, setSelectedItemId] = useState<string | null>(null);
  const [status, setStatus] = useState("等待截图");
  const [manualMode, setManualMode] = useState(false);
  const [manualBoxSize, setManualBoxSize] = useState(96);
  const [hasImage, setHasImage] = useState(false);
  const [modelMatches, setModelMatches] = useState<Match[] | null>(null);
  const [modelStatus, setModelStatus] = useState("模型服务未连接");

  const selectedRegion = regions.find((region) => region.id === selectedRegionId) ?? null;
  const fallbackMatches = useMemo(() => selectedRegion ? getMatches(selectedRegion) : [], [selectedRegion]);
  const matches = modelMatches ?? fallbackMatches;
  const selectedItem = objects.find((item) => item.id === selectedItemId) ?? null;

  useEffect(() => {
    let cancelled = false;
    setModelMatches(null);
    if (!selectedRegion) {
      setModelStatus("模型服务未连接");
      return;
    }

    setModelStatus("模型识别中...");
    fetch("http://127.0.0.1:8766/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: selectedRegion.modelImageUrl, topK: 8 }),
    })
      .then((response) => response.ok ? response.json() : Promise.reject(new Error(`HTTP ${response.status}`)))
      .then((payload: { device?: string; model?: string; topK?: Array<{ itemId: string; confidence?: number; score?: number }> }) => {
        if (cancelled) return;
        const nextMatches = (payload.topK ?? [])
          .map((match) => {
            const item = objects.find((candidate) => candidate.id === match.itemId);
            const score = match.score ?? match.confidence ?? 0;
            return item ? { item, similarity: Math.round(score * 100), source: "model" as const } : null;
          })
          .filter((match): match is Match => Boolean(match));
        setModelMatches(nextMatches.length ? nextMatches : null);
        setModelStatus(nextMatches.length ? `${payload.model ?? "embedding"} top-k · ${payload.device ?? "local"}` : "模型无结果，已回退旧匹配");
      })
      .catch(() => {
        if (cancelled) return;
        setModelMatches(null);
        setModelStatus("模型服务未启动，已回退旧匹配");
      });

    return () => {
      cancelled = true;
    };
  }, [selectedRegion]);

  function handleFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;

    const uploadId = uploadIdRef.current + 1;
    uploadIdRef.current = uploadId;
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        imageRef.current = img;
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext("2d", { willReadFrequently: true });
        if (!canvas || !ctx) return;
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);
        setHasImage(true);
        setRegions([]);
        setSelectedRegionId(null);
        setSelectedItemId(null);
        setModelMatches(null);
        setManualMode(false);
        setManualBoxSize(defaultManualBoxSize(img.naturalWidth, img.naturalHeight));
        setStatus(`已载入 ${img.naturalWidth} x ${img.naturalHeight} 截图，正在检测房间道具...`);
        detectRoomObjects(String(reader.result), canvas, ctx, img, uploadId);
      };
      img.src = String(reader.result);
    };
    reader.readAsDataURL(file);
  }

  async function detectRoomObjects(
    imageDataUrl: string,
    canvas: HTMLCanvasElement,
    ctx: CanvasRenderingContext2D,
    image: HTMLImageElement,
    uploadId: number,
  ) {
    try {
      const response = await fetch("http://127.0.0.1:8767/detect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: imageDataUrl }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json() as DetectionResponse;
      if (uploadId !== uploadIdRef.current) return;
      const candidateRegions = (payload.boxes ?? [])
        .filter((box) => (
          box.score >= DETECTOR_CANDIDATE_THRESHOLD &&
          isInsideRoomFloor(box, canvas.width, canvas.height)
        ))
        .map((box, index) => makeRegion(
        canvas,
        ctx,
        image,
        { ...box, count: box.score },
        `检测 ${index + 1}`,
      ));
      setStatus(`检测到 ${candidateRegions.length} 个候选，正在进行向量验证...`);
      const nextRegions = await verifyDetectedRegions(candidateRegions, uploadId);
      if (uploadId !== uploadIdRef.current) return;
      const selectedId = nextRegions[0]?.id ?? null;
      setRegions(nextRegions);
      setSelectedRegionId(selectedId);
      setSelectedItemId(null);
      setModelMatches(null);
      draw(canvas, ctx, image, nextRegions, selectedId);
      setStatus(nextRegions.length
        ? `标注 ${nextRegions.length} 个房间道具 · 检测 + MobileCLIP 验证 · ${payload.device ?? "local"}`
        : "未检测到房间道具，可使用“框选道具”手动添加");
    } catch {
      if (uploadId !== uploadIdRef.current) return;
      draw(canvas, ctx, image, [], null);
      setStatus("目标检测服务未启动，可使用“框选道具”手动添加");
    }
  }

  async function verifyDetectedRegions(candidateRegions: Region[], uploadId: number): Promise<Region[]> {
    if (!candidateRegions.length) return [];
    try {
      const response = await fetch("http://127.0.0.1:8766/predict-batch", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ images: candidateRegions.map((region) => region.modelImageUrl), topK: 2 }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const payload = await response.json() as BatchVerificationResponse;
      if (uploadId !== uploadIdRef.current) return [];
      return candidateRegions.filter((_, index) => {
        const matches = payload.results?.[index]?.topK ?? [];
        const bestScore = matches[0]?.score ?? matches[0]?.confidence ?? 0;
        const secondScore = matches[1]?.score ?? matches[1]?.confidence ?? 0;
        if (bestScore >= EMBEDDING_STRONG_THRESHOLD) return true;
        return (
          bestScore >= EMBEDDING_WEAK_THRESHOLD &&
          bestScore - secondScore >= EMBEDDING_WEAK_MARGIN &&
          candidateRegions[index].score >= EMBEDDING_WEAK_DETECTOR_THRESHOLD
        );
      });
    } catch {
      return candidateRegions.filter((region) => region.score >= 0.75);
    }
  }

  function handleCanvasClick(event: MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d", { willReadFrequently: true });
    const image = imageRef.current;
    if (!canvas || !ctx || !image) return;

    const rect = canvas.getBoundingClientRect();
    const point = {
      x: (event.clientX - rect.left) * canvas.width / rect.width,
      y: (event.clientY - rect.top) * canvas.height / rect.height,
    };

    if (manualMode) {
      const size = Math.max(24, Math.min(Math.min(canvas.width, canvas.height), manualBoxSize));
      const box = {
        x: Math.max(0, point.x - size / 2),
        y: Math.max(0, point.y - size / 2),
        w: Math.min(size, canvas.width - point.x + size / 2),
        h: Math.min(size, canvas.height - point.y + size / 2),
        count: 1,
      };
      const region = makeRegion(canvas, ctx, image, box, `手动 ${regions.length + 1}`);
      const nextRegions = [...regions, region];
      setRegions(nextRegions);
      setSelectedRegionId(region.id);
      setSelectedItemId(null);
      setModelMatches(null);
      draw(canvas, ctx, image, nextRegions, region.id);
      setStatus(`已添加手动候选框 ${size}x${size}`);
      return;
    }

    const hit = regions.find((region) => (
      point.x >= region.x &&
      point.x <= region.x + region.w &&
      point.y >= region.y &&
      point.y <= region.y + region.h
    ));
    if (hit) {
      setSelectedRegionId(hit.id);
      setSelectedItemId(null);
      setModelMatches(null);
      draw(canvas, ctx, image, regions, hit.id);
    }
  }

  function chooseRegion(regionId: string) {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d", { willReadFrequently: true });
    const image = imageRef.current;
    setSelectedRegionId(regionId);
    setSelectedItemId(null);
    setModelMatches(null);
    if (canvas && ctx && image) draw(canvas, ctx, image, regions, regionId);
  }

  return (
    <main className="app-shell">
      <section className="workspace" aria-label="截图识别工作区">
        <header className="topbar">
          <div>
            <h1>Isaac Item Lens</h1>
            <p>{objects.length} objects · MobileCLIP2-S0 检索版</p>
          </div>
          <div className="actions">
            <label className="file-button">
              <input type="file" accept="image/*" onChange={handleFile} />
              上传截图
            </label>
            <button
              type="button"
              disabled={!hasImage}
              aria-pressed={manualMode}
              onClick={() => setManualMode((value) => !value)}
            >
              框选道具
            </button>
            {manualMode && (
              <label className="manual-size-control">
                <span>框 {manualBoxSize}px</span>
                <input
                  type="range"
                  min="32"
                  max="180"
                  step="2"
                  value={manualBoxSize}
                  onChange={(event) => setManualBoxSize(Number(event.target.value))}
                />
              </label>
            )}
          </div>
        </header>

        <div className="stage-wrap">
          <canvas
            ref={canvasRef}
            className={hasImage ? "image-canvas visible" : "image-canvas"}
            aria-label="上传后的游戏截图"
            onClick={handleCanvasClick}
          />
          {!hasImage && (
            <div className="empty-state">
              <strong>上传一张游戏截图</strong>
              <span>上传后自动检测房间道具，漏检时可手动补框。</span>
            </div>
          )}
        </div>

        <footer className="statusbar">
          <span>{status}</span>
          <span>{manualMode ? `框选道具：当前 ${manualBoxSize}px，点击截图中的道具中心` : "识别在本地完成，不上传图片"}</span>
        </footer>
      </section>

      <aside className="side-panel" aria-label="识别结果">
        <section>
          <h2>候选区域</h2>
          <div className="region-list">
            {regions.length === 0 && <p className="muted">上传后自动检测房间道具；漏检时可用“框选道具”补充。</p>}
            {regions.map((region) => (
              <button
                className={`region-card${region.id === selectedRegionId ? " active" : ""}`}
                key={region.id}
                type="button"
                onClick={() => chooseRegion(region.id)}
              >
                <span className="thumb"><img src={region.imageUrl} alt="" /></span>
                <span>
                  <span className="card-title">{region.label}</span>
                  <span className="card-meta">
                    框 {region.w}x{region.h} · 模型 {region.modelBox.w}x{region.modelBox.h}
                  </span>
                </span>
              </button>
            ))}
          </div>
        </section>

        <section>
          <h2>Top K 相似对象</h2>
          <p className="model-status">{modelStatus}</p>
          <div className="match-list">
            {matches.length === 0 && <p className="muted">选择一个候选区域后显示相似对象。</p>}
            {matches.map(({ item, similarity, source }) => (
              <button
                className={`match-card${item.id === selectedItemId ? " active" : ""}`}
                key={item.id}
                type="button"
                onClick={() => setSelectedItemId(item.id)}
              >
                <span className="thumb"><img src={item.iconPath} alt="" /></span>
                <span>
                  <span className="card-title">{item.nameZh}</span>
                  <span className="card-meta">
                    #{item.gameId} · {item.nameEn} · {source === "model" ? "置信度" : "相似度"} {similarity}%
                  </span>
                </span>
              </button>
            ))}
          </div>
        </section>

        <section className="detail-panel">
          <h2>对象说明</h2>
          {!selectedItem && <p className="muted">点击候选区域或相似对象后显示描述。</p>}
          {selectedItem && (
            <>
              <div className="thumb large"><img src={selectedItem.iconPath} alt="" /></div>
              <h3>{selectedItem.nameZh}</h3>
              <p className="muted">#{selectedItem.gameId} · {selectedItem.nameEn}</p>
              <div className="tag-row">
                {selectedItem.type && <span className="tag">{selectedItem.type}</span>}
                {selectedItem.pools.slice(0, 4).map((pool) => <span className="tag" key={pool}>{pool}</span>)}
              </div>
              <p className="muted">{selectedItem.pickup || selectedItem.description}</p>
              <ul className="effect-list">
                {selectedItem.effects.slice(0, 8).map((effect) => <li key={effect}>{effect}</li>)}
              </ul>
              <p className="source-line">来源：{selectedItem.sourceName}</p>
            </>
          )}
        </section>
      </aside>
    </main>
  );
}

function isInsideRoomFloor(
  box: { x: number; y: number; w: number; h: number },
  width: number,
  height: number,
) {
  const centerX = box.x + box.w / 2;
  const centerY = box.y + box.h / 2;
  return (
    centerX >= width * 0.12 &&
    centerX <= width * 0.88 &&
    centerY >= height * 0.24 &&
    centerY <= height * 0.82
  );
}

function defaultManualBoxSize(width: number, height: number) {
  return Math.max(48, Math.min(140, Math.round(Math.min(width, height) * 0.075)));
}

function makeRegion(
  canvas: HTMLCanvasElement,
  ctx: CanvasRenderingContext2D,
  image: HTMLImageElement,
  box: { x: number; y: number; w: number; h: number; count: number },
  label: string,
): Region {
  drawBaseImage(canvas, ctx, image);
  const x = Math.max(0, Math.round(box.x));
  const y = Math.max(0, Math.round(box.y));
  const w = Math.max(1, Math.min(canvas.width - x, Math.round(box.w)));
  const h = Math.max(1, Math.min(canvas.height - y, Math.round(box.h)));
  const matchCrop = createMatchCrop(canvas, x, y, w, h);
  return {
    id: `region-${crypto.randomUUID()}`,
    label,
    x,
    y,
    w,
    h,
    score: box.count,
    feature: matchCrop.feature,
    hash: matchCrop.hash,
    descriptor: matchCrop.descriptor,
    imageUrl: matchCrop.imageUrl,
    modelImageUrl: matchCrop.modelImageUrl,
    modelBox: matchCrop.modelBox,
    matchBox: matchCrop.matchBox,
  };
}

function createMatchCrop(source: HTMLCanvasElement, x: number, y: number, w: number, h: number) {
  const raw = document.createElement("canvas");
  raw.width = w;
  raw.height = h;
  const rawCtx = raw.getContext("2d", { willReadFrequently: true });
  if (!rawCtx) {
    const fallbackFeature = { r: 0, g: 0, b: 0, edge: 0, aspect: w / h };
    return {
      feature: fallbackFeature,
      hash: "",
      descriptor: [],
      imageUrl: "",
      modelImageUrl: "",
      modelBox: { x, y, w, h },
      matchBox: { x: 0, y: 0, w, h },
    };
  }
  rawCtx.drawImage(source, x, y, w, h, 0, 0, w, h);
  const modelCrop = createModelInputCrop(source, x, y, w, h);

  const rawImage = rawCtx.getImageData(0, 0, w, h);
  const matchBox = findForegroundBox(rawImage, w, h);
  const normalized = document.createElement("canvas");
  normalized.width = 64;
  normalized.height = 64;
  const normalizedCtx = normalized.getContext("2d", { willReadFrequently: true });
  if (!normalizedCtx) {
    const fallbackFeature = extractFeature(rawCtx, 0, 0, w, h);
    return {
      feature: fallbackFeature,
      hash: "",
      descriptor: visualDescriptor(raw),
      imageUrl: modelCrop.imageUrl,
      modelImageUrl: modelCrop.imageUrl,
      modelBox: modelCrop.box,
      matchBox,
    };
  }

  normalizedCtx.imageSmoothingEnabled = false;
  normalizedCtx.clearRect(0, 0, 64, 64);
  const scale = Math.min(56 / matchBox.w, 56 / matchBox.h);
  const dw = Math.max(1, Math.round(matchBox.w * scale));
  const dh = Math.max(1, Math.round(matchBox.h * scale));
  normalizedCtx.drawImage(
    raw,
    matchBox.x,
    matchBox.y,
    matchBox.w,
    matchBox.h,
    Math.round((64 - dw) / 2),
    Math.round((64 - dh) / 2),
    dw,
    dh,
  );

  return {
    feature: extractFeature(normalizedCtx, 0, 0, 64, 64),
    hash: perceptualHash(normalized),
    descriptor: visualDescriptor(normalized),
    imageUrl: modelCrop.imageUrl,
    modelImageUrl: modelCrop.imageUrl,
    modelBox: modelCrop.box,
    matchBox,
  };
}

function createModelInputCrop(source: HTMLCanvasElement, x: number, y: number, w: number, h: number) {
  const modelInput = document.createElement("canvas");
  modelInput.width = 96;
  modelInput.height = 96;
  const modelCtx = modelInput.getContext("2d", { willReadFrequently: true });
  const centerX = x + w / 2;
  const centerY = y + h / 2;
  const side = Math.round(Math.max(w, h));
  const sx = Math.max(0, Math.min(Math.max(0, source.width - side), Math.round(centerX - side / 2)));
  const sy = Math.max(0, Math.min(Math.max(0, source.height - side), Math.round(centerY - side / 2)));
  const sw = Math.min(side, source.width - sx);
  const sh = Math.min(side, source.height - sy);
  if (modelCtx) {
    modelCtx.imageSmoothingEnabled = true;
    modelCtx.fillStyle = "#000";
    modelCtx.fillRect(0, 0, 96, 96);
    const scale = 96 / side;
    const dw = Math.max(1, Math.round(sw * scale));
    const dh = Math.max(1, Math.round(sh * scale));
    modelCtx.drawImage(
      source,
      sx,
      sy,
      sw,
      sh,
      Math.round((96 - dw) / 2),
      Math.round((96 - dh) / 2),
      dw,
      dh,
    );
  }
  return {
    imageUrl: modelInput.toDataURL("image/png"),
    box: { x: sx, y: sy, w: sw, h: sh },
  };
}

function findForegroundBox(imageData: ImageData, width: number, height: number) {
  const data = imageData.data;
  const darkBox = findNonDarkBox(data, width, height);
  if (darkBox) return darkBox;

  const border = Math.max(2, Math.round(Math.min(width, height) * 0.12));
  const background = estimateBorderColor(data, width, height, border);
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  let active = 0;

  for (let py = 0; py < height; py += 1) {
    for (let px = 0; px < width; px += 1) {
      const index = (py * width + px) * 4;
      const r = data[index];
      const g = data[index + 1];
      const b = data[index + 2];
      const colorDistance = Math.hypot(r - background.r, g - background.g, b - background.b);
      const sat = saturationOf(r, g, b);
      const contrast = localContrast(data, width, height, px, py);
      const inLooseCenter = px > width * 0.08 && px < width * 0.92 && py > height * 0.08 && py < height * 0.92;
      if (inLooseCenter && (colorDistance > 30 || sat > background.saturation + 0.16 || contrast > 24)) {
        minX = Math.min(minX, px);
        minY = Math.min(minY, py);
        maxX = Math.max(maxX, px);
        maxY = Math.max(maxY, py);
        active += 1;
      }
    }
  }

  const minActive = Math.max(12, width * height * 0.01);
  if (active < minActive || maxX <= minX || maxY <= minY) {
    const size = Math.round(Math.min(width, height) * 0.72);
    return {
      x: Math.max(0, Math.round((width - size) / 2)),
      y: Math.max(0, Math.round((height - size) / 2)),
      w: Math.min(width, size),
      h: Math.min(height, size),
    };
  }

  const pad = Math.max(3, Math.round(Math.min(width, height) * 0.08));
  const x = Math.max(0, minX - pad);
  const y = Math.max(0, minY - pad);
  return {
    x,
    y,
    w: Math.min(width - x, maxX - minX + 1 + pad * 2),
    h: Math.min(height - y, maxY - minY + 1 + pad * 2),
  };
}

function findNonDarkBox(data: Uint8ClampedArray, width: number, height: number) {
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  let active = 0;
  for (let py = 0; py < height; py += 1) {
    for (let px = 0; px < width; px += 1) {
      const index = (py * width + px) * 4;
      const r = data[index];
      const g = data[index + 1];
      const b = data[index + 2];
      const luma = 0.299 * r + 0.587 * g + 0.114 * b;
      const sat = saturationOf(r, g, b);
      const inCenter = px > width * 0.05 && px < width * 0.95 && py > height * 0.05 && py < height * 0.95;
      if (inCenter && (luma > 72 || sat > 0.28)) {
        minX = Math.min(minX, px);
        minY = Math.min(minY, py);
        maxX = Math.max(maxX, px);
        maxY = Math.max(maxY, py);
        active += 1;
      }
    }
  }
  if (active < Math.max(18, width * height * 0.035) || maxX <= minX || maxY <= minY) return null;
  const pad = Math.max(2, Math.round(Math.min(width, height) * 0.05));
  const x = Math.max(0, minX - pad);
  const y = Math.max(0, minY - pad);
  return {
    x,
    y,
    w: Math.min(width - x, maxX - minX + 1 + pad * 2),
    h: Math.min(height - y, maxY - minY + 1 + pad * 2),
  };
}

function estimateBorderColor(data: Uint8ClampedArray, width: number, height: number, border: number) {
  let r = 0;
  let g = 0;
  let b = 0;
  let saturation = 0;
  let count = 0;
  for (let py = 0; py < height; py += 1) {
    for (let px = 0; px < width; px += 1) {
      if (px > border && px < width - border && py > border && py < height - border) continue;
      const index = (py * width + px) * 4;
      r += data[index];
      g += data[index + 1];
      b += data[index + 2];
      saturation += saturationOf(data[index], data[index + 1], data[index + 2]);
      count += 1;
    }
  }
  return {
    r: r / Math.max(1, count),
    g: g / Math.max(1, count),
    b: b / Math.max(1, count),
    saturation: saturation / Math.max(1, count),
  };
}

function saturationOf(r: number, g: number, b: number) {
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  return max === 0 ? 0 : (max - min) / max;
}

function localContrast(pixels: Uint8ClampedArray, width: number, height: number, x: number, y: number) {
  const base = lumaAt(pixels, width, x, y);
  let maxDelta = 0;
  for (const [ox, oy] of [[-2, 0], [2, 0], [0, -2], [0, 2]]) {
    const nx = Math.max(0, Math.min(width - 1, x + ox));
    const ny = Math.max(0, Math.min(height - 1, y + oy));
    maxDelta = Math.max(maxDelta, Math.abs(base - lumaAt(pixels, width, nx, ny)));
  }
  return maxDelta;
}

function lumaAt(pixels: Uint8ClampedArray, width: number, x: number, y: number) {
  const index = (y * width + x) * 4;
  return 0.299 * pixels[index] + 0.587 * pixels[index + 1] + 0.114 * pixels[index + 2];
}

function extractFeature(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number): Feature {
  const imageData = ctx.getImageData(x, y, w, h);
  const data = imageData.data;
  let r = 0;
  let g = 0;
  let b = 0;
  let edge = 0;
  let n = 0;
  for (let i = 0; i < data.length; i += 16) {
    if (data[i + 3] < 24) continue;
    r += data[i];
    g += data[i + 1];
    b += data[i + 2];
    edge += Math.max(data[i], data[i + 1], data[i + 2]) - Math.min(data[i], data[i + 1], data[i + 2]);
    n += 1;
  }
  const box = alphaBox(imageData, w, h);
  return {
    r: r / Math.max(1, n),
    g: g / Math.max(1, n),
    b: b / Math.max(1, n),
    edge: edge / Math.max(1, n),
    aspect: box ? box.w / box.h : w / h,
  };
}

function getMatches(region: Region): Match[] {
  return fallbackItems
    .filter((item) => item.iconFeature)
    .map((item) => {
      const iconFeature = item.iconFeature;
      if (!iconFeature) return { item, similarity: 0, source: "fallback" as const };
      const descriptorDistance = descriptorMse(region.descriptor, iconFeature.descriptor);
      const hashDistance = hammingDistance(region.hash, iconFeature.hash);
      const colorDistance = Math.hypot(
        region.feature.r - iconFeature.r,
        region.feature.g - iconFeature.g,
        region.feature.b - iconFeature.b,
      ) / 255;
      const aspectDistance = Math.abs(region.feature.aspect - iconFeature.aspect);
      const distance = descriptorDistance * 1.9 + hashDistance * 0.22 + colorDistance * 6 + aspectDistance * 4;
      return { item, similarity: Math.max(0, Math.min(99, Math.round(100 - distance))), source: "fallback" as const };
    })
    .sort((a, b) => b.similarity - a.similarity)
    .slice(0, 5);
}

function visualDescriptor(canvas: HTMLCanvasElement) {
  const sample = document.createElement("canvas");
  sample.width = 16;
  sample.height = 16;
  const sampleCtx = sample.getContext("2d", { willReadFrequently: true });
  if (!sampleCtx) return [];
  sampleCtx.imageSmoothingEnabled = true;
  sampleCtx.drawImage(canvas, 0, 0, 16, 16);
  const data = sampleCtx.getImageData(0, 0, 16, 16).data;
  const gray: number[] = [];
  for (let index = 0; index < data.length; index += 4) {
    gray.push(Math.round((0.299 * data[index] + 0.587 * data[index + 1] + 0.114 * data[index + 2]) / 8));
  }
  const edges: number[] = [];
  for (let y = 0; y < 16; y += 1) {
    for (let x = 0; x < 16; x += 1) {
      const current = gray[y * 16 + x];
      const right = gray[y * 16 + Math.min(15, x + 1)];
      const down = gray[Math.min(15, y + 1) * 16 + x];
      edges.push(Math.min(32, Math.abs(current - right) + Math.abs(current - down)));
    }
  }
  return [...gray, ...edges];
}

function descriptorMse(left: number[], right: number[]) {
  const length = Math.min(left.length, right.length);
  if (!length) return 80;
  let total = 0;
  for (let index = 0; index < length; index += 1) {
    const delta = left[index] - right[index];
    total += delta * delta;
  }
  return Math.sqrt(total / length);
}

function alphaBox(imageData: ImageData, width: number, height: number) {
  const data = imageData.data;
  let minX = width;
  let minY = height;
  let maxX = -1;
  let maxY = -1;
  for (let py = 0; py < height; py += 1) {
    for (let px = 0; px < width; px += 1) {
      if (data[(py * width + px) * 4 + 3] < 24) continue;
      minX = Math.min(minX, px);
      minY = Math.min(minY, py);
      maxX = Math.max(maxX, px);
      maxY = Math.max(maxY, py);
    }
  }
  return maxX >= minX && maxY >= minY ? { x: minX, y: minY, w: maxX - minX + 1, h: maxY - minY + 1 } : null;
}

function perceptualHash(canvas: HTMLCanvasElement) {
  const sample = document.createElement("canvas");
  sample.width = 32;
  sample.height = 32;
  const sampleCtx = sample.getContext("2d", { willReadFrequently: true });
  if (!sampleCtx) return "";
  sampleCtx.drawImage(canvas, 0, 0, 32, 32);
  const data = sampleCtx.getImageData(0, 0, 32, 32).data;
  const values: number[] = [];
  for (let index = 0; index < data.length; index += 4) {
    values.push((0.299 * data[index] + 0.587 * data[index + 1] + 0.114 * data[index + 2]) / 255);
  }

  const coeffs: number[] = [];
  for (let v = 0; v < 8; v += 1) {
    for (let u = 0; u < 8; u += 1) {
      let total = 0;
      for (let py = 0; py < 32; py += 1) {
        for (let px = 0; px < 32; px += 1) {
          total += values[py * 32 + px] *
            Math.cos(((2 * px + 1) * u * Math.PI) / 64) *
            Math.cos(((2 * py + 1) * v * Math.PI) / 64);
        }
      }
      const cu = u === 0 ? 1 / Math.sqrt(2) : 1;
      const cv = v === 0 ? 1 / Math.sqrt(2) : 1;
      coeffs.push(0.25 * cu * cv * total);
    }
  }

  const comparable = coeffs.slice(1);
  const median = [...comparable].sort((a, b) => a - b)[Math.floor(comparable.length / 2)];
  return bitsToHex(comparable.map((value) => value > median));
}

function bitsToHex(bits: boolean[]) {
  const padded = [...bits];
  while (padded.length % 4 !== 0) padded.push(false);
  let hash = "";
  for (let index = 0; index < padded.length; index += 4) {
    const nibble = padded.slice(index, index + 4).reduce((value, bit) => value * 2 + (bit ? 1 : 0), 0);
    hash += nibble.toString(16);
  }
  return hash;
}

function hammingDistance(left: string, right: string) {
  const length = Math.min(left.length, right.length);
  let distance = Math.abs(left.length - right.length) * 4;
  for (let index = 0; index < length; index += 1) {
    distance += bitCount(parseInt(left[index], 16) ^ parseInt(right[index], 16));
  }
  return distance;
}

function bitCount(value: number) {
  let count = 0;
  let rest = value;
  while (rest) {
    count += rest & 1;
    rest >>= 1;
  }
  return count;
}

function drawBaseImage(canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D, image: HTMLImageElement) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(image, 0, 0);
}

function draw(canvas: HTMLCanvasElement, ctx: CanvasRenderingContext2D, image: HTMLImageElement, regions: Region[], activeId: string | null) {
  drawBaseImage(canvas, ctx, image);
  for (const region of regions) {
    const active = region.id === activeId;
    ctx.save();
    ctx.lineWidth = active ? 5 : 3;
    ctx.strokeStyle = active ? "#ffd166" : "#69c6b8";
    ctx.fillStyle = active ? "rgba(255, 209, 102, 0.14)" : "rgba(105, 198, 184, 0.12)";
    ctx.fillRect(region.x, region.y, region.w, region.h);
    ctx.strokeRect(region.x, region.y, region.w, region.h);
    ctx.font = "700 28px system-ui";
    ctx.textBaseline = "top";
    const metrics = ctx.measureText(region.label);
    ctx.fillStyle = active ? "#ffd166" : "#69c6b8";
    ctx.fillRect(region.x, Math.max(0, region.y - 34), metrics.width + 18, 32);
    ctx.fillStyle = "#101113";
    ctx.fillText(region.label, region.x + 9, Math.max(0, region.y - 30));
    ctx.restore();
  }
}
