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

type DetectionStage = "idle" | "detecting" | "verifying" | "complete" | "error";
type ManualBoxSize = "tiny" | "small" | "medium" | "large" | "huge";

const MANUAL_BOX_OPTIONS: Array<{ value: ManualBoxSize; label: string; scale: number }> = [
  { value: "tiny", label: "超级小", scale: 0.52 },
  { value: "small", label: "小", scale: 0.76 },
  { value: "medium", label: "中等", scale: 1 },
  { value: "large", label: "大", scale: 1.36 },
  { value: "huge", label: "超级大", scale: 1.78 },
];

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
  const [manualTipVisible, setManualTipVisible] = useState(false);
  const [manualBoxSize, setManualBoxSize] = useState<ManualBoxSize>("medium");
  const [hasImage, setHasImage] = useState(false);
  const [modelMatches, setModelMatches] = useState<Match[] | null>(null);
  const [modelStatus, setModelStatus] = useState("选择一个标注查看相似道具");
  const [detectionStage, setDetectionStage] = useState<DetectionStage>("idle");
  const [detectionProgress, setDetectionProgress] = useState(0);
  const [detectedCount, setDetectedCount] = useState(0);

  const selectedRegion = regions.find((region) => region.id === selectedRegionId) ?? null;
  const fallbackMatches = useMemo(() => selectedRegion ? getMatches(selectedRegion) : [], [selectedRegion]);
  const matches = modelMatches ?? fallbackMatches;
  const selectedItem = objects.find((item) => item.id === selectedItemId) ?? null;

  useEffect(() => {
    let cancelled = false;
    setModelMatches(null);
    if (!selectedRegion) {
      setModelStatus("选择一个标注查看相似道具");
      return;
    }

    setModelStatus("正在查找相似道具...");
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
        setModelStatus(nextMatches.length ? "请选择最符合截图的道具" : "没有找到合适的相似道具");
      })
      .catch(() => {
        if (cancelled) return;
        setModelMatches(null);
        setModelStatus("暂时无法查找相似道具");
      });

    return () => {
      cancelled = true;
    };
  }, [selectedRegion]);

  useEffect(() => {
    const ceiling = detectionStage === "detecting" ? 58 : detectionStage === "verifying" ? 92 : null;
    if (ceiling === null) return;
    const timer = window.setInterval(() => {
      setDetectionProgress((value) => Math.min(ceiling, value + Math.max(1, Math.ceil((ceiling - value) * 0.14))));
    }, 260);
    return () => window.clearInterval(timer);
  }, [detectionStage]);

  useEffect(() => {
    if (!manualTipVisible) return;
    const timer = window.setTimeout(() => setManualTipVisible(false), 4200);
    return () => window.clearTimeout(timer);
  }, [manualTipVisible]);

  function handleFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    event.target.value = "";

    const uploadId = uploadIdRef.current + 1;
    uploadIdRef.current = uploadId;
    setDetectionStage("detecting");
    setDetectionProgress(8);
    setDetectedCount(0);
    const reader = new FileReader();
    reader.onload = () => {
      const img = new Image();
      img.onload = () => {
        imageRef.current = img;
        const canvas = canvasRef.current;
        const ctx = canvas?.getContext("2d", { willReadFrequently: true });
        if (!canvas || !ctx) {
          setDetectionStage("error");
          setDetectionProgress(100);
          return;
        }
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        ctx.drawImage(img, 0, 0);
        setHasImage(true);
        setRegions([]);
        setSelectedRegionId(null);
        setSelectedItemId(null);
        setModelMatches(null);
        setManualMode(false);
        setManualTipVisible(false);
        setStatus("正在检查截图中的道具...");
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
        `道具 ${index + 1}`,
      ));
      setDetectionStage("verifying");
      setDetectionProgress((value) => Math.max(value, 64));
      setStatus("正在核对检测结果...");
      const nextRegions = await verifyDetectedRegions(candidateRegions, uploadId);
      if (uploadId !== uploadIdRef.current) return;
      const selectedId = nextRegions[0]?.id ?? null;
      setRegions(nextRegions);
      setSelectedRegionId(selectedId);
      setSelectedItemId(null);
      setModelMatches(null);
      draw(canvas, ctx, image, nextRegions, selectedId);
      setDetectedCount(nextRegions.length);
      setDetectionProgress(100);
      setDetectionStage("complete");
      setStatus(nextRegions.length
        ? `已找到 ${nextRegions.length} 个可能的道具`
        : "未自动检测到道具，请手动选取");
    } catch {
      if (uploadId !== uploadIdRef.current) return;
      draw(canvas, ctx, image, [], null);
      setDetectionProgress(100);
      setDetectionStage("error");
      setStatus("自动检测暂时不可用，请手动选取");
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

    const deleteTarget = regions.find((region) => isDeleteButtonHit(point, region, canvas));
    if (deleteTarget) {
      removeRegion(deleteTarget.id);
      return;
    }

    if (manualMode) {
      const size = manualBoxPixels(canvas.width, canvas.height, manualBoxSize);
      const box = {
        x: Math.max(0, point.x - size / 2),
        y: Math.max(0, point.y - size / 2),
        w: Math.min(size, canvas.width - point.x + size / 2),
        h: Math.min(size, canvas.height - point.y + size / 2),
        count: 1,
      };
      const region = makeRegion(canvas, ctx, image, box, `道具 ${regions.length + 1}`);
      const nextRegions = [...regions, region];
      setRegions(nextRegions);
      setSelectedRegionId(region.id);
      setSelectedItemId(null);
      setModelMatches(null);
      draw(canvas, ctx, image, nextRegions, region.id);
      setStatus("已添加手动标注");
      setManualTipVisible(false);
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

  function removeRegion(regionId: string) {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d", { willReadFrequently: true });
    const image = imageRef.current;
    const nextRegions = regions.filter((region) => region.id !== regionId);
    const nextSelectedId = selectedRegionId === regionId ? nextRegions[0]?.id ?? null : selectedRegionId;
    setRegions(nextRegions);
    setSelectedRegionId(nextSelectedId);
    setSelectedItemId(null);
    setModelMatches(null);
    if (canvas && ctx && image) draw(canvas, ctx, image, nextRegions, nextSelectedId);
    setStatus(nextRegions.length ? `已保留 ${nextRegions.length} 个标注` : "标注已清空，可手动选取");
  }

  function startManualCorrection() {
    setDetectionStage("idle");
    setManualMode(true);
    setManualTipVisible(true);
    setStatus("点击截图中的道具进行手动选取");
  }

  function toggleManualMode() {
    setManualMode((value) => {
      const next = !value;
      setManualTipVisible(next);
      if (next) setStatus("点击图片中物品进行手动选取");
      return next;
    });
  }

  return (
    <main className={`app-shell${selectedRegion ? "" : " no-selection"}`}>
      <section className="workspace" aria-label="截图识别工作区">
        <header className="topbar">
          <div>
            <h1>Isaac Item Lens</h1>
            <p>离线道具识别</p>
          </div>
          <div className="actions">
            <label className="file-button">
              <input type="file" accept="image/*" onChange={handleFile} />
              {hasImage ? "换一张图" : "上传截图"}
            </label>
            <button
              type="button"
              disabled={!hasImage}
              aria-pressed={manualMode}
              onClick={toggleManualMode}
            >
              手动选取
            </button>
            {manualMode && (
              <label className="manual-size-control">
                <span>检测框大小：</span>
                <select
                  value={manualBoxSize}
                  onChange={(event) => setManualBoxSize(event.target.value as ManualBoxSize)}
                >
                  {MANUAL_BOX_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>{option.label}</option>
                  ))}
                </select>
              </label>
            )}
          </div>
          {manualTipVisible && (
            <div className="manual-guidance" role="status">点击图片中物品进行手动选取</div>
          )}
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
              <label className="empty-upload-button">
                <input type="file" accept="image/*" onChange={handleFile} />
                上传截图
              </label>
              <span>上传后自动检测房间道具，漏检时可手动选取。</span>
            </div>
          )}
          {detectionStage !== "idle" && (
            <div className="detection-backdrop" role="presentation">
              <section className="detection-dialog" role="dialog" aria-modal="true" aria-labelledby="detection-title">
                <div className={`detection-mark ${detectionStage}`} aria-hidden="true">
                  {detectionStage === "complete" ? "✓" : detectionStage === "error" ? "!" : ""}
                </div>
                <h2 id="detection-title">
                  {detectionStage === "detecting" && "正在查找道具"}
                  {detectionStage === "verifying" && "正在核对候选"}
                  {detectionStage === "complete" && (detectedCount ? `找到 ${detectedCount} 个道具` : "未自动检测到道具，请手动选取")}
                  {detectionStage === "error" && "自动检测未完成"}
                </h2>
                <p>
                  {detectionStage === "detecting" && "正在扫描房间画面，请稍候。"}
                  {detectionStage === "verifying" && "正在排除角色、界面与普通拾取物。"}
                  {detectionStage === "complete" && (detectedCount
                    ? "请检查自动检测结果。若有遗漏，可以继续手动选取。"
                    : "请在图片中手动选取需要识别的道具。")}
                  {detectionStage === "error" && "请在图片中手动选取需要识别的道具。"}
                </p>
                <div className="progress-track" aria-label="检测进度" aria-valuemin={0} aria-valuemax={100} aria-valuenow={detectionProgress} role="progressbar">
                  <span style={{ width: `${detectionProgress}%` }} />
                </div>
                {(detectionStage === "complete" || detectionStage === "error") && (
                  <div className={`dialog-actions${detectionStage === "complete" && detectedCount > 0 ? "" : " single"}`}>
                    {detectionStage === "complete" && detectedCount > 0 && (
                      <button type="button" className="secondary-action" onClick={() => setDetectionStage("idle")}>查看自动检测结果</button>
                    )}
                    <button type="button" className="primary-action" onClick={startManualCorrection}>手动选取</button>
                  </div>
                )}
              </section>
            </div>
          )}
          {selectedItem && (
            <div className="item-detail-backdrop" role="presentation">
              <section className="item-detail-dialog" role="dialog" aria-modal="true" aria-labelledby="item-detail-title">
                <button
                  className="delete-detail-window"
                  type="button"
                  title="删除窗口"
                  aria-label="删除道具详情窗口"
                  onClick={() => setSelectedItemId(null)}
                >×</button>
                <div className="item-detail-heading">
                  <div className="thumb large"><img src={selectedItem.iconPath} alt="" /></div>
                  <div>
                    <h2 id="item-detail-title">{selectedItem.nameZh}</h2>
                    <p>{selectedItem.nameEn}</p>
                  </div>
                </div>
                <div className="tag-row">
                  {selectedItem.type && <span className="tag">{selectedItem.type}</span>}
                  {selectedItem.pools.slice(0, 4).map((pool) => <span className="tag" key={pool}>{pool}</span>)}
                </div>
                <p className="item-description">{selectedItem.pickup || selectedItem.description}</p>
                <ul className="effect-list">
                  {selectedItem.effects.slice(0, 8).map((effect) => <li key={effect}>{effect}</li>)}
                </ul>
                <p className="source-line">来源：{selectedItem.sourceName}</p>
              </section>
            </div>
          )}
        </div>

        <footer className="statusbar">
          <span>{status}</span>
          <span>{manualMode ? "手动选取已开启：点击需要识别的道具" : "识别在设备本地完成"}</span>
        </footer>
      </section>

      {selectedRegion && <aside className="side-panel" aria-label="识别结果">
        <section>
          <h2>选中区域</h2>
          <div className="region-list">
            {regions.length === 0 && <p className="muted">上传后自动检测房间道具；漏检时可手动选取。</p>}
            {regions.map((region) => (
              <div className="region-row" key={region.id}>
                <button
                  className={`region-card${region.id === selectedRegionId ? " active" : ""}`}
                  type="button"
                  onClick={() => chooseRegion(region.id)}
                >
                  <span className="thumb"><img src={region.imageUrl} alt="" /></span>
                  <span>
                    <span className="card-title">{region.label}</span>
                    <span className="card-meta">点击查看相似道具</span>
                  </span>
                </button>
                <button className="delete-region" type="button" title="删除标注" aria-label={`删除 ${region.label}`} onClick={() => removeRegion(region.id)}>×</button>
              </div>
            ))}
          </div>
        </section>

        <section>
          <h2>相似道具</h2>
          <p className="model-status">{modelStatus}</p>
          <div className="match-list">
            {matches.length === 0 && <p className="muted">选择上方区域后显示相似道具。</p>}
            {matches.map(({ item }) => (
              <button
                className={`match-card${item.id === selectedItemId ? " active" : ""}`}
                key={item.id}
                type="button"
                onClick={() => setSelectedItemId(item.id)}
              >
                <span className="thumb"><img src={item.iconPath} alt="" /></span>
                <span>
                  <span className="card-title">{item.nameZh}</span>
                  <span className="card-meta">{item.nameEn}</span>
                </span>
              </button>
            ))}
          </div>
        </section>

      </aside>}
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

function manualBoxPixels(width: number, height: number, size: ManualBoxSize) {
  const option = MANUAL_BOX_OPTIONS.find((candidate) => candidate.value === size) ?? MANUAL_BOX_OPTIONS[2];
  const base = Math.max(48, Math.min(140, Math.round(Math.min(width, height) * 0.075)));
  return Math.max(24, Math.min(Math.min(width, height), Math.round(base * option.scale)));
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
    const deleteButton = deleteButtonGeometry(region, canvas);
    ctx.beginPath();
    ctx.arc(deleteButton.x, deleteButton.y, deleteButton.radius, 0, Math.PI * 2);
    ctx.fillStyle = "#d84f5f";
    ctx.fill();
    ctx.lineWidth = Math.max(3, deleteButton.radius * 0.14);
    ctx.strokeStyle = "#fff";
    const cross = deleteButton.radius * 0.38;
    ctx.beginPath();
    ctx.moveTo(deleteButton.x - cross, deleteButton.y - cross);
    ctx.lineTo(deleteButton.x + cross, deleteButton.y + cross);
    ctx.moveTo(deleteButton.x + cross, deleteButton.y - cross);
    ctx.lineTo(deleteButton.x - cross, deleteButton.y + cross);
    ctx.stroke();
    ctx.restore();
  }
}

function deleteButtonGeometry(region: Region, canvas: HTMLCanvasElement) {
  const radius = Math.max(24, Math.min(52, Math.min(canvas.width, canvas.height) * 0.035));
  return {
    x: Math.max(radius, Math.min(canvas.width - radius, region.x + region.w)),
    y: Math.max(radius, Math.min(canvas.height - radius, region.y)),
    radius,
  };
}

function isDeleteButtonHit(
  point: { x: number; y: number },
  region: Region,
  canvas: HTMLCanvasElement,
) {
  const button = deleteButtonGeometry(region, canvas);
  return Math.hypot(point.x - button.x, point.y - button.y) <= button.radius * 1.25;
}
