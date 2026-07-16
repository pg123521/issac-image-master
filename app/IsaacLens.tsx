"use client";

import * as ort from "onnxruntime-web";
import {
  ChangeEvent,
  PointerEvent as ReactPointerEvent,
  WheelEvent,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

type IndexedObject = {
  vectorIndex: number;
  id: string;
  kind: string;
  gameId: number;
  nameZh: string;
  nameEn: string;
  iconPath: string;
  pickup?: string;
  description?: string;
  effects?: string[];
  type?: string;
  pools?: string[];
  sourceName?: string;
};

type IndexMetadata = {
  rows: number;
  dimensions: number;
  objects: IndexedObject[];
};

type Match = {
  item: IndexedObject;
  score: number;
};

type Point = { x: number; y: number };
type Selection = { x: number; y: number };
type DragMode = "pan" | "selection";
type InstallPromptEvent = Event & {
  prompt: () => Promise<void>;
  userChoice: Promise<{ outcome: "accepted" | "dismissed" }>;
};

const BASE_URL = import.meta.env.BASE_URL || "/";
const MODEL_URL = `${BASE_URL}models/mobileclip-image-encoder.onnx`;
const VECTOR_URL = `${BASE_URL}models/item-vectors.f16`;
const METADATA_URL = `${BASE_URL}models/item-vectors.json`;
const OBJECTS_URL = `${BASE_URL}models/objects.json`;
const MIN_ZOOM = 0.35;
const MAX_ZOOM = 12;
const MIN_BOX = 24;
const MAX_BOX = 420;

let modelPromise: Promise<ort.InferenceSession> | null = null;
let indexPromise: Promise<{ metadata: IndexMetadata; vectors: Float32Array }> | null = null;

export function IsaacLens() {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const sourceCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const pointersRef = useRef(new Map<number, Point>());
  const dragRef = useRef<{
    mode: DragMode;
    pointerId: number;
    start: Point;
    startPan: Point;
    moved: boolean;
  } | null>(null);
  const pinchRef = useRef<{ distance: number; zoom: number; pan: Point; midpoint: Point } | null>(null);
  const requestRef = useRef(0);

  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [imageSize, setImageSize] = useState({ width: 1, height: 1 });
  const [stageSize, setStageSize] = useState({ width: 1, height: 1 });
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState<Point>({ x: 0, y: 0 });
  const [selection, setSelection] = useState<Selection | null>(null);
  const [boxSize, setBoxSize] = useState(82);
  const [matches, setMatches] = useState<Match[]>([]);
  const [candidateLimit, setCandidateLimit] = useState(20);
  const [selectedItem, setSelectedItem] = useState<IndexedObject | null>(null);
  const [status, setStatus] = useState("上传截图以识别物品");
  const [isSearching, setIsSearching] = useState(false);
  const [modelReady, setModelReady] = useState(false);
  const [loadProgress, setLoadProgress] = useState<number | null>(null);
  const [loadLabel, setLoadLabel] = useState("准备本地模型");
  const [installPrompt, setInstallPrompt] = useState<InstallPromptEvent | null>(null);
  const [showInstallHelp, setShowInstallHelp] = useState(false);
  const [isInstalled, setIsInstalled] = useState(false);

  const baseScale = Math.min(stageSize.width / imageSize.width, stageSize.height / imageSize.height);
  const renderedScale = baseScale * zoom;
  const imageOrigin = {
    x: stageSize.width / 2 + pan.x - imageSize.width * renderedScale / 2,
    y: stageSize.height / 2 + pan.y - imageSize.height * renderedScale / 2,
  };
  const selectionScreen = selection
    ? {
        x: imageOrigin.x + selection.x * renderedScale,
        y: imageOrigin.y + selection.y * renderedScale,
      }
    : null;

  useEffect(() => {
    const storedCandidateLimit = localStorage.getItem("candidateDisplayLimitV2");
    if (storedCandidateLimit !== null) {
      const saved = Number(storedCandidateLimit);
      if (Number.isFinite(saved)) setCandidateLimit(Math.min(50, Math.max(1, saved)));
    }
    if ("serviceWorker" in navigator) navigator.serviceWorker.register(`${BASE_URL}sw.js`).catch(() => undefined);
    const standalone = window.matchMedia("(display-mode: standalone)").matches
      || Boolean((navigator as Navigator & { standalone?: boolean }).standalone);
    setIsInstalled(standalone);
    const captureInstallPrompt = (event: Event) => {
      event.preventDefault();
      setInstallPrompt(event as InstallPromptEvent);
    };
    window.addEventListener("beforeinstallprompt", captureInstallPrompt);
    return () => window.removeEventListener("beforeinstallprompt", captureInstallPrompt);
  }, []);

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage) return;
    const observer = new ResizeObserver(([entry]) => {
      setStageSize({
        width: Math.max(1, entry.contentRect.width),
        height: Math.max(1, entry.contentRect.height),
      });
    });
    observer.observe(stage);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!selection || !sourceCanvasRef.current) return;
    const timer = window.setTimeout(() => runSearch(selection), 90);
    return () => window.clearTimeout(timer);
    // renderedScale changes the source crop size while keeping the visible box fixed.
  }, [selection, boxSize, renderedScale]);

  const visibleMatches = useMemo(() => matches.slice(0, candidateLimit), [matches, candidateLimit]);

  function handleFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    const url = URL.createObjectURL(file);
    const image = new Image();
    image.onload = () => {
      if (imageUrl) URL.revokeObjectURL(imageUrl);
      const sourceCanvas = document.createElement("canvas");
      sourceCanvas.width = image.naturalWidth;
      sourceCanvas.height = image.naturalHeight;
      sourceCanvas.getContext("2d")?.drawImage(image, 0, 0);
      sourceCanvasRef.current = sourceCanvas;
      imageRef.current = image;
      setImageSize({ width: image.naturalWidth, height: image.naturalHeight });
      setImageUrl(url);
      setZoom(1);
      setPan({ x: 0, y: 0 });
      setSelection(null);
      setSelectedItem(null);
      setMatches([]);
      setStatus("点击图片中的物品进行手动选取");
      void warmModel();
    };
    image.src = url;
  }

  async function warmModel() {
    try {
      setLoadProgress(1);
      setLoadLabel("正在下载本地识别模型");
      setStatus("首次加载本地模型…");
      await Promise.all([
        getModel((progress, phase) => {
          setLoadLabel(phase === "initializing" ? "正在初始化本地识别" : "正在下载本地识别模型");
          setLoadProgress(Math.max(1, Math.round(progress * 90)));
        }),
        getIndex(),
      ]);
      setLoadLabel("正在初始化本地识别");
      setLoadProgress(96);
      await new Promise((resolve) => window.setTimeout(resolve, 180));
      setModelReady(true);
      setLoadProgress(100);
      setStatus("点击图片中的物品进行手动选取");
      window.setTimeout(() => setLoadProgress(null), 320);
    } catch {
      setLoadProgress(null);
      setStatus("模型加载失败，请检查网络后重试");
    }
  }

  async function runSearch(nextSelection: Selection) {
    const source = sourceCanvasRef.current;
    if (!source || renderedScale <= 0) return;
    const request = ++requestRef.current;
    setIsSearching(true);
    setStatus(modelReady ? "正在本地识别…" : "正在加载并运行本地模型…");
    try {
      const side = Math.min(Math.min(source.width, source.height), boxSize / renderedScale);
      const sx = clamp(nextSelection.x - side / 2, 0, source.width - side);
      const sy = clamp(nextSelection.y - side / 2, 0, source.height - side);
      const input = cropToTensor(source, sx, sy, side);
      const [session, index] = await Promise.all([getModel(), getIndex()]);
      const output = await session.run({ image: new ort.Tensor("float32", input, [1, 3, 256, 256]) });
      const embedding = output.embedding.data as Float32Array;
      normalize(embedding);
      const nextMatches = topMatches(embedding, index.metadata, index.vectors, 50);
      if (request !== requestRef.current) return;
      setMatches(nextMatches);
      setModelReady(true);
      setStatus(`已完成本地识别 · ${nextMatches[0] ? `${(nextMatches[0].score * 100).toFixed(1)}%` : "无结果"}`);
    } catch {
      if (request !== requestRef.current) return;
      setMatches([]);
      setStatus("本地识别失败，请重新选择区域");
    } finally {
      if (request === requestRef.current) setIsSearching(false);
    }
  }

  function screenToImage(point: Point): Point {
    return {
      x: clamp((point.x - imageOrigin.x) / renderedScale, 0, imageSize.width),
      y: clamp((point.y - imageOrigin.y) / renderedScale, 0, imageSize.height),
    };
  }

  function localPoint(event: ReactPointerEvent): Point {
    const rect = stageRef.current?.getBoundingClientRect();
    return { x: event.clientX - (rect?.left ?? 0), y: event.clientY - (rect?.top ?? 0) };
  }

  function pointerDown(event: ReactPointerEvent<HTMLDivElement>) {
    if (!imageUrl) return;
    const point = localPoint(event);
    pointersRef.current.set(event.pointerId, point);
    event.currentTarget.setPointerCapture(event.pointerId);
    if (pointersRef.current.size === 2) {
      const [a, b] = [...pointersRef.current.values()];
      pinchRef.current = {
        distance: distance(a, b),
        zoom,
        pan,
        midpoint: midpoint(a, b),
      };
      dragRef.current = null;
      return;
    }
    const target = event.target as HTMLElement;
    dragRef.current = {
      mode: target.closest(".selection-box") ? "selection" : "pan",
      pointerId: event.pointerId,
      start: point,
      startPan: pan,
      moved: false,
    };
  }

  function pointerMove(event: ReactPointerEvent<HTMLDivElement>) {
    if (!pointersRef.current.has(event.pointerId)) return;
    const point = localPoint(event);
    pointersRef.current.set(event.pointerId, point);
    if (pointersRef.current.size >= 2 && pinchRef.current) {
      const [a, b] = [...pointersRef.current.values()];
      const nextZoom = clamp(
        pinchRef.current.zoom * distance(a, b) / Math.max(1, pinchRef.current.distance),
        MIN_ZOOM,
        MAX_ZOOM,
      );
      const nextMidpoint = midpoint(a, b);
      setZoom(nextZoom);
      setPan({
        x: pinchRef.current.pan.x + nextMidpoint.x - pinchRef.current.midpoint.x,
        y: pinchRef.current.pan.y + nextMidpoint.y - pinchRef.current.midpoint.y,
      });
      return;
    }
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const dx = point.x - drag.start.x;
    const dy = point.y - drag.start.y;
    if (Math.hypot(dx, dy) > 4) drag.moved = true;
    if (drag.mode === "pan") {
      setPan({ x: drag.startPan.x + dx, y: drag.startPan.y + dy });
    } else {
      setSelection(screenToImage(point));
    }
  }

  function pointerUp(event: ReactPointerEvent<HTMLDivElement>) {
    const point = localPoint(event);
    const drag = dragRef.current;
    if (drag && drag.pointerId === event.pointerId && !drag.moved && drag.mode === "pan") {
      setSelection(screenToImage(point));
      setMatches([]);
    }
    pointersRef.current.delete(event.pointerId);
    if (pointersRef.current.size < 2) pinchRef.current = null;
    if (drag?.pointerId === event.pointerId) dragRef.current = null;
  }

  function handleWheel(event: WheelEvent<HTMLDivElement>) {
    if (!imageUrl) return;
    event.preventDefault();
    setZoom((value) => clamp(value * Math.exp(-event.deltaY * 0.0015), MIN_ZOOM, MAX_ZOOM));
  }

  function updateCandidateLimit(value: number) {
    const next = clamp(Math.round(value), 1, 50);
    setCandidateLimit(next);
    localStorage.setItem("candidateDisplayLimitV2", String(next));
  }

  function closeImage() {
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    requestRef.current += 1;
    sourceCanvasRef.current = null;
    imageRef.current = null;
    setImageUrl(null);
    setSelection(null);
    setSelectedItem(null);
    setMatches([]);
    setStatus("上传截图以识别物品");
  }

  async function installToHomeScreen() {
    if (!installPrompt) {
      setShowInstallHelp(true);
      return;
    }
    await installPrompt.prompt();
    const choice = await installPrompt.userChoice;
    if (choice.outcome === "accepted") setIsInstalled(true);
    setInstallPrompt(null);
  }

  return (
    <main className="lens-app">
      <header className="app-bar">
        <button className="icon-button" type="button" disabled={!imageUrl} onClick={closeImage} aria-label="退出当前截图">×</button>
        <div className="app-title">
          <strong>Isaac Item Lens</strong>
        </div>
        <div className="bar-actions">
          {!isInstalled && (
            <button
              className="icon-button install-button"
              type="button"
              aria-label="添加到桌面"
              title="添加到桌面"
              onClick={installToHomeScreen}
            >
              ⇩
            </button>
          )}
          <label className="icon-button upload-button" aria-label={imageUrl ? "换一张图" : "上传截图"}>
            <input type="file" accept="image/*" onChange={handleFile} />
            {imageUrl ? "↻" : "+"}
          </label>
        </div>
      </header>

      {showInstallHelp && (
        <div className="item-detail-backdrop" role="presentation" onClick={() => setShowInstallHelp(false)}>
          <section
            className="install-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="install-title"
            onClick={(event) => event.stopPropagation()}
          >
            <button
              className="detail-close"
              type="button"
              aria-label="关闭安装说明"
              onClick={() => setShowInstallHelp(false)}
            >
              ×
            </button>
            <div className="install-mark">⇧</div>
            <h2 id="install-title">添加到主屏幕</h2>
            <ol>
              <li>点击 Safari 底部的“分享”按钮</li>
              <li>选择“添加到主屏幕”，然后点击“添加”</li>
            </ol>
            <p>添加后可像普通 App 一样从桌面打开，模型缓存完成后也可离线使用。</p>
          </section>
        </div>
      )}

      {selectedItem && (
        <div className="item-detail-backdrop" role="presentation" onClick={() => setSelectedItem(null)}>
          <section
            className="item-detail-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="item-detail-title"
            onClick={(event) => event.stopPropagation()}
          >
            <button
              className="detail-close"
              type="button"
              aria-label="关闭物品详情"
              onClick={() => setSelectedItem(null)}
            >
              ×
            </button>
            <div className="item-detail-heading">
              <img src={assetUrl(selectedItem.iconPath)} alt="" />
              <div>
                <h2 id="item-detail-title">{selectedItem.nameZh}</h2>
                <p>{selectedItem.nameEn}</p>
              </div>
            </div>
            {(selectedItem.type || (selectedItem.pools?.length ?? 0) > 0) && (
              <div className="detail-tags">
                {selectedItem.type && <span>{selectedItem.type}</span>}
                {selectedItem.pools?.slice(0, 5).map((pool) => <span key={pool}>{pool}</span>)}
              </div>
            )}
            {(selectedItem.pickup || selectedItem.description) && (
              <p className="detail-description">{selectedItem.pickup || selectedItem.description}</p>
            )}
            {(selectedItem.effects?.length ?? 0) > 0 && (
              <ul>
                {selectedItem.effects?.slice(0, 10).map((effect) => <li key={effect}>{effect}</li>)}
              </ul>
            )}
            {selectedItem.sourceName && <p className="detail-source">来源：{selectedItem.sourceName}</p>}
          </section>
        </div>
      )}

      <section
        ref={stageRef}
        className={`image-stage${imageUrl ? " has-image" : ""}`}
        aria-label="截图选取区域"
        onPointerDown={pointerDown}
        onPointerMove={pointerMove}
        onPointerUp={pointerUp}
        onPointerCancel={pointerUp}
        onWheel={handleWheel}
      >
        {imageUrl ? (
          <>
            <img
              className="source-image"
              src={imageUrl}
              alt="用户上传的游戏截图"
              draggable={false}
              style={{
                width: imageSize.width * renderedScale,
                height: imageSize.height * renderedScale,
                transform: `translate3d(${imageOrigin.x}px, ${imageOrigin.y}px, 0)`,
              }}
            />
            {selectionScreen && (
              <div
                className="selection-box"
                style={{
                  width: boxSize,
                  height: boxSize,
                  transform: `translate3d(${selectionScreen.x - boxSize / 2}px, ${selectionScreen.y - boxSize / 2}px, 0)`,
                }}
              >
                <span className="selection-check">✓</span>
              </div>
            )}
            <div className="canvas-hint">单指拖动 · 双指缩放 · 点击选取</div>
            {loadProgress !== null && loadProgress < 100 && (
              <div className="model-loading" role="status" aria-live="polite">
                <div className="model-loading-heading">
                  <strong>{loadLabel}</strong>
                  <span>{loadProgress}%</span>
                </div>
                <div
                  className="model-progress-track"
                  role="progressbar"
                  aria-valuemin={0}
                  aria-valuemax={100}
                  aria-valuenow={loadProgress}
                >
                  <span style={{ width: `${loadProgress}%` }} />
                </div>
                <small>首次运行需要下载约 46MB，完成后将离线缓存</small>
              </div>
            )}
          </>
        ) : (
          <label className="empty-upload">
            <input type="file" accept="image/*" onChange={handleFile} />
            <span className="viewfinder">⌗</span>
            <strong>上传截图以识别物品</strong>
            <small>图片和模型推理仅在本机处理</small>
          </label>
        )}
      </section>

      {imageUrl && (
        <section className="controls-bar">
          <label>
            <span>选框大小</span>
            <input
              type="range"
              min={MIN_BOX}
              max={Math.min(MAX_BOX, Math.max(80, stageSize.width * 0.72))}
              value={boxSize}
              onChange={(event) => setBoxSize(Number(event.target.value))}
            />
          </label>
          <button type="button" onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }}>复位</button>
        </section>
      )}

      <footer className="status-bar">
        <span className="status-dot" />
        <span>{status}</span>
        <span className="offline-label">{isSearching ? "处理中" : "离线"}</span>
      </footer>

      {selection && (
        <section className="candidate-panel" aria-label="相似对象">
          <div className="candidate-heading">
            <div className="selection-preview">
              <SelectionPreview source={sourceCanvasRef.current} selection={selection} side={boxSize / renderedScale} />
            </div>
            <div>
              <strong>选中区域</strong>
              <span>Top {candidateLimit} 相似对象</span>
            </div>
            {isSearching && <span className="spinner" aria-label="正在识别" />}
            <label className="limit-picker">
              <span>Top</span>
              <input
                type="number"
                min={1}
                max={50}
                value={candidateLimit}
                onChange={(event) => updateCandidateLimit(Number(event.target.value))}
              />
            </label>
            <button className="panel-close" type="button" onClick={() => { setSelection(null); setSelectedItem(null); setMatches([]); }}>×</button>
          </div>
          <div className="candidate-strip">
            {visibleMatches.map(({ item, score }) => (
              <button
                className="candidate-card"
                key={item.id}
                type="button"
                title={`${item.nameZh} ${item.nameEn}`}
                aria-label={`查看 ${item.nameZh} 详情，相似度 ${(score * 100).toFixed(1)}%`}
                onClick={() => setSelectedItem(item)}
              >
                <img src={assetUrl(item.iconPath)} alt={item.nameZh} />
                <span>{(score * 100).toFixed(1)}%</span>
              </button>
            ))}
            {!isSearching && visibleMatches.length === 0 && <p>没有找到相似对象</p>}
          </div>
        </section>
      )}
    </main>
  );
}

function SelectionPreview({
  source,
  selection,
  side,
}: {
  source: HTMLCanvasElement | null;
  selection: Selection;
  side: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !source) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const actualSide = Math.min(Math.min(source.width, source.height), side);
    const sx = clamp(selection.x - actualSide / 2, 0, source.width - actualSide);
    const sy = clamp(selection.y - actualSide / 2, 0, source.height - actualSide);
    ctx.clearRect(0, 0, 48, 48);
    ctx.drawImage(source, sx, sy, actualSide, actualSide, 0, 0, 48, 48);
  }, [source, selection, side]);
  return <canvas ref={canvasRef} width={48} height={48} />;
}

async function getModel(onProgress?: (progress: number, phase: "downloading" | "initializing") => void) {
  if (!modelPromise) {
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    modelPromise = fetchBinaryWithProgress(MODEL_URL, (progress) => onProgress?.(progress, "downloading")).then((model) => {
      onProgress?.(1, "initializing");
      return ort.InferenceSession.create(model, {
        executionProviders: ["wasm"],
        graphOptimizationLevel: "all",
      });
    });
  }
  return modelPromise;
}

async function fetchBinaryWithProgress(url: string, onProgress?: (progress: number) => void) {
  return new Promise<Uint8Array>((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("GET", url);
    request.responseType = "arraybuffer";
    request.timeout = 120_000;
    request.onprogress = (event) => {
      if (event.lengthComputable && event.total > 0) onProgress?.(Math.min(1, event.loaded / event.total));
    };
    request.onload = () => {
      if (request.status < 200 || request.status >= 300 || !(request.response instanceof ArrayBuffer)) {
        reject(new Error(`HTTP ${request.status}`));
        return;
      }
      onProgress?.(1);
      resolve(new Uint8Array(request.response));
    };
    request.onerror = () => reject(new Error("模型下载失败"));
    request.ontimeout = () => reject(new Error("模型下载超时"));
    request.send();
  });
}

async function getIndex() {
  if (!indexPromise) {
    indexPromise = Promise.all([
      fetch(METADATA_URL).then((response) => {
        if (!response.ok) throw new Error("metadata");
        return response.json() as Promise<IndexMetadata>;
      }),
      fetch(VECTOR_URL).then((response) => {
        if (!response.ok) throw new Error("vectors");
        return response.arrayBuffer();
      }),
      fetch(OBJECTS_URL).then((response) => {
        if (!response.ok) throw new Error("objects");
        return response.json() as Promise<IndexedObject[]>;
      }),
    ]).then(([metadata, buffer, objects]) => {
      const encoded = new Uint16Array(buffer);
      const vectors = new Float32Array(encoded.length);
      for (let index = 0; index < encoded.length; index += 1) vectors[index] = float16ToFloat32(encoded[index]);
      const details = new Map(objects.map((item) => [item.id, item]));
      return {
        metadata: {
          ...metadata,
          objects: metadata.objects.map((item) => ({ ...item, ...details.get(item.id) })),
        },
        vectors,
      };
    });
  }
  return indexPromise;
}

function cropToTensor(source: HTMLCanvasElement, sx: number, sy: number, side: number) {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 256;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  if (!ctx) throw new Error("canvas");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, sx, sy, side, side, 0, 0, 256, 256);
  const rgba = ctx.getImageData(0, 0, 256, 256).data;
  const plane = 256 * 256;
  const tensor = new Float32Array(plane * 3);
  for (let pixel = 0; pixel < plane; pixel += 1) {
    tensor[pixel] = rgba[pixel * 4] / 255;
    tensor[plane + pixel] = rgba[pixel * 4 + 1] / 255;
    tensor[plane * 2 + pixel] = rgba[pixel * 4 + 2] / 255;
  }
  return tensor;
}

function topMatches(query: Float32Array, metadata: IndexMetadata, vectors: Float32Array, count: number): Match[] {
  const scored = metadata.objects.map((item) => {
    let score = 0;
    const offset = item.vectorIndex * metadata.dimensions;
    for (let dimension = 0; dimension < metadata.dimensions; dimension += 1) {
      score += query[dimension] * vectors[offset + dimension];
    }
    return { item, score };
  });
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, count);
}

function normalize(values: Float32Array) {
  let norm = 0;
  for (const value of values) norm += value * value;
  const inverse = 1 / Math.max(1e-12, Math.sqrt(norm));
  for (let index = 0; index < values.length; index += 1) values[index] *= inverse;
}

function float16ToFloat32(value: number) {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >> 10) & 0x1f;
  const fraction = value & 0x03ff;
  if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
  if (exponent === 31) return fraction ? Number.NaN : sign * Number.POSITIVE_INFINITY;
  return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

function distance(a: Point, b: Point) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function midpoint(a: Point, b: Point): Point {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function assetUrl(path: string) {
  return `${BASE_URL}${path.replace(/^\/+/, "")}`;
}
