# Isaac Item Lens

《以撒的结合》离线截图识别原型。上传游戏截图后，工具使用本地目标检测器标出房间内可拾取道具，再通过 MobileCLIP2-S0 检索相似对象并显示本地百科描述。漏检时仍可手动框选。

## 当前数据

- 943 个可检索对象
- 717 个道具、121 个饰品、105 张卡牌
- 百科数据来自 [IcaCat 以撒图鉴](https://issac-icecat.azurewebsites.net/) 与 [Platinum God](https://tboi.com/)
- 图标、百科 JSON、微调视觉权重和 943 项向量索引均保存在仓库中
- 房间检测器不检测 HUD、角色、敌人、心、硬币、炸弹、钥匙或药丸
- 截图不会上传到远程服务

## 环境

- Node.js 22.13+
- pnpm
- Python 3.12+
- PyTorch、OpenCLIP 和 Pillow

```bash
pnpm install
python3 -m venv .venv-train
.venv-train/bin/pip install -r requirements-mobileclip.txt
.venv-train/bin/pip install -r requirements-detector.txt
```

MobileCLIP 首次启动时会下载 `MobileCLIP2-S0` 的预训练权重。

## 启动

终端 1：启动本地向量检索服务。

```bash
.venv-train/bin/python scripts/mobileclip_item_search.py serve --port 8766 --top-k 10
```

终端 2：启动房间道具检测服务。

```bash
.venv-train/bin/python scripts/room_detector_service.py --port 8767
```

终端 3：启动前端。

```bash
pnpm run dev
```

打开 `http://localhost:3000/`。上传截图后会自动检测房间道具，漏检时可使用“框选道具”。

## 主要文件

- `data/objects.en.json`：道具、饰品和卡牌百科清单
- `public/items/icons/`：IcaCat 道具图标
- `public/objects/icons/`：Platinum God 道具、饰品和卡牌图标
- `models/mobileclip-partial-v1.pt`：针对残缺图标微调的 MobileCLIP 视觉权重
- `models/mobileclip-object-partial-index-v1.pt`：当前 943 项微调向量索引
- `models/mobileclip-object-icon-index-v1.pt`：未微调基线索引，用于前后评测
- `models/room-collectible-detector-v1.pt`：房间内可拾取道具单类别检测模型
- `scripts/mobileclip_item_search.py`：索引构建、查询和本地服务
- `scripts/train_mobileclip_partial.py`：残缺图标评测与 MobileCLIP 对比微调
- `scripts/generate_room_detector_dataset.py`：从透明 icon 和真实房间背景生成检测数据
- `scripts/train_room_detector.py`：YOLO26n 检测训练与真实/合成评测
- `scripts/room_detector_service.py`：重叠切片检测和本地 HTTP 服务
- `scripts/import_tboi_objects.py`：百科及 sprite 图标导入器

## 重建数据

重新抓取并裁剪 Platinum God / IcaCat 图标：

```bash
python3 scripts/import_tboi_objects.py
```

使用每个对象的原始百科图标和当前微调权重重建索引：

```bash
.venv-train/bin/python scripts/mobileclip_item_search.py build-index \
  --output models/mobileclip-object-partial-index-v1.pt \
  --batch-size 96
```

图库索引只编码百科原始完整图标；残缺、换背景等训练增强在内存中实时生成，不写入磁盘。旧 CNN 分类器、旧合成训练集、调试截图和本地构建缓存均未保留。

## 房间道具检测

生成单类别检测数据。训练集使用前三张真实截图的地面区域，后三张截图只用于验证：

```bash
.venv-train/bin/python scripts/generate_room_detector_dataset.py \
  --train-count 600 --val-count 120
```

训练 1024 输入的 nano 检测器：

```bash
.venv-train/bin/python scripts/train_room_detector.py train \
  --epochs 24 --batch-size 4 --image-size 1024
```

单独复测合成集和真实截图：

```bash
.venv-train/bin/python scripts/train_room_detector.py evaluate \
  --weights models/room-collectible-detector-v1.pt
```

按 App 的整图与重叠切片路径验收 6 张真实截图，并生成逐图画框结果：

```bash
.venv-train/bin/python scripts/evaluate_room_detector_tiled.py \
  --weights models/room-collectible-detector-v1.pt
```

检测服务会对超宽截图同时执行整图和重叠方形切片推理，再将检测框映射回原图并去重。上传后的原图只发送给本机 `127.0.0.1:8767`。

## 残缺图标微调

先评测原始 MobileCLIP 在不同可见比例下的召回率：

```bash
.venv-train/bin/python scripts/train_mobileclip_partial.py evaluate \
  --samples-per-object 1
```

微调直接更新 MobileCLIP 视觉塔末端，不使用 Adapter，也不会把增强图片保存到磁盘：

```bash
.venv-train/bin/python scripts/train_mobileclip_partial.py train \
  --output models/mobileclip-partial-v1.pt
```

训练后评测并重建图库索引：

```bash
.venv-train/bin/python scripts/train_mobileclip_partial.py evaluate \
  --weights models/mobileclip-partial-v1.pt

.venv-train/bin/python scripts/mobileclip_item_search.py build-index \
  --weights models/mobileclip-partial-v1.pt \
  --output models/mobileclip-object-partial-index-v1.pt
```

使用微调模型启动服务时，权重和索引必须配套：

```bash
.venv-train/bin/python scripts/mobileclip_item_search.py serve \
  --weights models/mobileclip-partial-v1.pt \
  --index models/mobileclip-object-partial-index-v1.pt \
  --port 8766 --top-k 10
```

## 验证

```bash
pnpm run build
curl http://127.0.0.1:8766/health
curl http://127.0.0.1:8767/health
```
