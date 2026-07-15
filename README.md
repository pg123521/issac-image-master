# Isaac Item Lens

《以撒的结合》离线截图识别原型。上传游戏截图、框选目标后，工具使用 MobileCLIP2-S0 在本地检索相似对象，并显示本地百科描述。

## 当前数据

- 943 个可检索对象
- 717 个道具、121 个饰品、105 张卡牌
- 百科数据来自 [IcaCat 以撒图鉴](https://issac-icecat.azurewebsites.net/) 与 [Platinum God](https://tboi.com/)
- 图标、百科 JSON 和 943 项向量索引均保存在仓库中
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
```

MobileCLIP 首次启动时会下载 `MobileCLIP2-S0` 的预训练权重。

## 启动

终端 1：启动本地向量检索服务。

```bash
.venv-train/bin/python scripts/mobileclip_item_search.py serve --port 8766 --top-k 10
```

终端 2：启动前端。

```bash
pnpm run dev
```

打开 `http://localhost:3000/`，上传截图后使用自动标注或手动补框。

## 主要文件

- `data/objects.en.json`：道具、饰品和卡牌百科清单
- `public/items/icons/`：IcaCat 道具图标
- `public/objects/icons/`：Platinum God 道具、饰品和卡牌图标
- `models/mobileclip-object-icon-index-v1.pt`：当前 943 项 MobileCLIP 向量索引
- `scripts/mobileclip_item_search.py`：索引构建、查询和本地服务
- `scripts/import_tboi_objects.py`：百科及 sprite 图标导入器

## 重建数据

重新抓取并裁剪 Platinum God / IcaCat 图标：

```bash
python3 scripts/import_tboi_objects.py
```

使用每个对象的原始百科图标重建索引：

```bash
.venv-train/bin/python scripts/mobileclip_item_search.py build-index \
  --output models/mobileclip-object-icon-index-v1.pt \
  --synthetic-per-item 0 \
  --batch-size 96
```

当前上线索引没有混入合成训练图。`data/training/`、旧分类模型、调试截图和本地构建缓存不会提交到 GitHub。

## 验证

```bash
pnpm run build
curl http://127.0.0.1:8766/health
```
