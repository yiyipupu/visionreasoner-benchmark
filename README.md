# VisionReasoner — 自有数据测试交付

完整官方源码保留在原目录，新增适配代码仅在 `delivery/`。默认针对单张 RTX 4090（24GB），**尚未在 4090 上实测**。支持检测、分割、计数和 VQA；自有数据尚未验证。

## 1. 克隆

```bash
git clone https://github.com/yiyipupu/visionreasoner-benchmark.git
cd visionreasoner-benchmark
```

## 2. 前置条件

Linux、Conda、可用 NVIDIA 驱动。使用 PyTorch CUDA 11.8 wheel，编译 SAM2 时需要兼容的 CUDA Toolkit。4090 支持 BF16。先确认 `nvidia-smi` 正常。

## 3. 环境和缓存

```bash
export VISIONREASONER_ROOT=/data/yukun/VisionReasoner
mkdir -p "$VISIONREASONER_ROOT"/{env,cache,checkpoints,datasets,outputs,weights}
conda create -p "$VISIONREASONER_ROOT/env" python=3.12 -y
conda activate "$VISIONREASONER_ROOT/env"
export HF_HOME="$VISIONREASONER_ROOT/cache/huggingface"
export TORCH_HOME="$VISIONREASONER_ROOT/cache/torch"
export PIP_CACHE_DIR="$VISIONREASONER_ROOT/cache/pip"
export TMPDIR="$VISIONREASONER_ROOT/cache/tmp"
mkdir -p "$TMPDIR"
python -m pip install --upgrade pip setuptools wheel packaging ninja
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r delivery/requirements.txt --no-build-isolation
python -m pip check
```

后续新终端需重新激活环境并设置上述环境变量。核心依赖固定，辅助依赖未完全锁定，安装后建议 `pip freeze > "$VISIONREASONER_ROOT/outputs/environment.txt"`。

## 4. 下载权重

```bash
python delivery/download.py
```

下载主模型 Ricky06662/VisionReasoner-7B、TaskRouter-1.5B 和 facebook/sam2-hiera-large 到 weights。中断后重跑同一命令，保留下载目录及其缓存以便复用；部分传输可能重新开始。本入口明确指定任务，不加载路由模型，仍提供其权重供官方入口使用。模型下载使用仓库默认 revision；下载目录缓存元数据保留解析的 revision，请在正式实验前另行固定权重 revision。本交付不声称该权重是论文所有实验的最优 checkpoint，ViSurf 等后续 checkpoint 未混用。

## 5. 公开样例单样本

```bash
python delivery/run.py delivery/example.jsonl --validate
CUDA_VISIBLE_DEVICES=0 python delivery/run.py delivery/example.jsonl
cat "$VISIONREASONER_ROOT/outputs/results/predictions.jsonl"
```

成功标志：status 为 ok，输出 count；公开样例参考数量为 10。原始回答保存在 full_response。计数不生成可视化图。分割额外保存每条样本的二值 PNG mask。

## 6. 自有数据接口

这是新增统一 JSONL 接口，不是官方 Hugging Face 数据集格式。支持手工生成 JSONL；不提供 COCO/RLE、多边形等任意格式的自动转换，需先转换为下述格式。

```text
datasets/custom/
  samples.jsonl
  images/a.png
  masks/a.png
```

每行一个样本：

```json
{"id":"seg-a","image":"images/a.png","query":"please segment the person","task":"segmentation","mask":"masks/a.png"}
```

必填：唯一字符串 id、相对 manifest 的 image、非空 query、task。task 为 segmentation/detection/counting/vqa。计数可选 count（非负整数），VQA 可选 answer，检测可选 bboxes（原图 xyxy）；后两者当前仅推理，不自动评估。

mask 仅分割评估需要：单通道 PNG，与原图宽高一致，0 背景、1 目标、255 ignore。多实例必须合并为该问题要求的目标并集；不支持直接输入类别 ID 图。需自行按目标类别提取为 1，其他类别为 0，保留 ignore 为 255。每个问题与一个目标并集 mask 对应；同图不同问题应创建不同 id。无目标图全部为 0。

```bash
python delivery/run.py "$VISIONREASONER_ROOT/datasets/custom/samples.jsonl" --validate
```

此命令校验文件、任务、重复 ID、mask 尺寸和值域；不验证标注语义是否正确。

## 7. 批量推理

```bash
CUDA_VISIBLE_DEVICES=0 python delivery/run.py "$VISIONREASONER_ROOT/datasets/custom/samples.jsonl" --output "$VISIONREASONER_ROOT/outputs/custom"
```

模型仅加载一次，逐样本复用。每行结果含 id/status；异常保存在 errors.log。初始化错误直接退出。重复运行会覆盖同一输出目录的记录，使用新目录保留实验。原模型部分方法会捕获内部异常，本入口额外将空响应判为失败，但无法保证识别所有上游静默错误。

## 8. 评估

```bash
python delivery/run.py "$VISIONREASONER_ROOT/datasets/custom/samples.jsonl" --output "$VISIONREASONER_ROOT/outputs/custom" --evaluate
```

分割：成功且有标注样本的逐样本 IoU 均值，ignore 不参与交并集，双方均空记 1，只有一方非空记 0，全部 ignore 不计分。计数：成功且有 count 标注的绝对误差均值 MAE。失败样本不纳入指标，但明确报告 failed、missing 和实际计分数量，比较实验须同时看覆盖率。该指标不是 COCO AP，也不是官方基准全部指标。检测/VQA 未实现通用评分，按任务选择官方 evaluation/ 中相应脚本或另写数据专用评价，不能用 IoU/MAE 代替它们。

## 9. 常见问题

- 下载失败：保留缓存重跑；检查磁盘和网络，不需要下载论文完整训练集。
- OOM：确认 4090 无其他任务；本入口单卡运行，SAM2 仅分割任务加载。若分割仍溢出，需进一步测量后调整部署，不能保证所有输入均可运行。
- 缺依赖：确认 conda 环境、pip check；不建议安装官方未锁定 requirements 全套训练依赖。
- 路径/标注错误：先 --validate；mask 必须原图大小，不能输入 RGB 彩色标注。
- SAM2：本地 .pt 使用 build_sam2；官方 from_pretrained 仅识别指定模型 ID。

## 10. 来源、数据与验证状态

论文：https://arxiv.org/abs/2505.12081

官方：https://github.com/JIA-Lab-research/VisionReasoner

固定源码提交：c013132276d85a3a3a4f03a6cd78d3e901e0128d。原说明完整保留于 UPSTREAM_README.md，许可证见 LICENSE（Apache-2.0）；内嵌第三方代码遵守各自许可。官方原目录保持该提交内容，交付入口通过复用其方法适配本地权重和单卡部署。

与原始实现差异：原默认 BF16 + FlashAttention2；本入口保留 BF16，改用 SDPA，避免额外 flash-attn 编译。取消旧部署的多卡设置，显式单卡。明确 task、不加载 TaskRouter；只在分割时加载 SAM2。图片仍按官方 840×840 缩放、坐标还原及原提示模板执行。多个实例掩码仍按官方逻辑合并。

官方 README 的评测集：RefCOCO/RefCOCO+/RefCOCOg（验证及 testA/testB/test）、ReasonSeg（val/test）、COCO val、PixMo counting validation/test、CountBench。来源及具体命令见 UPSTREAM_README.md Evaluation；相关公开打包数据：https://huggingface.co/Ricky06662 。训练代码官方指向 https://github.com/dvlab-research/Seg-Zero 。**本次尚未完成论文训练集构成及各 split 用途的逐项核验，不应把上述评测集全部当成训练集**。自有数据测试不要求下载这些完整数据集。

当前检查找到既有部署分割结果文件，但未以此宣称复现成功。本次执行环境无法连接 NVIDIA 驱动，因此没有完成 GPU 端到端推理，4090 未实测，真实私有数据未测试。语法/公开样例接口检查与真实模型推理必须区分。后续发布使用者应记录 GPU、环境 freeze、权重 revision 和完整命令。
