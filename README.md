# RW-VAE v2.1：冻结 tokenizer 上的相关性加权码率分配（本科毕设代码）

问题只有一个：对冻结的 ImageNet-256 tokenizer（LDM-f16d32，未对齐 DINO），在**相同名义码率**（期望 KL 预算）下，按 latent 方差分配信息还是按 DINOv2 语义相关性分配信息，会不会改变语义可解码性、重建误差和扩散生成质量。方法是闭式的：R2 解释协方差 → 加权 KLT → 固定码率反向注水 → 方差保持高斯信道 → LMMSE 解码前置层（必要时统一配方微调 decoder），再在信道输出上训练 LightningDiT-B/1。计划与预登记见 [`../auto-encoder/RWVAE_V2_CORE_PLAN.md`](../auto-encoder/RWVAE_V2_CORE_PLAN.md)，模块契约见 [`docs/ARCHITECTURE_V2.md`](docs/ARCHITECTURE_V2.md)。

v1（逐通道可学率门、双射干预、理论相图）保留为 L0 遗留代码，行为不变，说明在文末。

## 0. v3.1（当前主线）：可扩散性来自信息分配

v2.1 的有损信道删掉了细节，生成变差。v3 换成一个新 tokenizer：冻结 DINOv2-L 在深度上均匀取 8 层（第 3,6,…,24 层，各自 LayerNorm 后拼接，1×1 卷积）作语义支路，另加一条 RGB 卷积支路，两者按位置、按通道组由门控 α 混合，后验头输出 32×16×16 的高斯 latent。核心模块是**分噪声读出**：语义在强噪声（t=0.35）下读 DINO 特征，细节在弱噪声（t=0.80）下读 64×64 的 RGB 和边缘图。DiT 直接在标准化后的后验样本上训练，可选 REPA 语义 anchor；解码器从 VA-VAE 初始化，只做 L1 + LPIPS 重建。方案见 [`../auto-encoder/RWVAE_V3_ALLOCATION_PLAN.md`](../auto-encoder/RWVAE_V3_ALLOCATION_PLAN.md)。

| 文件 | 作用 |
|---|---|
| `rwvae/tokenizer_v3.py` | 编码器两支路、门控、后验头、读出头、KL / 锁码率拉格朗日、存取 |
| `rwvae/generation.py` | `train_dit(..., repa=REPAConfig, repa_target_fn=...)`：REPA 语义 anchor（v2 调用方式不变） |
| `rwvae/datasets.py` | `PosteriorLatentDataset`：每个 (epoch, index) 重新采样 z = μ + σε，REPA 时附带同视图原图 |
| `configs/v3.yaml`、`configs/v3_server.yaml` | v3 配置与实验臂预设 `v3_arms`（T-a…T-f、L-a、L-c） |
| `scripts/v3_*.py` | 训练 tokenizer → 缓存后验 → 训练 DiT → 评测 → 分配分析 |

一个实验臂的完整流程（A800 节点，`C` 为下列配置栈）：

```bash
C="--config configs/dataset_imagenet256.yaml --config configs/baseline_common.yaml --config configs/v3.yaml \
   --config configs/server_a800.yaml --config configs/v3_server.yaml"
torchrun --nproc_per_node 8 scripts/v3_train_tokenizer.py $C --arm T-c
torchrun --nproc_per_node 8 scripts/v3_cache_posterior.py $C --arm T-c      # 写 stats.json，含实测码率
torchrun --nproc_per_node 8 scripts/v3_train_dit.py $C --arm T-c [--repa]
python scripts/07_evaluate.py ref-stats $C --split val                     # 每个 experiment 只需一次
python scripts/v3_evaluate.py recon $C --arm T-c
python scripts/v3_evaluate.py generate $C --arm T-c [--repa] [--cfg-scale 1.5]
python scripts/v3_analysis.py $C --arm T-c                                  # 门控图、码率图、带噪探针曲线
# 锁码率对照：--arm L-c / L-a --set kl.target=<T-c 实测 R_hat 取整>
```

前置：`init.ckpt` 指向 hustvl 公开的 VA-VAE-f16d32 权重（`vavae-imagenet256-f16d32-dinov2.pt`）；拿不到时可临时指向 LDM-f16d32。

---

## 1. 代码地图

| 模块 | 职责 |
|---|---|
| `rwvae/runtime.py` | 配置合并与 `--set`、种子、日志、run 目录、`run_registry.jsonl`、sha256、`git_commit` / `git_dirty` |
| `rwvae/allocation.py` | [numpy f64] 加权 KLT、反向注水、equal_snr / variance / r2 三规则、A/B、闭式指标与恒等式残差 |
| `rwvae/relevance.py` | R2 解释协方差、CCA 谱、DINO↔latent 网格对齐扫描（v2 段在 v1 之后） |
| `rwvae/stats.py` | float64 分组流式矩（按图片 ID 分组，供折半/bootstrap）、`LatentStats` |
| `rwvae/channel.py` | `RWChannel`、可复现逐样本噪声 `noise_seed(base, epoch, index)` |
| `rwvae/decoder_adapter.py` | LMMSE 融合进 `post_quant_conv`、decoder 微调循环 |
| `rwvae/datasets.py` | manifest（train_fit / train_dev / val）、ADM 中心裁剪、safetensors 分片缓存、信道数据集 |
| `rwvae/evaluation.py` | PSNR、LPIPS、FID（`frechet_distance`）、线性探针、DCT 谱、bootstrap CI |
| `rwvae/generation.py` | LightningDiT 训练循环（DDP、EMA、续训）、ODE 采样 |
| `rwvae/integration.py` | 唯一接触 `third_party/LightningDiT`、timm、lpips、pytorch-fid 的模块 |
| `scripts/_common.py` | 脚本共用胶水（路径、配置、臂名、预登记检查） |

## 2. 阶段地图

默认输出目录是 `<run_root>/<experiment.name>/`（ImageNet-1k：`runs/imagenet256_ldm_f16d32`；SVP：`runs/svp`；本地：`runs/local_imagenette`）。完整参数见各脚本 docstring 和 `docs/ARCHITECTURE_V2.md` §4。

| 阶段 | 输入 | 主要输出 | 计划里程碑 |
|---|---|---|---|
| `00_verify_math.py` | 无（合成高斯） | `00_verify_math/verify_math.{json,md}` | M0 / M1：P1–P7 恒等式与蒙特卡洛核对，任一失败 exit 1 |
| `tools/make_svp_subset.py` | ImageNet train/val | SVP 子集 ImageFolder + `svp_subset.json` | SVP 专用 |
| `01_prepare_manifest.py` | train/val ImageFolder | manifest JSON、`DATA_CARD.md` §1 | M2 |
| `02_cache_latents.py` | manifest、tokenizer | `cache.dir/<split>/` 分片 + `cache_verification.json`、`DATA_CARD.md` §2 | M2 |
| `03_estimate_stats.py` | train_fit 缓存、DINOv2-L | `stats/{stats.npz, group_moments.pt, diagnostics.json}`、`DATA_CARD.md` §3 | M2（对齐门，失败 exit 2）/ M3 |
| `04_solve_channel.py` | stats、arms/grid | `allocations/*.npz`、`allocations_summary.{json,md}`（含 go/no-go） | M1 / M3 |
| `05_eval_recon_probe.py` | 缓存、allocations、tokenizer | `recon_probe/<split>/results.{json,md}`：PSNR、LPIPS、rFID、latent MSE、DCT、线性探针 | M3（R*、w* 只从这里选） |
| `05b_adapt_decoder.py` | allocation、图像 | `decoders/<arm>/decoder.safetensors` | M4 |
| `06_train_dit.py` | train_fit 缓存、stats、allocation | `dit/<arm>/{checkpoints, arm.json, metrics.jsonl}` | M5 / M6 |
| `07_evaluate.py ref-stats / generate / report` | 06 运行、图像 | `fid_ref/`、`gen/<arm>/step<k>_n<N>/results.json`、`report/REPORT.md` | M5 / M6 |

实验臂（`configs/arms.yaml`）：B0 = 后验均值、无信道（无码率约束参考，不进等码率比较）；B1 equal-SNR；B2 variance（w=1）；B3 r2（主方法）。筛查网格 `R ∈ {11, 22, 33, 44}` nats/token，`w ∈ {1.5, 2, 3}`。

## 3. 环境

```bash
# Linux A800 节点（torch 2.12.1+cu126，阿里云镜像）；先装 CUDA 版 torch，再装其它依赖，否则 pip 会换成 CPU 版
python -m venv .venv && source .venv/bin/activate
pip install torch==2.12.1 torchvision -f https://mirrors.aliyun.com/pytorch-wheels/cu126
pip install -e ".[gpu,dev]" pyarrow          # pyarrow 只给 parquet 转换工具用

# Windows 本机（RTX 4060，cu130）
.venv\Scripts\python.exe -m pip install torch==2.12.1 torchvision --index-url https://download.pytorch.org/whl/cu130
.venv\Scripts\python.exe -m pip install -e ".[gpu,dev]" pyarrow
```

- huggingface.co 不通时设 `HF_ENDPOINT=https://hf-mirror.com`；`configs/server_a800.yaml` 与 `local_imagenette.yaml` 的 `env` 块会在导入第三方库前写入 `HF_HOME`、`HF_ENDPOINT`、`TORCH_HOME`。
- LightningDiT 不进仓库，放在 `../third_party/LightningDiT`（或设 `RWVAE_LIGHTNINGDIT_ROOT`）。git clone 被代理拦时用 tarball：`curl -sSL https://codeload.github.com/hustvl/LightningDiT/tar.gz/refs/heads/main | tar -xz` 后改名。
- 资产：
  - tokenizer：`ldm-imagenet256-f16d32-50ep.ckpt`，来自 HF `hustvl/va-vae-imagenet256-experimental-variants`，路径写在 `tokenizer.ckpt`；
  - DINOv2-L、FID Inception、LPIPS-VGG：`python scripts/tools/prefetch_weights.py`，事先把 `HF_HOME`、`TORCH_HOME` 设成节点路径（脚本里的默认值是 Windows 路径）；
  - 终验另需 ADM evaluator 与 `VIRTUAL_imagenet256_labeled.npz`（OpenAI guided-diffusion）。
- Windows 上 numpy(MKL) 与 torch(libiomp) 的 `OMP Error #15` 已由 `_common.py` 和 `conftest.py` 设置的 `KMP_DUPLICATE_LIB_OK=TRUE` 规避。Windows 版 torch 2.12 起不了 torchrun，多卡只在 Linux。

## 4. SVP：第一次真实运行

SVP（sanity-validation pilot）把 ImageNet 的一个**随机小子集**推过 00 → 07 全流程：10 个随机类 × 100 张训练图，加这 10 类各 50 张官方 val（seed 2026），切成 train_fit 800 / train_dev 200 / val 500。生成器 LightningDiT-B/1 被**故意训到过拟合**这 800 个 latent（20k 步 × batch 256，约 6400 个 epoch，EMA 0.999）。这样每个指标都有明显的预期值，哪一环接错了一眼能看出来。它不是论文结果，不与 ImageNet-1k 混表。

```bash
# 默认读 /home/jinyankai/data/imagenet/{train,val}，子集写到 /home/jinyankai/data/svp/imagenet_c10_n100
bash scripts/run_pipeline.sh svp
bash scripts/run_pipeline.sh svp --train-root /data/imagenet/train --val-root /data/imagenet/val --svp-out /data/svp
FROM=06 bash scripts/run_pipeline.sh svp          # 从某阶段继续（已完成阶段的 run 目录会拒绝重写）
# PY（默认 .venv/bin/python）、NGPU（默认 8）、LOG_DIR（分片进程日志）可用环境变量改
```

配置栈：三个基础配置 + `server_a800.yaml` + `svp.yaml`。02/03 各起 8 个后台进程（`--num-procs 8 --proc-id k --device cuda:k`），之后 `--finalize` / `--merge`；06 用 `torchrun --nproc_per_node=$NGPU` 训练 B0、B2@R22、B3@R22 w2（均 `--exploratory`）；07 先做三份参考（train_fit 500 张、train_fit 特征含水平翻转、val 及特征），再对每个臂算 gFID 和最近邻，最后出报告 `runs/svp/report/REPORT.md`。

报告的 “SVP verdict” 一节逐条给出 `svp.pass_criteria` 的数值、阈值和 PASS/FAIL，阈值在运行前写死在 `configs/svp.yaml`：

| 判据 | 预期 | 阈值 |
|---|---|---|
| `dit_loss_drop` | 训练损失明显下降 | 末 1k 步均值 / 前 1k 步均值 ≤ 0.8 |
| `dit_loss_plateau` | 已收敛到平台 | 最后两个 2k 窗口相对变化 ≤ 3% |
| `gfid_memorisation` | 样本贴近训练子集 | gFID(train_fit 500) / gFID(val 500) ≤ 0.7 |
| `nn_memorisation` | 样本几乎是训练图的复制 | 样本→train_fit 的 L2 最近邻中位数 ≤ 0.8 × val→train_fit |
| `allocation_identity_residual` | 闭式数学精确 | 恒等式残差 ≤ 1e-8 |
| `latent_mse_vs_d_latent` | 经验失真吻合闭式 D_latent | 相对误差 ≤ 5% |
| `monotone_in_rate` | 码率越高越好 | PSNR、探针随 R 不降（0.05 dB / 1 点容差），latent MSE 不升 |
| `b0_reconstruction` | tokenizer 重建正常 | B0 PSNR ∈ [23, 32] dB，LPIPS ≤ 0.25 |
| `data_gates` | 缓存与对齐正确 | 各 split 重编码校验通过，对齐峰值在 (0,0) |

这些阈值是事先定下的工程判据，不是标定值。缺少输入的判据会列在 “Not evaluated”，不算通过。

## 5. 本地 Imagenette 工程测试（不是论文结果）

Imagenette 是 ImageNet 的 10 个类（train 9,469 / val 3,925），用真实 tokenizer、DINOv2-L、LightningDiT 和 FID 代码，但预算极小。结果只用来确认代码在真实资产上能跑，报告抬头会标成工程测试。

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_local_imagenette.ps1
powershell -ExecutionPolicy Bypass -File scripts\run_local_imagenette.ps1 -From 06
```

配置栈：三个基础配置 + `configs/local_imagenette.yaml`（单进程；05 只评 R=22、w=2；06 训 200 步 batch 16；07 用 val 1000 张做参考）。

## 6. ImageNet-1k 正式流程（GPU 节点）

1. **数据**：HF `ILSVRC/imagenet-1k`（需在 HF 上同意许可；国内走 hf-mirror）下载 parquet，再转成 ImageFolder：
   `python -I scripts/tools/imagenet_parquet_to_folder.py <snapshot_dir> /home/jinyankai/data/imagenet --split train --split validation --procs 32`
   （原始字节直写、不重编码；可断点续跑；val 按 wnid 分目录）。路径写进 `configs/server_a800.yaml`。
2. **00 → 05**：`bash scripts/run_pipeline.sh imagenet`。02/03 分 8 进程并行再合并；05 在 train_dev 上跑全网格。脚本在 06 之前**主动停下**。
3. **预登记**：读 `recon_probe/train_dev/results.md` 和 `allocations/allocations_summary.md`，只按这些无生成器指标选 R*、w*，在 `configs/arms.yaml` 里设 `preregistration.frozen: true`、`R_star`、`w_star` 并提交。没冻结时 06 会拒绝有损臂，除非显式 `--exploratory`（registry 会标记）。
4. **06**：`torchrun --standalone --nproc_per_node=8 scripts/06_train_dit.py <configs> --arm B0`，再训 B1/B2/B3 于 (R*, w*)。
5. **07**：`ref-stats --split val`；`generate` 默认 FID-10K，只用于筛查；终验用 `--num-samples 50000 --save-npz` 得到 FID-50K，并用 ADM evaluator 在 `samples.npz` 上复核；`report` 汇总。终报再跑一次 `05 --split val`。

## 7. 测试

```bash
python -m pytest -q                                   # 默认套件：CPU、合成数据，真实资产测试自动跳过
python -m pytest -q tests/test_pipeline_smoke.py      # SVP 全链路冒烟（约 15–20 s）
RWVAE_ASSETS=E:/GraduateDesign/assets python -m pytest -q -m real_assets   # 真实权重/Imagenette，需要 CUDA
```

脚本测试（`tests/test_scripts_*.py`）在进程内调用各脚本的 `main()`，用随机初始化的微型 AutoencoderKL、假 DINO/Inception/LPIPS 和 depth-2 的 DiT；它们把 registry 指到临时文件，不碰 `manifests/run_registry.jsonl`。

## 8. 可复现规则

- 所有随机性由显式种子决定：训练噪声 `train.noise_seed=2026`、采样初始噪声 `sample.noise_seed=9001`、评测 1234、探针 4321、decoder 微调 7。各臂共用同一份 `mu` 缓存、同一归一化和同一组初始噪声。
- 每次运行在 `manifests/run_registry.jsonl` 追加一行：配置指纹、`git_commit`、`git_dirty`、环境、摘要。`git_dirty: true` 说明工作区有未提交改动或未忽略的未跟踪文件，这次运行无法只凭 commit 复现；正式运行前先提交。
- 产物 meta 记录输入的 sha256/指纹、split、种子、dtype、TF32 设置；缓存、统计、评测一律关 TF32。
- **禁止按 gFID 选 R 或 w**。R*、w* 只从 05 的无生成器指标选，并在 06 之前冻结。
- **FID 种类不混表**：FID-10K（筛查）、FID-50K + ADM（终验）、FID-N（SVP 等小样本）分表；rFID 在 n < 2048 时标为 “small-n, indicative only”。
- 本地 Imagenette 和 SVP 的数字都不进论文的 ImageNet-1k 表。

## 9. 已知限制

- `LatentCacheWriter` 的 safetensors 元数据键顺序不固定，同样内容的分片 sha256 会变；重建缓存会改变 06 的 cache_index 身份（续训会拒绝）。
- 分片 meta 含 git commit：换了 commit 再续写半截缓存会报错（有意为之的严格）。
- 07 generate 只能单进程采样；FID-50K 在单张 A800 上较慢。`--save-npz` 在 50k 时约占 9.8 GB 内存。
- 06 在 8 卡 NCCL 下尚未实测（本机 Windows 起不了 torchrun）；LightningDiT 的 `torch.compile` 路径本地也没跑过。
- 03 的 `--max-images` / `stats.max_images` 取 manifest 的前 N 行，而 manifest 按类排序，截断会偏向前几个类；只用于调试，正式统计不要截断。
- 首步梯度范数约 2.7e7（余弦损失 + 零初始化输出），`grad_clip` 为 null 时会抬高 Adam 二阶矩若干百步，可能影响 SVP 的 `dit_loss_drop` 窗口。
- 逐 token 信道只保证逐 token 统计；跨 token 的谱变化靠 05 的 DCT 诊断去看，不做 DCT 信道。

## 10. 有意不实现（计划 §9 候选池）

DCT 频带信道与频带注水、可学率门、R1 生成敏感度、R3 标签度量、isotropic / top-k / placebo / decoder-sens 臂、GAN 感知损失适配、CFG-gFID / CMMD / sFID / IS / PR、第二 tokenizer（已对齐 VA-VAE）、DiT-XL、encoder 联合适配与跨数据集。`B_pub` 用上游 LightningDiT 脚本复现，不在本仓库重写。

---

## v1（L0 legacy）

以下是 v1 的原 README，原样保留（只把标题降了一级）。v1 的 29 个测试仍在默认套件里。其中的路径和状态描述写于 2026-11-01，v1 计划见 `../auto-encoder/PROPOSAL.md`。

### RW-VAE — 相关性重加权自编码器（本科毕设代码）

**Relevance-Weighted Autoencoder for image generation** — 配套开题报告
`../auto-encoder/PROPOSAL.md`（v0.3）的实验/架构代码。

核心论断：重建训练按**方差**分配潜在容量，而生成最优编码应按**相关性**分配。本仓库
提供 (1) 率失真相图理论的**数值验证**、(2) 新架构 **RW-VAE** 的核心模块（latent 上的
逐通道"率门"）、(3) 评估指标、(4) 接入 LightningDiT 的训练入口（GPU 部分已写好但在无
GPU 环境下自动跳过）。

当前状态（2026-11-01，CPU 环境）：**理论验证 25/25 通过，单元测试 29/29 通过，E1/E2/架构
PoC 三个脚本 CPU 全通过，训练代码路径 CPU smoke 通过**。GPU 训练（ImageNet-256 + DiT-XL +
gFID）代码已就绪、需在 GPU 节点运行。

---

### 代码与四支柱的对应

| 支柱（PROPOSAL §三） | 代码 |
|---|---|
| 一 · 理论（相图 + 深度桥梁） | `rwvae/theory.py`（闭式 + 模拟器）、`scripts/verify_theory.py`（E3 数值验证） |
| 二 · 新架构 RW-VAE | `rwvae/gate.py`（`RelevanceRateGate` + `rate_allocation`）、`rwvae/relevance.py`（R1 去噪敏感度 / R2 语义对齐 + 方差/频率基线）、`rwvae/integration.py`（接 VA-VAE）、`scripts/train_rwvae.py`（E4）、`scripts/demo_rwvae_synthetic.py`（架构 PoC） |
| 二 · 受控干预（双射） | `rwvae/bijection.py`（`CouplingFlow`）、`scripts/eval_e2_bijection.py`（E2，验证引理 2） |
| 三 · 可访问性缺口 AG | `rwvae/metrics.py::accessibility_gap` / `linear_probe_risk`、`scripts/eval_e1_ag.py`（E1） |
| 四 · 廉价潜在指标 | `rwvae/metrics.py::effective_rank`（RankMe）/ `latent_power_spectrum` |

---

### 环境

- Python ≥ 3.10；依赖 `numpy`、`torch`（CPU 即可跑理论/指标/门）、`pytest`。
- 安装：`pip install -r requirements.txt`（可选 `pip install -e .`）。

> Windows 注意：numpy(MKL) 与 torch(libiomp) 可能冲突报 `OMP Error #15`。入口脚本与
> `conftest.py` 已设 `KMP_DUPLICATE_LIB_OK=TRUE` 作为标准规避（CPU 数值不受影响）。

---

### 快速开始

```bash
# 1) 理论数值验证（E3，纯 numpy，几秒，无需 GPU）
#    模拟线性高斯模型并与闭式对照：主命题 (a)(b)(c) + 桥 4 的 rho_emp 恒等式
python scripts/verify_theory.py
#    -> 25/25 checks passed -> ALL PASS

# 2) 单元测试
python -m pytest -q
#    -> 22 passed

# 3) E2 双射干预（验证引理 2：重建不变、线性可用性坍塌、信息保留；CPU 真实实验）
python scripts/eval_e2_bijection.py

# 4) E1 可访问性缺口 AG 分解（恒等式 + 合成 demo；CPU）
python scripts/eval_e1_ag.py

# 5) 架构 PoC：在理论模型上训练真实率门，展示相关性重排把码率搬到任务子空间（CPU）
python scripts/demo_rwvae_synthetic.py

# 6) 训练代码路径 CPU 冒烟测试（门前向/反向 + 相关性信号 + 损失装配，无需 GPU/数据）
python scripts/train_rwvae.py --smoke

# 7) 真实训练（需 CUDA；无 GPU 时自动打印说明并以 exit 0 跳过）
python scripts/train_rwvae.py
```

`verify_theory.py` 验证的内容（节选，实测通过）：
- 主命题 (a)：PCA 相对重建误差与任务 R² 的模拟值匹配闭式；**k=d_n 时误差→0 而 R²=0** 的病态成立；
- 主命题 (b)：反向注水求解器 ≡ 闭式失真；R≤R* 时信号码率为 0（I(Z;S)=0）；R=R* 处失真 = (d_s+d_n)/(d_s+d_n·ρ)；R>R* 信号开始获码率；
- 主命题 (c)：加权 PCA 阈值 ρ*=w²（ρ<w² 恢复任务、ρ>w² 仍丢失）；加权率失真阈值 (d_n/2)ln(ρ/w²)；
- 桥 4：`ρ_emp = (d_n/d_s)·ρ` 的蒙特卡洛估计与恒等式吻合。

---

### 目录结构

```
rwvae/                         # 本仓库（git）
├── rwvae/                     # 包
│   ├── theory.py              # 相图闭式 + 模拟器（numpy）
│   ├── gate.py                # RelevanceRateGate：逐通道率门 + rate_allocation（新架构核心）
│   ├── relevance.py           # 分配信号：R1 去噪敏感度 / R2 语义对齐 + 方差/频率基线
│   ├── bijection.py           # CouplingFlow：可逆流（E2 受控干预）
│   ├── metrics.py             # AG 分解、RankMe/有效秩、潜在功率谱
│   └── integration.py         # GatedVAVAE：接 LightningDiT 的 VA-VAE（仅 GPU）
├── scripts/
│   ├── verify_theory.py       # E3 理论数值验证（可直接运行）
│   ├── eval_e2_bijection.py   # E2 双射干预（CPU 真实实验）
│   ├── eval_e1_ag.py          # E1 可访问性缺口 AG 分解（CPU）
│   ├── demo_rwvae_synthetic.py# 架构 PoC：率门在理论模型上端到端重排（CPU）
│   └── train_rwvae.py         # E4 训练入口（--smoke CPU；真实训练需 GPU）
├── tests/                     # pytest
├── README.md  requirements.txt  pyproject.toml  LICENSE  .gitignore
```

第三方的 LightningDiT **不纳入本仓库**，放在同级 `../third_party/LightningDiT/`（见下）。

---

### 架构：RW-VAE 率门

在标准连续 VAE/tokenizer 的 latent `z`（VA-VAE f16d32 → `[B,32,16,16]`）上加一个对角、
可学的逐通道门 `g∈R^32`：`z' = g ⊙ z`。这就是主命题 (c) 的对角 `W=diag(g)`。两件事使其成为
**重排**而非加容量：

- `budget_penalty()` 把总门质量 `Σg` 约束到固定预算 → 固定总率 → **rFID 与无门基线匹配**；
- `relevance_alignment_loss(relevance)` 把门分布拉向**生成相关性**向量 → 把容量从"方差序"
  推向"相关性序"。相关性信号两种，做消融：R1 去噪/生成敏感度（主、最新颖），R2 DINOv2 对齐（基线）。

训练损失（见 `scripts/train_rwvae.py::rwvae_loss`）：`重建(gated latent) + λ_budget·预算 + λ_rel·相关性对齐`。

**分配信号家族与分离论证**：同一个门可由不同的逐通道信号驱动，用于证明"生成相关性 ≠ 方差 ≠
频率"——`rwvae/relevance.py` 提供 `denoising_sensitivity`(R1)、`alignment_relevance`(R2)、
以及基线 `variance_allocation`（重建偏好的方差序）、`frequency_allocation`（FA-VAE/FreqWarm 频率轴）。
`gate.rate_allocation()` 给出每通道占固定码率预算的比例，是"潜在组织方式"的核心读数。

**架构 PoC（`demo_rwvae_synthetic.py`，已在 CPU 跑通）**：在理论的线性高斯模型上，固定总码率，
重建驱动分配把码率灌到高方差干扰子空间（usability≈0.01），相关性驱动分配把 98.5% 码率搬到任务
子空间（usability≈0.93）——端到端地证实"重排潜在组织 → 改变可用性"（线性模型里以更高失真为代价；
图像中解码器的冗余使 rFID 仍可匹配）。

**受控干预 E2（`eval_e2_bijection.py`）**：对冻结 latent 施加可逆 `CouplingFlow`，重建严格不变
（可逆性误差 ~1e-6），对抗性细化后线性探针 R² 由 1.00 压到 0.00、而 MLP 探针仍 0.63——即引理 2
"重建不变、线性可用性可被推向随机、信息保留"。

---

### 接入 LightningDiT（GPU 节点）

本环境的 `git clone` 被代理拦截（CONNECT 502），改用 tarball 下载并解压到 `third_party/`：

```bash
mkdir -p ../third_party && cd ../third_party
curl -sSL -o LightningDiT.tar.gz \
  "https://codeload.github.com/hustvl/LightningDiT/tar.gz/refs/heads/main"
tar -xzf LightningDiT.tar.gz && mv LightningDiT-main LightningDiT && rm LightningDiT.tar.gz
# 若网络允许，等价：git clone --depth 1 https://github.com/hustvl/LightningDiT.git
pip install -r LightningDiT/requirements.txt   # 在 GPU 节点
```

LightningDiT（MIT）打包了 VA-VAE + XL 级 DiT + latent 预缓存 + ADM FID 评测。`GatedVAVAE`
（`rwvae/integration.py`）把率门插在 `VA_VAE.encode_images()` 的输出上。

**训练注意**：LightningDiT 目前只开源 VA-VAE **推理**（编码器/解码器训练"later"）。端到端训练
率门有两条路：(a) 在**已发布 VA-VAE 权重**上微调；(b) 用 **REPA-E 的可训练 VAE**（其模块化 YAML
损失）并在 latent 处插入本率门。两种情况下 `gate.py`/`relevance.py`/`rwvae_loss` 不变。

**DiT-XL 受控协议**（匹配 rFID、只改 latent）：① 训小 VAE+门、调预算使 rFID 匹配无门基线；
② 预缓存 latent；③ 在 baseline-latent 与 gated-latent 上用**同配方同步数**训 LightningDiT-XL
（从 released 权重 warm-start 省算力）；④ 用 ADM evaluator + 参考 `.npz` 评 gFID。不能冻结复用
旧生成器（门改变了 latent 分布）。gFID 需另取 OpenAI `guided-diffusion` 的参考统计 `.npz`。

---

### 无 GPU 时的行为（本次交付）

- 理论验证、指标、门、E1/E2/架构 PoC 三个脚本、单元测试、训练 smoke **全部在 CPU 跑通**。
- `scripts/train_rwvae.py`（无 `--smoke`）检测到无 CUDA 时打印 GPU 运行说明并 `exit 0`，
  **不执行**真实训练（按毕设计划：写好代码、跳过 GPU 运行）。

---

### 许可

本仓库 MIT（见 `LICENSE`）。LightningDiT 亦为 MIT，但单独下载、不随本仓库分发。
