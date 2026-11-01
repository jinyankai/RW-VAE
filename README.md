# RW-VAE — 相关性重加权自编码器（本科毕设代码）

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

## 代码与四支柱的对应

| 支柱（PROPOSAL §三） | 代码 |
|---|---|
| 一 · 理论（相图 + 深度桥梁） | `rwvae/theory.py`（闭式 + 模拟器）、`scripts/verify_theory.py`（E3 数值验证） |
| 二 · 新架构 RW-VAE | `rwvae/gate.py`（`RelevanceRateGate` + `rate_allocation`）、`rwvae/relevance.py`（R1 去噪敏感度 / R2 语义对齐 + 方差/频率基线）、`rwvae/integration.py`（接 VA-VAE）、`scripts/train_rwvae.py`（E4）、`scripts/demo_rwvae_synthetic.py`（架构 PoC） |
| 二 · 受控干预（双射） | `rwvae/bijection.py`（`CouplingFlow`）、`scripts/eval_e2_bijection.py`（E2，验证引理 2） |
| 三 · 可访问性缺口 AG | `rwvae/metrics.py::accessibility_gap` / `linear_probe_risk`、`scripts/eval_e1_ag.py`（E1） |
| 四 · 廉价潜在指标 | `rwvae/metrics.py::effective_rank`（RankMe）/ `latent_power_spectrum` |

---

## 环境

- Python ≥ 3.10；依赖 `numpy`、`torch`（CPU 即可跑理论/指标/门）、`pytest`。
- 安装：`pip install -r requirements.txt`（可选 `pip install -e .`）。

> Windows 注意：numpy(MKL) 与 torch(libiomp) 可能冲突报 `OMP Error #15`。入口脚本与
> `conftest.py` 已设 `KMP_DUPLICATE_LIB_OK=TRUE` 作为标准规避（CPU 数值不受影响）。

---

## 快速开始

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

## 目录结构

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

## 架构：RW-VAE 率门

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

## 接入 LightningDiT（GPU 节点）

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

## 无 GPU 时的行为（本次交付）

- 理论验证、指标、门、E1/E2/架构 PoC 三个脚本、单元测试、训练 smoke **全部在 CPU 跑通**。
- `scripts/train_rwvae.py`（无 `--smoke`）检测到无 CUDA 时打印 GPU 运行说明并 `exit 0`，
  **不执行**真实训练（按毕设计划：写好代码、跳过 GPU 运行）。

---

## 许可

本仓库 MIT（见 `LICENSE`）。LightningDiT 亦为 MIT，但单独下载、不随本仓库分发。
