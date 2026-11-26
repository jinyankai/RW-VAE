# RW-VAE v2.1 代码架构与模块契约（Code Contracts）

> 对应计划：`../auto-encoder/RWVAE_V2_CORE_PLAN.md`（v2.1，2026-11-17）。
> 本文件是**实现契约**：每个模块的职责、接口签名、数学不变量、错误语义与必须的测试；已与 `rwvae` 0.2.0 的代码逐项核对。
> 代码注释/docstring 用英文；本文件与 README 用中文。v1 代码（`gate.py`、`bijection.py`、
> `theory.py`、`metrics.py`、`relevance.py` 的 v1 部分、`integration.GatedVAVAE`）保留为 L0 遗留，不删除、不改行为。

---

## 0. 分层与依赖

```
            scripts/00..07  (CLI：读 config → 调库 → 写带元数据的产物 → 记 run_registry)
                 │
 ┌───────────────┼────────────────────────────────────────────────────────┐
 │ rwvae/ (library, importable & unit-tested)                             │
 │  runtime.py        config / seed / logging / run dir / registry / hash  │
 │  allocation.py     [numpy f64] KLT、注水、三规则、A/B、闭式指标（P2–P7） │
 │  relevance.py      [numpy f64] R2 解释协方差、K、CCA 谱（P1）+ 对齐扫描  │
 │  stats.py          [torch f64] 分组流式矩、LatentStats、稳定性工具       │
 │  channel.py        [torch] RWChannel、可复现逐样本噪声（P4/P6）           │
 │  decoder_adapter.py[torch] LMMSE 前置层/融合首层、decoder 微调循环        │
 │  datasets.py       [torch] manifest、图像集、mu 缓存读写、信道数据集      │
 │  evaluation.py     [torch] PSNR/LPIPS/FID/线性探针/DCT 谱/bootstrap       │
 │  generation.py     [torch] LightningDiT 训练循环、EMA、采样               │
 │  integration.py    唯一接触 third_party/LightningDiT、timm、lpips、fid    │
 └────────────────────────────────────────────────────────────────────────┘
```

规则：
- `allocation.py`、`relevance.py` 的 v2 函数只依赖 numpy（`relevance.py` 文件顶部已有 torch import，属 v1，保持不动；`alignment_shift_scan` 只用 torch 接收张量输入）。
- `relevance.py`、`integration.py` 的 v2 段追加在 v1 代码之后并自带 import，v1 部分除模块 docstring 的一句说明外逐字节不变。
- 只有 `integration.py` 允许 import `third_party/LightningDiT`、`timm`、`lpips`、`pytorch_fid`（均在函数内惰性 import）；其它模块通过**参数注入**（模型对象/可调用）使用它们，从而可在 CPU 上用桩对象单测。
- 模块间 import：`stats → relevance`（`explained_covariance`）；`datasets`、`decoder_adapter`、`generation → channel`；`evaluation → stats`（`GroupedMomentAccumulator`）。`allocation` 与 `stats`/`channel` 只按属性鸭子类型对接（`m, Sigma, Sigma_expl, meta, fingerprint()` / `rule, R, w, name, C, m, A, B, fingerprint()`），互不 import。
- `import rwvae` 只急加载纯 numpy 的 `theory`、`metrics`；其余模块（含全部 v2 模块）在首次属性访问时惰性加载（PEP 562），`import rwvae` 不引入 torch。`rwvae.__version__ == "0.2.0"`。
- 依赖（`pyproject.toml`）：核心 `numpy, scipy, torch>=2.6, torchvision, pyyaml, safetensors, pillow, tqdm`；`[dev]` = `pytest`；`[gpu]`（真实资产路径）= `timm, einops, fairscale, torchdiffeq, omegaconf, requests, lpips, pytorch-fid, matplotlib, tensorboard`。
- 第三方代码一律不改。

---

## 1. 统一约定

| 项 | 约定 |
|---|---|
| 形状 | latent `[B,C,H,W]`，`C=32,H=W=16`；token 维度在 dim=1。图像 `[B,3,256,256]`，取值 `[-1,1]`（`ToTensor`+`Normalize(0.5,0.5)`）。DINO patch `[B,D,16,16]` |
| token 展平 | `x.permute(0,2,3,1).reshape(-1, C)`（`stats.tokens`，行主序 `(h,w)`），z 与 y 必须用同一展平 |
| 精度 | 统计/特征分解/闭式解一律 **float64**（numpy）；训练缓存 fp16/bf16/fp32；信道在 `promote(输入 dtype, float32)` 下计算（buffers 默认 fp32，可建 fp64） |
| 码率单位 | nats/token；`bpp = R*256/(256*256*ln2) = R/(256 ln 2)` |
| 随机性 | 所有随机都由显式 seed 决定；逐样本噪声 seed = `noise_seed(base, epoch, index)`（splitmix64，平台无关）。epoch 槽位：`ChannelLatentDataset` 为训练 epoch，`finetune_decoder` 为优化步 t，`initial_noise` 恒为 0 |
| 错误 | 非法输入/配置抛 `ValueError`（带具体数值）；文件缺失 `FileNotFoundError`；行号越界 `IndexError`；LightningDiT 包名冲突 `ImportError`；训练中 loss/梯度范数非有限 `FloatingPointError`；`runtime.to_jsonable` 遇不支持类型 `TypeError`。**禁止静默 clip/收缩**。数值容差内的修正（K 特征值截断、Σ_y 伪逆截断、FID 的 eps 偏移）必须写进 `meta`/返回的 info |
| 产物元数据 | 每个产物（缓存分片、stats、allocation、decoder 权重、DiT ckpt、结果 JSON）都带 `meta`：来源输入的 sha256/fingerprint、split、seed、dtype、代码 git commit |
| 文件 | 产物写到调用方给出的确切路径（不自动追加 `.npz`）；config.yaml、summary.json、`Run.save_json`、manifest、缓存分片与 cache_index.json、DiT ckpt 先写临时文件再 `os.replace`（原子） |
| 日志 | Python `logging` → stdout + run 目录 `log.txt`（handler 挂在包 logger `rwvae` 上，库内 `logging.getLogger(__name__)` 也写入）；标量指标写 `metrics.jsonl`；每次脚本运行在 `manifests/run_registry.jsonl` 追加一行 |

---

## 2. 数学（实现必须逐条满足；测试以 float64 相对误差 ≤1e-8 验证）

记 `z0∈R^C`（后验均值 token），`m=E z0`，`Σ=Cov z0 ≻ 0`；`y∈R^D`（DINO patch）。

- **P1 解释协方差**：`Σ_expl = Σ_zy (Σ_y+λI)^{-1} Σ_yz`（λ≥0；`ridge_rel` 模式下 `λ = ridge_rel·tr(Σ_y)/D`）。在联合（样本）协方差 PSD 的前提下 `0 ⪯ Σ_expl ⪯ Σ`。`K = Σ^{-1/2} Σ_expl Σ^{-1/2}`，`eig(K)∈[0,1]`；λ=0 时 `eig(K)` = 典型相关系数平方；λ>0 时是收缩版本，不能称 CCA²。
- **P2 加权 KLT**：`M = I+(w²-1)K`（w≥1），`G = M^{1/2} Σ M^{1/2} = Q diag(s) Qᵀ`（s 降序），`T = Qᵀ M^{1/2}`，`T^{-1} = M^{-1/2} Q`。则 `T Σ Tᵀ = diag(s)`，`s = eig(Σ_w)`，`Σ_w = Σ+(w²-1)Σ_expl`。T 一般非正交（不得用 Tᵀ 代替 T^{-1}）。对应加权失真 `E[(z0-ẑ)ᵀ M (z0-ẑ)]`。w=1 ⇒ M=I ⇒ PCA。
- **P3 反向注水**：`min Σ s_i e^{-2r_i}  s.t. r_i≥0, Σr_i=R` ⇒ `r_i = ½ ln⁺(s_i/θ)`，`κ_i = (1-θ/s_i)⁺ = 1-e^{-2r_i}`，θ 唯一（R>0 时 θ∈(0, max s)）。闭式求解：s 降序，k 个活跃时 `ln θ_k = (Σ_{i≤k} ln s_i − 2R)/k`，取满足 `s_k > θ_k ≥ s_{k+1}` 的 k。R=0 ⇒ 全部 κ=0，θ=max s。
- **P4 方差保持信道**：`u=T(z0−m)`，`u'_i = √κ_i u_i + √((1−κ_i)s_i) ε_i`；`A = T^{-1}diag(√κ)T`，`B = T^{-1}diag(√((1−κ)s))`，`z' = m + A(z0−m) + Bε`，`AΣAᵀ+BBᵀ = Σ`（不要求高斯）。跨 token 协方差 `A Cov(z0_p,z0_q) Aᵀ` 一般不保持。
- **P5 期望 KL**：`q(z'|x)=N(m+A(z0−m), BBᵀ)`，`p0=N(m,Σ)`，若 `E z0=m, Cov z0=Σ`、κ<1，则 `E_x KL(q‖p0) = Σ_i −½ln(1−κ_i) = R`（不要求高斯）。`I(X;Z') ≤ R`，仅当聚合后验 = p0（如 z0 高斯）时取等。
- **P6 LMMSE**：`Cov(z0,z')Cov(z')^{-1} = ΣAᵀΣ^{-1} = A` ⇒ `ẑ = m + A(z'−m)`；仅联合高斯时等于条件期望。融合进 decoder 首层（`post_quant_conv`, 1×1, `W0,b0`）：`W0' = W0 A`，`b0' = b0 + W0(I−A)m`。
- **P7 闭式失真**：LMMSE 原始 latent MSE `D_latent = tr[T^{-1} diag((1−κ)s) T^{-T}]`；加权失真 `Σ(1−κ_i)s_i = Σ min(s_i,θ)`。二者都不是像素 PSNR/rFID 的闭式。
- **白化保留矩阵**（诊断）：`V = Σ^{-1/2} T^{-1} diag(√s)` 正交；`G_ret = V diag(κ) Vᵀ`；规则间重叠 `tr(G_aG_b)/√(tr G_a² tr G_b²)`；语义保留 `tr(G_ret K)/tr(K)`；方向相关度 `k_i = v_iᵀ K v_i ∈[0,1]`。
- **三条有损规则**（等名义码率 R）：
  - `equal_snr`（B1）：PCA 基 `T=Q_Σᵀ`，`r_i=R/C`（与基无关：`A=√κ I`）。
  - `variance`（B2）：w=1 注水。
  - `r2`（B3，主方法）：w>1，Σ_expl 来自 DINO 岭回归。`r2` 在 w=1 时必须与 `variance` 完全一致（单测）。
  - `none`（B0）：`A=I, B=0`，无码率约束参考，不属于等码率比较。

数值核对方式（`allocation.identity_residuals`，见 §3.2）：P6 在 Σ 白化坐标下核对、码率/KL 残差除以 `max(1,R)`，因为原始坐标的舍入误差会被 cond(Σ) 放大。唯一放宽的容差：cond(Σ)=1e9 的近奇异族上 `gaussian_mutual_information` 与 R 的偏差为 1e-6。

---

## 3. 模块契约

### 3.1 `rwvae/runtime.py`

```python
REPO_ROOT = Path(runtime.py).resolve().parents[1]; DEFAULT_REGISTRY = REPO_ROOT/"manifests"/"run_registry.jsonl"; PACKAGE_LOGGER = "rwvae"
def load_config(paths: str | PathLike | Sequence[str | PathLike], overrides: Sequence[str] = ()) -> dict
    # 依次深合并 YAML（mapping 递归合并，标量/列表替换；空文件 = {}），返回不共享容器的普通 dict/list；
    # 未知顶层键不报错（配置是数据），但 override 指向不存在的路径时报错，除非以 "+a.b=..." 显式新建（"+" 也可覆盖已有键）；
    # overrides 形如 "a.b.c=VALUE"（VALUE 按 YAML 解析），不能是单个裸字符串；路径中途遇到非 mapping 值 → ValueError。
    # 解析 = yaml.safe_load 加一条规则：指数写法 "1e-3"/"1.0e5" 是 float（YAML 1.2），dump 时同形字符串加引号，配置精确往返。
def save_config(cfg: Mapping, path) -> None            # 原子写 UTF-8 YAML；load_config 读回后指纹不变（inf/nan/日期可往返）；cfg 非 mapping → ValueError
def config_fingerprint(cfg: Mapping) -> str            # sha256(canonical json of to_jsonable(cfg))[:16]；键序无关，1 与 1.0 不同；cfg 非 mapping → ValueError
def to_jsonable(obj) -> Any                            # numpy/torch → Python 标量/list，Path → POSIX 字符串，tuple → list，set → 排序 list，
                                                       # dataclass → dict，date → ISO；键 → str（碰撞 → ValueError）；非有限浮点 → "inf"/"-inf"/"nan"；其它类型 TypeError
def sha256_file(path, chunk=1<<20) -> str
def git_commit(repo_dir=None) -> str | None            # 默认本仓库 HEAD；失败返回 None，不抛；不检测未提交改动
def seed_everything(seed: int, deterministic: bool = False) -> None
    # random/numpy/torch(+全部 cuda)；seed ∈ [0, 2^32)；deterministic=True 另设 use_deterministic_algorithms(True, warn_only=True)、
    # cudnn.benchmark=False、CUBLAS_WORKSPACE_CONFIG=:4096:8（未设置时）
def select_device(pref: str = "auto") -> torch.device  # "auto"(cuda 可用则 cuda)|"cpu"|"cuda"|"cuda:N"；要求 CUDA 却不可用/索引不存在 → ValueError（不静默回退）
def env_info() -> dict                                 # {python, platform, torch, numpy, cuda_available, cuda_version, gpus}
def get_logger(name: str, log_file: str | Path | None = None) -> logging.Logger
    # INFO；stdout + 至多一个 UTF-8 文件 handler（换文件时替换并关闭旧的）；幂等，不重复加 handler；propagate=False；
    # 时间戳为带 UTC 偏移（%z）的本地时间，便于与 metrics/registry 的 UTC 时间对齐
class Run:                                             # 一次脚本运行；每进程一个（DDP 下仅 rank 0）
    def __init__(self, script: str, cfg: Mapping, out_dir, *, registry=None, overwrite=False, resume=False)
        # registry=None → DEFAULT_REGISTRY；out_dir=None → ValueError。out_dir 已有 config.yaml 时：无 flag → FileExistsError；
        # resume=True → 续跑（config 指纹须一致，否则 ValueError，除非同时 overwrite=True）；overwrite=True → 原地新跑
        # （重写 config.yaml、重置 metrics.jsonl）；两种情况都删旧 summary.json，不删其它文件；resume=True 而目录无 run → 新跑并 warning。
        # 被拒绝的构造不改动磁盘；log.txt 总是追加；git commit 与 env_info 在构造时记录一次。
    out_dir: Path（resolve 后）; logger: logging.Logger（"rwvae.<script stem>"）; script; fingerprint; registry; resumed: bool
    # JSON 中路径一律 POSIX 形式；metrics.jsonl / run_registry 每行一次 write 追加，无文件锁
    def log_metrics(self, step: int | None, **scalars) -> None   # 追加 {time(ISO-8601 UTC), step, **scalars} 到 out_dir/metrics.jsonl
    def save_json(self, name: str, obj) -> Path                   # to_jsonable 后原子写 out_dir/name（可含子目录；绝对路径/".." → ValueError）
    def finish(self, status: str = "ok", **summary) -> None
        # summary.json = registry 记录 {time, script, status, out_dir, config_fingerprint, git_commit, env, summary}，同一记录追加到
        # run_registry；关闭 log.txt；重复调用 → RuntimeError
def add_common_args(parser) -> None   # --config (可多次)、--set (可多次)、--out、--device (默认 auto)、--seed (int)、--overwrite、--resume
def parse_config_from_args(args) -> dict   # load_config(args.config or [], args.set or [])；给了 --seed 时写入 cfg["seed"]
```

### 3.2 `rwvae/allocation.py`（numpy float64，核心数学）

```python
RULES = ("none", "equal_snr", "variance", "r2")

def sym(M, name="M") -> np.ndarray               # 有限非空方阵、相对不对称 ‖M−Mᵀ‖_F/‖M‖_F ≤ 1e-8，否则 ValueError；返回 (M+Mᵀ)/2
def check_spd(Sigma, name="Sigma", rel_tol=1e-12) -> dict   # {min_eig, max_eig, cond, eff_rank(熵)}；min_eig ≤ rel_tol·max_eig 报错
def sym_sqrt(M) -> np.ndarray                    # eigh；任何负特征值报错（不 clip）
def sym_inv_sqrt(M) -> np.ndarray                # eigh；要求 PD：min eig > 1e-12·max eig

def whitened_relevance(Sigma, Sigma_expl, tol=1e-9) -> tuple[np.ndarray, dict]
    # K=Σ^{-1/2}Σ_explΣ^{-1/2}（对称化）；eig(K) 必须 ∈[-tol,1+tol]，否则 ValueError（说明违反 P1）；
    # 容差内的越界特征值截断到 [0,1] 后重构 K；info={"eig_min_raw","eig_max_raw","clipped": bool,"tol"}（写进 Allocation.meta["K"]）

class KLT(NamedTuple):  T, Tinv, s, Q, M_half, K, info   # info={"Sigma": check_spd(...), "K": whitened_relevance info 或 None}
def weighted_klt(Sigma, Sigma_expl=None, w=1.0) -> KLT
    # w<1、非有限，或 (w>1 且 Sigma_expl is None) → ValueError；s 降序；给了 Sigma_expl 就计算 K 并检查 P1（w=1 也算）；
    # w==1 时 T=Q_Σᵀ、Tinv=Q_Σ、M_half=I（与 PCA 同一代码路径，故 r2@w=1 与 variance 逐位相同）；
    # Q 每列绝对值最大的元素取正（同一 LAPACK 下可复现；重复特征值的特征空间内 B 只确定到旋转，A 与 BBᵀ 唯一）

def reverse_waterfill(s, R) -> tuple[np.ndarray, np.ndarray, float]
    # 返回 (rates, keep, theta)，按 s 的输入顺序；s 必须有限且 >0（零/负 → ValueError）；R 有限且 ≥0；
    # 闭式活跃集法（见 §2 P3），keep = -expm1(-2r)；sum(rates)==R 到机器精度；R=0 → rates=keep=0，theta=max s

def equal_snr_rates(C, R) -> np.ndarray          # R/C 常数

def channel_matrices(T, Tinv, s, rates) -> tuple[A, B]
    # A = Tinv diag(√keep) T；B = Tinv diag(√(e^{-2r} s))（用 exp(-2r) 计算 1-κ，避免消减误差）；s 有限且 >0，rates ≥0（拒绝 NaN，允许 inf）

@dataclass(frozen=True, eq=False)
class Allocation:
    rule: str; R: float; w: float
    m: np.ndarray; T: np.ndarray; Tinv: np.ndarray; s: np.ndarray
    rates: np.ndarray; keep: np.ndarray; theta: float | None
    A: np.ndarray; B: np.ndarray
    meta: dict = {}                # 须可 JSON 化；构造时校验 rule∈RULES 与数组形状、转 float64，meta 过一遍 JSON（numpy/Path 自动转换）
    @property C -> int
    @property name -> str          # none→"none"；equal_snr/variance→f"{rule}_R{R:g}"；r2→f"r2_R{R:g}_w{w:g}"
    def save(self, path) -> None   # .npz（确切路径，自动建父目录），allow_pickle=False 可读；meta 存 JSON 字符串；格式标签 "rwvae.allocation/v2.1"；theta=None 存空数组
    @classmethod load(cls, path) -> "Allocation"   # 格式标签不符 → ValueError；meta 精确往返
    def fingerprint(self) -> str   # sha256(rule, R, w 与 m、A、B、keep 的 <f8 字节)[:16]（不含 meta）

def solve_allocation(stats, rule: str, w: float | None, R: float | None) -> Allocation
    # stats 需有属性（或 dict 键）m, Sigma（r2 还需 Sigma_expl）——LatentStats 或任意同属性对象/dict 均可。
    # none（B0）：R=inf、w=1、T=Q_Σᵀ、Tinv=Q_Σ、s=eig(Σ)、keep=ones、rates=full(inf)、theta=None、A=I、B=0；
    #   R 参数被忽略（可为 None）；闭式码率/KL 函数对其返回 inf；只作无码率约束参考。
    # equal_snr: T=Q_Σᵀ, rates=R/C, theta=None；variance: w=1 注水；r2: w≥1（w=1 结果与 variance 逐元素一致）。
    # 除 r2 外 w 必须为 None 或 1（none 也一样），否则 ValueError。
    # meta 键：rule, R, w, C, Sigma_spectrum（check_spd）, K（截断 info 或 None）, created_utc,
    #   stats_fingerprint（stats 有 fingerprint() 时）, ridge（= stats.meta["ridge"]，若存在）

# 闭式诊断（全部 float64，返回 Python float 或 ndarray）
def nominal_rate(alloc) -> float                    # Σ rates（none → inf）
def expected_kl(alloc, Sigma) -> float              # P5，只用 A,B,Σ（不用 rates）：一般高斯 KL 公式在 Σ 白化坐标下求值
                                                    #   0.5(‖A_w‖²_F + ‖B_w‖²_F − C) − ln|det B_w|，A_w=Σ^{-1/2}AΣ^{1/2}，B_w=Σ^{-1/2}B；BBᵀ 奇异 → inf
def gaussian_mutual_information(alloc, Sigma) -> float   # ½ ln det Σ / det BBᵀ = −ln|det B_w|（z0 高斯时 = I(Z0;Z')）；对特征分解的后向误差一阶敏感：
                                                    #   cond(Σ)=1e9 且 w>1 时 |MI−R| 可达 ~3e-8（E KL 仍 <1e-10）
def latent_mse_lmmse(alloc) -> float                # P7 D_latent = Σ_i e^{-2r_i} s_i ‖Tinv[:,i]‖²
def weighted_distortion(alloc) -> float             # Σ(1-κ)s
def bpp_equivalent(R, tokens=256, pixels=256*256) -> float
def whitened_basis(alloc, Sigma) -> np.ndarray      # V（正交）；max|VᵀV−I| > 1e-6（Σ 不属于该 allocation）→ ValueError
def retention_matrix_whitened(alloc, Sigma) -> np.ndarray
def retention_overlap(Ga, Gb) -> float              # 零保留矩阵（R=0）→ ValueError
def semantic_retention(alloc, Sigma, K) -> float    # 需 tr K > 0
def direction_relevance(alloc, Sigma, K) -> np.ndarray   # k_i
def identity_residuals(alloc, Sigma) -> dict        # 无量纲残差，一致输入下 ~1e-15..1e-9（none 的 rate/kl：inf==inf 记 0）：
    # klt      ‖TΣTᵀ−diag s‖_F/‖Σ‖_F                    variance  ‖AΣAᵀ+BBᵀ−Σ‖_F/‖Σ‖_F
    # rate     |Σr−R|/max(1,R)                           kl        |E KL−R|/max(1,R)
    # lmmse    ‖Σ^{-1/2}(ΣAᵀΣ^{-1}−A)Σ^{1/2}‖_F/√C = ‖A_w−A_wᵀ‖_F/√C（白化坐标下的 P6）
```

### 3.3 `rwvae/relevance.py`（在 v1 内容之后追加 v2 段落）

```python
def ridge_lambda(Sigma_y, ridge_rel: float) -> float      # ridge_rel·tr(Σ_y)/D；ridge_rel 有限且 ≥0
def explained_covariance(Sigma_zy, Sigma_y, ridge_rel=0.0, rank_tol=1e-12) -> tuple[np.ndarray, dict]
    # 通过 eigh(Σ_y) 计算；λ=0 时对 e ≤ rank_tol·max e 的方向用伪逆（info 记录 rank）；输出严格对称；
    # Σ_y 不对称、有 < −rank_tol·max e 的特征值（非 PSD）或无正特征值 → ValueError；0 < λ ≤ rank_tol·max e → ValueError（不静默使用）
    # info = {lambda, ridge_rel, rank_tol, rank_y, dim_y, cond_y（rank_y<dim_y 时 inf）, eig_min_y, eig_max_y}
    # 注：DINOv2 forward_features 经最终 LayerNorm，patch 特征满足一个线性约束，实测 Σ_y 的秩为 D−1（1023/1024）
def cca_spectrum(Sigma, Sigma_zy, Sigma_y, ridge_rel=0.0) -> np.ndarray   # eig(K) 降序，原值不截断（λ=0 即 ρ_i²）；Σ 须 PD
def explained_fraction(Sigma, Sigma_expl) -> float        # tr(Σ_expl)/tr(Σ)（扫描对齐时用）
def alignment_shift_scan(z: torch.Tensor, y: torch.Tensor, max_shift=2, ridge_rel=1e-3) -> dict[tuple[int,int], float]
    # z [B,C,H,W]、y [B,D,H,W]（同图配对）；对每个 (dy,dx) 把 z[:, :, h, w] 与 y[:, :, h+dy, w+dx] 在重叠区域配对，
    # 用无偏联合协方差（f64）计算 explained_fraction；若 y = roll(y0, (a,b), dims=(2,3))，峰值在 (a,b)；正确对齐时峰值应在 (0,0)。
    # max_shift 须为 [0, min(H,W)) 内的整数；最小重叠的配对数 B(H−k)(W−k) ≤ C+D → ValueError。用于 03 的“patch 网格一一对应”检查。
```

### 3.4 `rwvae/stats.py`

```python
@dataclass(eq=False)
class Moments:                       # 中心化充分统计（float64 numpy）
    n: float; mean: np.ndarray; m2: np.ndarray        # m2 = Σ(x−mean)(x−mean)ᵀ；空样本 n=0
    def merge(self, other) -> "Moments"               # Chan et al. 并行公式，精确
    def cov(self, ddof: int = 1) -> np.ndarray        # n ≤ ddof → ValueError
    @staticmethod
    def combine(items: Sequence["Moments"], weights: Sequence[float] | None = None) -> "Moments"  # 权重 k_g = “样本 g 复制 k_g 次”（bootstrap 用整数计数）

class GroupedMomentAccumulator:      # torch float64 在 device 上累计；G 组（按图片 ID 哈希分组）
    def __init__(self, dim: int, num_groups: int = 1, device="cpu")
    n [G], mean [G,d], m2 [G,d,d]（device 上 f64）; images_per_group [G] int64（CPU）; @property n_images -> int（其和）
    def update(self, x: torch.Tensor, group_ids: torch.Tensor | int) -> None   # x [N,dim]（任意浮点，内部转 f64，组内两遍法 + Chan 合并）；int = 全部行同组
    def add_images(self, group_ids) -> None          # 每图一个组号 [B]；单组累加器也接受 int k（= k 张图，契约里的 add_images(k)）；多组时给 int → ValueError
    def group(self, g) -> Moments; def total(self) -> Moments
    def subset(self, groups: Sequence[int], weights=None) -> Moments
    def state_dict(self) -> dict                     # {dim, num_groups, n, mean, m2, images_per_group}（CPU 张量）
    def load_state_dict(self, sd) -> None            # 张量或 numpy 数组（如 np.savez 读回）
    def merge_(self, other) -> None                  # 多进程/多分片合并（可跨 device）

def image_group_id(image_id: str, num_groups: int, salt: str = "rwvae-v2") -> int
    # int.from_bytes(sha1(f"{salt}\0{image_id}")[:8], "big") % num_groups（与平台、进程、PYTHONHASHSEED 无关）
def tokens(x: torch.Tensor) -> torch.Tensor          # [B,C,H,W] → [B·H·W, C]（§1 展平约定）

@dataclass(eq=False)
class LatentStats:
    m: np.ndarray; Sigma: np.ndarray
    Sigma_zy: np.ndarray | None; Sigma_y: np.ndarray | None
    Sigma_expl: np.ndarray | None                      # 由 with_explained(ridge_rel) 填充
    n_tokens: int; n_images: int; meta: dict = {}
    @property C, D                                     # 无 y 时 D=0
    @classmethod from_joint(cls, mom: Moments, C: int, n_images: int, meta=None) -> "LatentStats"
        # 拆分无偏联合协方差 m2/(n−1) 的块，记 meta["cov_ddof"]=1；mom.n 须为整数
    def with_explained(self, ridge_rel: float) -> "LatentStats"   # 计算 Σ_expl，explained_covariance 的 info 写入 meta["ridge"]
    def channel_norm(self) -> tuple[np.ndarray, np.ndarray]        # (m, sqrt(diag Σ))：所有臂共用的 DiT 逐通道归一化
    def save(self, path) / load(cls, path)   # .npz（确切路径），allow_pickle=False；meta 存 JSON；另存 fingerprint，load 时复核（不符 → ValueError）
    def fingerprint(self) -> str             # sha256(各数组的名字/形状/<f8 字节 + n_tokens/n_images)[:16]（不含 meta）

def stream_stats(batches: Iterable[tuple[mu, y, group_ids]], C: int, D: int, num_groups=16, device="cpu") -> GroupedMomentAccumulator
    # 计划 §3.2 的 stream_stats：mu [B,C,H,W]、y [B,D,H,W]（D=0 时为 None）、group_ids [B]（每图一个）→ 联合 [z;y] token 矩，
    # 图 b 的全部 token 进组 group_ids[b]；统计量 = LatentStats.from_joint(acc.total(), C, acc.n_images, meta)
def spectrum_report(Sigma) -> dict                   # {eig(降序), min, max, cond(min≤0 → inf), eff_rank(正特征值上的熵), n_nonpositive}；不对称 >1e-8 报错
def principal_angles(U, V) -> np.ndarray             # 列正交基之间的 min(p,q) 个主夹角（弧度，升序）；列不正交 → ValueError
def relative_frobenius(A, B) -> float                # ‖A−B‖_F/‖B‖_F
def split_half_groups(num_groups) -> tuple[list[int], list[int]]   # 偶/奇组（num_groups ≥ 2）
def bootstrap_group_weights(num_groups, n_boot, seed) -> np.ndarray   # [n_boot, G] int64 多项式计数（组级 = 图片级块 bootstrap）
```

### 3.5 `rwvae/channel.py`

```python
def splitmix64(x: int) -> int
def noise_seed(base_seed: int, epoch: int, index: int) -> int          # 结果 ∈ [0, 2^63)；负数或非整数 → ValueError
def sample_noise(shape_chw, seeds: Sequence[int], *, device=None, dtype=torch.float32) -> torch.Tensor
    # 每个样本独立 CPU Generator(seed) 生成 → 与 batch 组成/worker 数/device 无关

class RWChannel(nn.Module):
    """z' = m + A(mu − m) + B ε  （逐 token，作用于 dim=1）。buffers: m[C], A[C,C], B[C,C], rate(标量 f64)。"""
    def __init__(self, m, A, B, rate=float("inf"), *, dtype=torch.float32, meta: dict | None = None)   # 形状不符/非有限 → ValueError
    meta: dict; @property C -> int
    @classmethod from_allocation(cls, alloc, dtype=torch.float32) -> "RWChannel"   # meta={rule, R, w, name, allocation_fingerprint}；rule none → identity
    @classmethod identity(cls, C, m=None, dtype=torch.float32) -> "RWChannel"   # B0
    def signal(self, mu) -> torch.Tensor                 # m + A(mu−m)；A=I 时原样返回 mu（B0 逐位精确）
    def forward(self, mu, noise: torch.Tensor | None = None, generator: torch.Generator | None = None) -> torch.Tensor
        # noise 显式给出时形状必须与 mu 相同；B 全零时不采样噪声；计算 dtype = promote(mu.dtype, float32)，buffers 转到该 dtype
        # （fp64 信道作用于 fp32 输入时按 fp32 计算；半精度输入按 fp32）
    def lmmse(self, z_prime) -> torch.Tensor             # P6：m + A(z'−m)
    @property is_identity -> bool
```

### 3.6 `rwvae/decoder_adapter.py`

```python
BASE_MODES = ("identity", "lmmse"); MIXED_PRECISION = ("bf16", "no")
def lmmse_affine(A, m) -> tuple[np.ndarray, np.ndarray]   # float64 (W_pre=A, b_pre=(I−A)m)；应传 Allocation 的 f64 A、m，而非 RWChannel 的 fp32 buffers
def fuse_affine_into_conv1x1(conv: nn.Conv2d, W_pre, b_pre) -> nn.Conv2d
    # 返回新 conv：W0·W_pre、b0+W0·b_pre（无 bias 时 b0=0；结果总带 bias），f64 计算后转回 conv 的 dtype/device；不改原 conv；
    # W_pre 可为 [in, k]（新 conv 有 k 个输入，LMMSE 是方阵情形）；复制 requires_grad/training；skip_init（不抽全局 RNG）；非 plain 1×1 → ValueError
class AdaptedDecoder(nn.Module):
    """decode_with_adapter(z') —— 计划 §3.2。mode ∈ {"identity","lmmse","finetuned"}，base_mode ∈ BASE_MODES。
       持有 post_quant_conv 与 decoder 的**拷贝**（不改原 tokenizer）；lmmse 模式下前置层被融合进 post_quant_conv
       （实际 f16d32 的 post_quant_conv 为 32→32、1×1、带 bias，real_assets 测试核对）。"""
    def __init__(self, post_quant_conv: nn.Conv2d, decoder: nn.Module, *, mode="identity", A=None, m=None)
        # mode 只能是 identity/lmmse（finetuned 由 finetune_decoder/load_weights 得到）；A、m 只用于 lmmse；
        # post_quant_conv 在任何模式下都须是 plain 1×1 conv（也能发现参数顺序写反）；起始 train/eval 状态随 decoder
    mode: str; base_mode: str
    def forward(self, z_prime) -> torch.Tensor            # z' [B,C,H,W] → 图像 [-1,1]（不 clamp；clamp 在评测量化时做）；形状不符 → ValueError
    def trainable_parameters(self) -> Iterator[nn.Parameter]   # post_quant_conv + decoder
    def save(self, path, meta) -> None                    # 仅 mode=="finetuned"（解析适配器由 (A,m) 重建）；safetensors，metadata {"meta": json(meta), "base_mode"}
    def load_weights(self, path) -> dict                  # strict 加载（不匹配 → torch RuntimeError；缺 meta/base_mode metadata → ValueError）；
                                                          # mode="finetuned"、base_mode 取自文件；原样返回保存的 meta
def decode_with_adapter(adapter: AdaptedDecoder, z_prime) -> torch.Tensor   # == adapter(z_prime)
def finetune_decoder(adapter, loader, channel: RWChannel, *, lpips_fn, steps, lr, l1_weight=1.0, lpips_weight=1.0,
                     betas=(0.5, 0.9), weight_decay=0.0, grad_clip=None, mixed_precision="bf16", device,
                     noise_base_seed, log_fn=None, log_every=50) -> dict
    # loader 产出 (images, mu, label, index)，耗尽后重新迭代，每轮之前（含第 0 轮）调 loader.dataset.set_epoch(epoch)（若有；
    # 不调 sampler.set_epoch），数据顺序由调用方播种；
    # 第 t 步（0 起）样本 i：z' = channel(mu, noise=sample_noise(seed=noise_seed(base, t, index_i)))；
    # loss = l1·mean|x̂−x| + lpips·mean LPIPS(x̂,x)；AdamW(lr, betas, weight_decay)（wd=0 即 Adam）；可选梯度范数裁剪；
    # bf16 autocast 仅在 CUDA 上，CPU 用参数 dtype；encoder 不参与；GAN 不实现（候选池）。
    # 副作用：adapter 移到 device、参数 requires_grad=True、首次更新后 mode="finetuned"、结束时恢复 train/eval。
    # 返回 {steps, final_loss, final_l1, final_lpips（最后一个日志窗口的均值）, mixed_precision（实际运行的 "bf16"/"no"）}；
    # log_fn(step, {loss, l1, lpips}) 每 log_every 步与最后一步收到窗口均值（step = 已完成步数，1 起）；窗口均值非有限 → FloatingPointError；
    # 空 loader、解码形状与图像不符 → ValueError
```

### 3.7 `rwvae/datasets.py`

```python
IMG_EXTENSIONS = (".jpg",".jpeg",".png",".ppm",".bmp",".pgm",".tif",".tiff",".webp")
SPLITS = ("train_dev", "train_fit", "val"); MANIFEST_SCHEMA_VERSION = 1; CACHE_INDEX = "cache_index.json"
PREPROCESSING = {"crop": "adm_center_crop", "resample": "box_halving_then_bicubic", "range": "[-1,1]"}
CACHE_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
def center_crop_arr(pil_image, image_size) -> PIL.Image     # 与 ADM / LightningDiT 完全一致
def image_transform(image_size=256, hflip=False) -> Callable  # RGB → center_crop → (确定性 TF.hflip) → ToTensor → Normalize(0.5,0.5)；可 pickle

@dataclass
class Manifest:
    meta: dict; classes: list[str]                 # classes[label] = wnid（ImageFolder：类目录名排序）
    splits: dict[str, dict[str, list]]             # split → {"image_id": [...], "path": [...], "label": [...]}（列式，POSIX path 相对 meta["split_roots"][split]）
    def save(path) / load(cls, path)               # 单行 JSON（含 schema_version），save/load 都先 validate
    def fingerprint() -> str                       # 完整 64 位 hex：sha256(canonical json {classes, splits})（不含 meta，故与数据位置/时间无关）
    def split_size(split) -> int; def record(split, i) -> dict（越界 IndexError）; def image_path(split, i) -> Path
    def validate(self) -> None                     # 类名唯一、每个 split 恰有三列且等长、label∈[0,K) 的 int、image_id 跨 split 不重复
def scan_image_folder(root) -> tuple[list[str], list[tuple[str, int]]]
    # (classes, [(relpath, label)])：torchvision ImageFolder 约定（递归、跟随符号链接、扩展名不区分大小写；空类目录 → ValueError）；
    # relpath 为 POSIX，类内按普通字符串排序（平台无关）
def build_manifest(train_root, val_root, *, dev_per_class=20, seed=2026, name="imagenet1k", image_size=256) -> Manifest
    # train_dev：每类在排序后的文件列表上用 np.random.default_rng([seed,label]).choice(n, dev_per_class, replace=False)；train_fit：其余；val：全部
    # （每类须多于 dev_per_class 张；val 类目录须是 train 类的子集）；image_id = 文件名 stem（跨 split 唯一）；
    # meta：schema_version、name、train/val_root、split_roots、seed、dev_per_class、image_size、preprocessing、class_map_sha256、counts、numpy_version、created

class ManifestImageDataset(Dataset):                # → (image[-1,1] float32 [3,S,S], label, index)
    def __init__(self, manifest, split, *, image_size=256, hflip=False, indices=None)

class LatentCacheWriter:                            # safetensors 分片；分片号确定、可断点续写、可多进程分片并行
    # 分片 k 固定对应 split 行号 [k·S, min((k+1)·S, N))（S=shard_size）；文件 {split}_shard{k:05d}.safetensors
    # 键：latents [n,C,H,W]（存储 dtype）、latents_flip（可选）、labels int64、indices int64（split 内行号）
    # 分片 metadata：{"meta": json(meta), "shard_id", "n", "dtype"}；
    # 目录下 cache_index.json：{meta, split, total, dtype, has_flip, shard_size, latent_shape, shards:[{file,shard_id,n,sha256}]}；一个目录只放一个 split
    def __init__(self, out_dir, *, split, meta: dict, dtype="float16", shard_size=10000)   # meta 须可 JSON 化，应含 manifest_fingerprint 等来源信息
    def num_shards(self, total) -> int; def shard_rows(self, shard_id, total) -> range; def shard_path(self, shard_id) -> Path
    def shard_done(self, shard_id, expected_n) -> bool   # 文件存在且 metadata 的 n 一致；已有文件的 meta/dtype/shard_id 不同 → ValueError（不混入、不覆盖）
    def write_shard(self, shard_id, latents, labels, indices, latents_flip=None) -> dict
        # 先写 .tmp 再 os.replace（原子）；indices 必须恰为 [k·S, k·S+n) 且 1≤n≤S；labels ≥0；转存储 dtype 后须有限（fp16 溢出 → ValueError）；
        # 存独立的连续 CPU 拷贝；返回 {file, shard_id, n, has_flip}
    def finalize(self, expected_total) -> dict
        # 校验分片号恰为 0..K−1、每片 metadata/键/布局/indices（恰好覆盖 [0,N) 一次）/labels，latent 形状与 has_flip 一致；
        # 写 cache_index.json（含每片 sha256）；目录已有其它 split 的索引 → ValueError
class LatentCache:                                  # 只读、随机访问、worker 安全（句柄惰性打开，pickle 时丢弃）
    def __init__(self, cache_dir, *, manifest: Manifest | None = None, verify: bool = True)
        # 缺 cache_index.json → FileNotFoundError；给了 manifest 就检查（与 verify 无关）：meta["manifest_fingerprint"]、split 行数、labels 逐行一致；
        # verify=True 复核每片 metadata、布局与 indices 覆盖（不重算 sha256）
    def __len__(self); def get(self, i, flip=False) -> tuple[torch.Tensor(float32 [C,H,W]), int label, int index]   # 越界 IndexError；无 flip 时 flip=True → ValueError
    meta: dict; split: str; dtype: str; has_flip: bool; total: int; shard_size: int; latent_shape: tuple; cache_dir: Path
    def iter_batches(self, batch_size) -> Iterator[(mu [b,C,H,W] f32, labels int64, indices int64)]   # 按行号顺序读（可跨分片），统计用

class ChannelLatentDataset(Dataset):                # 计划 §3.2：__getitem__(idx) → (z_prime, label, index)
    def __init__(self, cache: LatentCache, channel: RWChannel | None, norm_mean, norm_std, *,
                 base_seed: int, flip_prob=0.5, resample_each_epoch=True, latent_multiplier=1.0)
    def set_epoch(self, epoch: int) -> None; epoch: int
    # 逐样本 Generator(noise_seed(base_seed, epoch_eff, index))：先抽 flip 的 uniform（flip_prob=0 也抽，故 ε 与 flip 设置无关），再抽 ε；
    # channel=None（B0）时不抽 ε；flip_prob>0 要求 cache.has_flip；epoch_eff = epoch if resample_each_epoch else 0；
    # 返回 ((z'−mean)/std·multiplier).float()；mu 为 fp32，故信道按 fp32 计算。
    # 所有臂共用同一 cache 与同一 (mean,std)=LatentStats.channel_norm()。每个 epoch 前调 set_epoch，DataLoader 用 persistent_workers=False

class ImageLatentPairDataset(Dataset):              # → (image, mu, label, index)，decoder 微调/重建评测用；split 恒为 cache.split（无 split 参数）
    def __init__(self, manifest, cache: LatentCache, *, image_size=256, indices=None)   # 初始化检查 manifest 指纹与行数，取样时逐项核对 label
```

### 3.8 `rwvae/evaluation.py`

```python
def to_uint8(images) -> torch.Tensor     # [-1,1] float [B,3,H,W] → uint8，公式与 LightningDiT 相同：clamp(127.5x+128,0,255).to(uint8)；半精度先升 fp32；非有限 → ValueError
def psnr_per_image(ref_u8, test_u8) -> torch.Tensor   # float64，每图 20log10(255/√MSE)，MSE=0 → inf（调用方应报告）
class FIDStatistics:                     # Inception pool3(2048) 流式 f64 矩（复用 stats.GroupedMomentAccumulator，单组，在 device 上）
    def __init__(self, inception: nn.Module, device, batch_size=100)
    def update(self, images_u8) -> None  # uint8 [B,3,H,W] → CPU 上 /255（与 ToTensor 逐位一致）→ device → inception（pytorch-fid InceptionV3 内部 resize 299）；
                                         # 按 batch_size 分块，eval、no_grad；输出为张量或 list/tuple（取 [0]），展平为 [b,d]
    def compute(self) -> tuple[np.ndarray, np.ndarray, int]   # (mu, sigma（ddof=1）, n)
def frechet_distance(mu1, s1, mu2, s2, eps=1e-6, *, return_info=False) -> float | tuple[float, dict]
    # 与 pytorch-fid 相同的 sqrtm 处理（scipy.linalg.sqrtm(A)；SciPy 1.18 已去掉 disp 参数）：结果非有限时给两侧协方差加 eps·I 重算（记 warning），
    # 仍非有限 → ValueError；虚部对角 ≤1e-3 时取实部，否则 ValueError；略负的结果原样返回；return_info=True 另返回 {"eps_offset_applied", "max_abs_imag"}
def save_fid_stats(path, mu, sigma, n, meta) / load_fid_stats(path) -> (mu, sigma, n, meta)
    # npz：mu、sigma、n（int64，≥2）、meta（JSON）；兼容只有 mu/sigma 的 ADM / pytorch-fid npz（返回 n=-1、meta={"source":"external"}）；只有 n/meta 之一 → ValueError
class LPIPSMeter:                         # 注入 lpips 模型；update(x̂, x)（eval、no_grad、不 clamp）收集每图值；values() -> f64 [n]；mean()
    def __init__(self, lpips_model)
def pooled_features(z, grid=1) -> torch.Tensor           # adaptive_avg_pool2d → [B, C·grid²]（通道优先展平）；grid ≤ min(H,W)
PROBE_STD_EPS = 1e-6
def linear_probe(train_x, train_y, test_x, test_y, *, num_classes, weight_decay=1e-4, max_iter=200, device="cpu", seed=0) -> dict
    # 训练集统计量标准化 z=(x−mean)/(std+1e-6)（f64，ddof=0）→ 多项 logistic 回归：W=0,b=0 起，全批 L-BFGS 一步（lr=1, strong-Wolfe,
    # max_iter, history_size=100, tolerance_grad=1e-7, tolerance_change=1e-9），fp32，bias 不加罚；确定性，seed 只做校验
    # 返回 {top1, top5（K<5 时 None）, train_top1, correct: np.ndarray[bool]（每个测试图）, n_train, n_test, n_iter, min_train_std}
def dct_matrix(N) -> torch.Tensor                         # 正交 DCT-II（f64，= scipy.fft.dct(norm="ortho")）
def dct2_power(z, per_channel=False) -> torch.Tensor      # [B,C,H,W] → 每图每通道系数功率 → 对 B（与 C，除非 per_channel）取均值 → [H,W] 或 [C,H,W]（谱的均值；z 不中心化）
def radial_band_power(power_hw, n_bands) -> np.ndarray    # 按 √(kh²+kw²) 在 [0, r_max] 等宽分带的平均功率；归属用整数精确判定；空带 → ValueError
def bootstrap_mean_ci(values, n_boot=1000, seed=0, alpha=0.05) -> dict   # 以图片为单位的百分位 bootstrap（≥2 个有限值）：{mean, lo, hi, n, n_boot, alpha}
def paired_bootstrap_diff(a, b, n_boot=1000, seed=0, alpha=0.05) -> dict  # 同一批图片上 a−b 的 CI（= bootstrap_mean_ci(a−b)）；接受 bool 数组
```

### 3.9 `rwvae/generation.py`

```python
MIXED_PRECISION = ("bf16", "fp16", "no")
def update_ema(ema, model, decay: float) -> None          # 与 LightningDiT 相同（按 named_parameters，冻结参数也平均，buffers 不动）；只去掉开头的 "module."；
                                                          # decay ∉ [0,1] 或名字/形状不匹配 → ValueError（更新前检查）
def class_balanced_labels(num_samples, num_classes, seed) -> torch.Tensor   # 每类 ⌊N/K⌋ 或 +1，顺序 = randperm(N, Generator(seed))
def initial_noise(indices, shape_chw, base_seed, device) -> torch.Tensor     # 逐样本 seed=noise_seed(base_seed, 0, idx)，fp32，臂间共享
@dataclass(frozen=True)
class DiTTrainConfig: max_steps, global_batch_size, lr=2e-4, beta2=0.95, weight_decay=0.0, ema_decay=0.9999, grad_clip=None,
                      mixed_precision="bf16"("bf16"|"fp16"|"no"), log_every=100, ckpt_every=5000, keep_last=3, num_workers=4, seed=0
    # __post_init__ 校验（ValueError）；from_dict(d)：未知键或缺 max_steps/global_batch_size → ValueError（调用方先滤掉 train 块里的数据键）
def train_dit(model, transport, dataset, cfg: DiTTrainConfig, out_dir, *, device, logger, resume=False, log_fn=None, meta=None) -> Path
    # 单进程或 DDP（WORLD_SIZE>1：Linux+CUDA 用 nccl，否则 gloo；未初始化时自建并自毁进程组；CUDA 下 device=cuda:LOCAL_RANK）；
    #   Windows 版 torch 的 torchrun 缺 libuv 无法启动：手动设 WORLD_SIZE/RANK/LOCAL_RANK/MASTER_ADDR/MASTER_PORT 起进程；
    # 进程 seed = cfg.seed*world+rank（random/numpy/torch）；数据顺序 = DistributedSampler(num_replicas=world, rank, shuffle=True, seed=cfg.seed,
    #   drop_last=True)（world=1 也用）；每 epoch 调 sampler.set_epoch 与 dataset.set_epoch，DataLoader 用该 epoch 的 sampler 列表（续跑时切掉已用 batch）
    #   和种子为 noise_seed(cfg.seed, epoch, rank) 的 generator；dataset 项为 (x, y) 或 (x, y, index)；
    # loss = mean(transport loss) + mean(cos_loss)（与 LightningDiT train.py 一致；只有 model forward 在 CUDA autocast 下，输出转 fp32）；
    # AdamW(lr, betas=(0.9,beta2), wd)；可选全局梯度范数裁剪；梯度范数非有限 → FloatingPointError（fp16 时由 GradScaler 跳过该步）；
    # EMA 每步更新、开始时 decay=0 同步；日志（rank 0，每 log_every 步与最后一步，窗口均值）：loss_mse, loss_cos（有余弦损失时）, loss,
    #   grad_norm（裁剪前）, lr, steps_per_sec, samples_per_sec, max_mem_gb（CUDA）→ logger.info 与 log_fn(step, metrics)；
    # ckpt（rank 0，每 ckpt_every 步与 max_steps，原子写，只留最新 keep_last 个）：out_dir/checkpoints/{step:07d}.pt =
    #   {model, ema, opt, scaler（仅 CUDA fp16，否则 None）, step, epoch, batches_done_in_epoch, cfg, invariants, meta,
    #    rng: [每个 rank 的 {python, numpy, torch, cuda}]}，torch.load(weights_only=True) 可读；meta 须可 JSON 化，原样保存；
    # resume 取最新 ckpt 精确续跑；其 step > max_steps、或 invariants（global_batch_size, lr, beta2, weight_decay, ema_decay, grad_clip,
    #   mixed_precision, seed, world_size, device_type, dataset_len）改变 → ValueError；meta 改变只 warning；无 ckpt 则从头开始；
    #   resume=False 而目录已有 ckpt → ValueError；返回 out_dir/checkpoints/{max_steps:07d}.pt
@torch.no_grad()
def sample_latents(model, sample_fn, labels, noise_indices, *, base_seed, batch_size, device, latent_shape=(32,16,16),
                   mixed_precision="bf16") -> torch.Tensor
    # 无 CFG：样本 i 从 initial_noise([noise_indices[i]], ...) 出发，取 sample_fn(x0, model_fn, y=labels_batch)[-1]；
    # model_fn = model.forward（仅 CUDA 上 autocast，输出 fp32，ODE 在 fp32 中积分）；采样时 eval，结束恢复原模式；
    # 返回 CPU fp32 [N, *latent_shape] 的**归一化空间**样本（调用方反归一化）
def denormalize(x, mean, std, multiplier=1.0) -> torch.Tensor   # x·std/multiplier + mean（逐通道 dim=1；mean/std 为含 C 个值的任意数组/张量，如 [C] 或 [1,C,1,1]）；保持 x 的 dtype/device
```

### 3.10 `rwvae/integration.py`（保留 v1 `GatedVAVAE`，追加 v2）

```python
DEFAULT_LIGHTNINGDIT_ROOT = <repo>/../third_party/LightningDiT（可被环境变量 RWVAE_LIGHTNINGDIT_ROOT 覆盖，调用时读取）
def add_lightningdit_to_path(root=None) -> Path
    # root=None：先复用上次成功的 root，再 $RWVAE_LIGHTNINGDIT_ROOT，再默认值；把 checkout 放到 sys.path[0]（不重复）；
    # checkout 不完整 → FileNotFoundError；models/tokenizer/transport 已从别处 import → ImportError。
    # 此后 LightningDiT 的通用顶层包名（datasets、tools 等）会遮蔽同名包
def configure_torch_compile(enable: bool | None = None) -> bool   # None=auto（Linux+CUDA+triton 才开），经 torch._dynamo.config.disable 进程级生效；
                                                                  # 必须在 import models.lightningdit 前调用（之后改设置 → ValueError）；非 bool/None（如字符串 "auto"）→ ValueError；
                                                                  # build_lightningdit 在首次 import 前自动按 auto 调用（调用方未选择时）
def load_tokenizer_state_dict(ckpt_path) -> dict                  # torch.load(weights_only=True)，失败再 weights_only=False（日志警告：仅限可信 ckpt）；
                                                                  # 接受 Lightning {"state_dict": ...} 或普通 state dict（实际的 ldm-imagenet256-f16d32-50ep.ckpt 是后者）；
                                                                  # 只保留 encoder./decoder./quant_conv./post_quant_conv.（丢弃 loss.* 等）；缺文件 → FileNotFoundError
class FrozenTokenizer(nn.Module):                                 # LightningDiT tokenizer.autoencoder.AutoencoderKL（f16d32 结构）
    def __init__(self, autoencoder: nn.Module, meta: dict | None = None)   # 冻结 + eval
    @classmethod from_checkpoint(cls, ckpt_path, *, embed_dim=32, ch_mult=(1,1,2,2,4), device="cpu") -> "FrozenTokenizer"
        # strict 加载：missing/unexpected/形状不符 → ValueError（在 load_state_dict 之前检查）；记录 ckpt sha256；
        # meta = {tokenizer, lightningdit_root, embed_dim, ch_mult, use_variational, model_type, downsample, checkpoint_path,
        #         checkpoint_sha256, checkpoint_format("lightning"|"state_dict"), weights_only_load, num_tensors, dropped_keys}
    def encode_mean(self, images) -> torch.Tensor                 # posterior.mean（不采样）[B,32,16,16]；images 为 [-1,1] 的 float [B,3,H,W]，H、W 可被 16 整除
    def encode_moments(self, images) -> tuple[mean, logvar]       # logvar 已被上游截断到 [-30,20]
    def decode(self, z) -> torch.Tensor                           # [B,3,256,256]，不 clamp；encode_*/decode 都在 no_grad 下运行
    post_quant_conv: nn.Conv2d; decoder: nn.Module; autoencoder: nn.Module; embed_dim: int; downsample: int; meta: dict
class DinoPatchFeatures(nn.Module):                               # timm DINOv2，输出 [B,D,16,16]
    def __init__(self, model_name="vit_large_patch14_dinov2.lvd142m", *, input_size=224, device="cpu", checkpoint_path=None)
        # 预处理：[-1,1]→[0,1]→bilinear(antialias) 到 input_size→ImageNet mean/std；去掉 num_prefix_tokens；行主序 reshape；
        # 冻结 + eval；输入须为正方形；checkpoint_path 不存在 → FileNotFoundError；输出 dtype 随模型/autocast
    def forward(self, images) -> torch.Tensor ; meta: dict（model_name、checkpoint_path、input_size、patch_size、grid、num_prefix_tokens、embed_dim、mean/std、preprocess、tokens）
def build_lightningdit(model_type="LightningDiT-B/1", *, input_size=16, in_channels=32, num_classes=1000,
                       use_qknorm=False, use_swiglu=True, use_rope=True, use_rmsnorm=True, wo_shift=False,
                       class_dropout_prob=0.1, use_checkpoint=False, custom_arch: dict | None = None) -> nn.Module
    # CPU 上新初始化（learn_sigma=False，同 train.py）；custom_arch 恰为 {depth,hidden_size,num_heads,patch_size} 的正整数时直接构造 LightningDiT
    # （测试用小模型，忽略 model_type；hidden_size 须被 num_heads、input_size 须被 patch_size 整除）
def build_transport(tcfg: dict)                                   # LightningDiT create_transport；恰好接受 path_type, prediction, loss_weight, train_eps, sample_eps（必需）
                                                                  # + use_cosine_loss, use_lognorm（可选，默认 False）；未知键或会被上游静默改写的取值 → ValueError
def build_ode_sample_fn(transport, scfg: dict) -> Callable        # Sampler(transport).sample_ode(...)；读 mode（须 "ODE"）, sampling_method, num_sampling_steps（≥2）, atol, rtol,
                                                                  # reverse, timestep_shift（默认 0），其余采样协议键归调用方；返回整条轨迹 [steps, *x0.shape]，样本取 [-1]
def load_inception(device) -> nn.Module                           # pytorch_fid InceptionV3([2048 块])，eval，冻结；输入 [0,1]；输出 [[B,2048,1,1]]
def load_lpips(device) -> nn.Module                               # lpips.LPIPS(net="vgg")，eval，冻结，输入 [-1,1] → [B,1,1,1]
```

---

## 4. 脚本（`scripts/`）

所有阶段脚本：`--config` 可多次（后者覆盖前者）、`--set a.b=v` 可多次（`+a.b=v` 新建键）、`--out` 覆盖输出目录、`--device`、`--seed`、`--overwrite`、`--resume`（`runtime.add_common_args`；07 的这些通用参数写在子命令之后）。
配置栈：ImageNet-1k = `dataset_imagenet256.yaml + baseline_common.yaml + arms.yaml + server_a800.yaml`；SVP 再叠 `svp.yaml`；本地工程测试 = 三个基础配置 + `local_imagenette.yaml`。
输出默认在 `<experiment.run_root>/<experiment.name>/<阶段目录>/`。每次运行写 `config.yaml`、`log.txt`、`metrics.jsonl`，结束写 `summary.json`，并在 `manifests/run_registry.jsonl`（或配置键 `registry` 指定的文件，测试用）追加一行；记录含 `git_commit` 与 `git_dirty`（工作区有未提交改动或未忽略的未跟踪文件时为 true）。

| 脚本 | 脚本专有参数 | 输入 | 输出（相对实验目录） | 关键检查 / 退出码 |
|---|---|---|---|---|
| `00_verify_math.py` | `--quick`、`--figures`、`--registry PATH`（无需 config） | 无（合成高斯 D0） | `00_verify_math/verify_math.{json,md}`、`figures/*.png` | A 精确恒等式、B 蒙特卡洛、C 理论模型、D 守卫；任一失败 exit 1 |
| `tools/make_svp_subset.py` | `--train-root --val-root --out`、`--num-classes 10 --train-per-class 100 --val-per-class 50 --seed 2026 --mode symlink\|copy`（普通 argparse，不写 registry） | ImageNet train/val ImageFolder | `<out>/{train,val}/<wnid>/*`、`<out>/svp_subset.json` | 确定性抽样；已存在且一致则 "up to date"，不一致报错 |
| `01_prepare_manifest.py` | `--force` | train/val ImageFolder（SVP 时读 `<train_root>/../svp_subset.json`） | `cfg.dataset.manifest`、`DATA_CARD.md` §1、`01_manifest/manifest_info.json` | split 不相交、每类 dev 数、指纹；指纹变化需 `--force`；根目录移动则重写（roots_updated） |
| `02_cache_latents.py` | `--split S`（可多次）、`--num-procs N --proc-id k`（兼容 RANK/WORLD_SIZE）、`--finalize`、`--no-verify` | manifest、tokenizer ckpt | `cache.dir/<split>/{*_shard#####.safetensors, cache_index.json, cache_verification.json}`、`DATA_CARD.md` §2、`02_cache/<splits>[/proc<k>of<N>][/finalize]/` | `posterior.mean`、fp32、TF32 关；已完成分片跳过；多进程后单独 `--finalize`；抽样重编码超出存储精度界即失败 |
| `03_estimate_stats.py` | `--num-procs N --proc-id k`、`--merge`、`--max-images N`、`--allow-misaligned`、`--figures` | manifest、train_fit 缓存、DINO | `stats/{stats.npz, group_moments.pt, diagnostics.json, figures/}`、`stats/shards/acc_<k>.pt`、`DATA_CARD.md` §3 | 拒绝 val；对齐扫描峰值须在 (0,0)，否则 exit 2 且不写 stats |
| `04_solve_channel.py` | `--figures` | `stats/stats.npz`、`group_moments.pt`、arms/grid | `allocations/{<name>.npz, none.npz, allocations_summary.{json,md}, figures/}` | 恒等式残差 >1e-8（MI 1e-6）即失败；重叠矩阵、折半/bootstrap 稳定性、go/no-go |
| `05_eval_recon_probe.py` | `--split`、`--arms B0,B2`、`--R 11,22`、`--w 2`、`--max-images`、`--decoder-ckpt LABEL=PATH`、`--no-lpips`、`--no-rfid`、`--skip-probe`、`--skip-recon`、`--figures` | 缓存（eval split + probe 训练 split）、allocations、tokenizer、Inception、LPIPS | `recon_probe/<split>/{results.json, results.md, per_image.npz, ref_fid.npz, figures/}` | 一次读图服务所有臂、同噪声；探针门、R 单调性门（容差取自 `svp.pass_criteria.monotone_in_rate`，否则严格） |
| `05b_adapt_decoder.py` | `--arm`、`--R`、`--w`、`--val-images N` | allocation（B0 无）、缓存 + 图像 | `decoders/<arm_label>/{decoder.safetensors, adapt.json}` | 统一 L1+LPIPS 配方；子集不足一个 batch 报错 |
| `06_train_dit.py` | `--arm`、`--R`、`--w`、`--exploratory`（多卡用 `torchrun`） | train_fit 缓存、stats、allocation | `dit/<arm_label>/{checkpoints/*.pt, arm.json, metrics.jsonl}` | 有损臂须等于冻结的 `R*/w*`，否则需 `--exploratory`；小数据集自动重复（`arm.json` 的 `repeat`） |
| `07_evaluate.py ref-stats` | `--split`、`--max-images N`（等间距行）、`--save-features [--hflip]` | manifest 图像 | `fid_ref/<split>[_n<N>].npz`、`fid_ref/<split>[_n<N>]_features.npz` | Inception pool3，TF32 关 |
| `07_evaluate.py generate` | `--arm --R --w`、`--dit-run`、`--ckpt latest\|STEP`、`--num-samples`、`--decoder-ckpt`、`--ref-stats [LABEL=]PATH`（可多次）、`--nn-ref LABEL=NPZ`、`--nn-baseline NPZ`、`--save-npz`、`--save-grid K` | 06 运行目录、参考统计 | `gen/<arm_label>/step<k>_n<N>/{results.json, results.md, gen_fid_stats.npz, samples.npz, grid.png}` | `cfg_scale` 必须为 1；类均衡标签、臂间同初始噪声；`fid_kind`（FID-10K/50K/N）永不混表；单进程 |
| `07_evaluate.py report` | 无（重跑需 `--overwrite`） | `recon_probe/*`、`allocations/allocations_summary.json`、`dit/*`、`gen/**`、缓存校验与 `stats/diagnostics.json` | `report/{REPORT.md, report.json, figures/}` | 有 `svp` 配置块时输出 “SVP verdict” 一节 |

流水线驱动：`scripts/run_pipeline.sh svp|imagenet`（Linux 8 卡节点）与 `scripts/run_local_imagenette.ps1`（Windows 本机），一阶段一条命令、遇错即停，`--from STAGE` / `-From STAGE` 从中途继续。
ImageNet-1k 数据：`scripts/tools/imagenet_parquet_to_folder.py`（HF `ILSVRC/imagenet-1k` parquet → ImageFolder）。

### 4.1 脚本详细规格（2026-11-17 设计；实现状态见 `../auto-encoder/handoff_v2.md`）

通用：`main(argv=None) -> int`，`if __name__ == "__main__": raise SystemExit(main())`；先 `import _common`；
`cfg = _common.load_cfg(args)` 必须在导入重型第三方代码之前；`run = _common.open_run(__file__, cfg, out_dir, args)`；
异常时 `run.finish("failed", error=repr(e))` 再抛出；重型构造一律经模块属性调用（`_common.make_tokenizer`、
`_common.make_semantic`、`_common.build_dit`、`integration.load_inception/load_lpips`），便于测试 monkeypatch；
缓存/统计/评测调用 `_common.set_tf32(False)` 并把返回值写进产物 meta；测试用 tmp 配置设置 `registry`，绝不写真实
`manifests/run_registry.jsonl`。

- **00_verify_math**（无需 config；`--quick`、`--figures`、`--registry`）：A 精确恒等式（C=32，四类谱 × equal_snr/variance/r2(w=1.5,3) ×
  R∈{0,1e-3,11,22,44,全激活}）；B 蒙特卡洛（C=8，N=4e5，quick 6e4；高斯与非高斯源）：逐 token 协方差、P5 平均 KL=R、
  经验 LMMSE=A、经验 latent MSE=D_latent、跨 token 协方差 = A·Cov·Aᵀ；C 理论模型（d_s=6,d_n=10,ρ=6）：Σ_w 谱、
  R*_w 阈值、R=0.8R* 时信号保留 = 0/0.24/0.47/0.68/0.83（w=1/1.5/2/3/5，闭式已手算核对）+ MC 探针 R²、样本估计的 K 恢复信号子空间；
  D 守卫（非 PD、违反 P1、R<0、w<1 报错）。输出 verify_math.{json,md}，`--figures` 出相图/κ 曲线；全通过 exit 0。
- **01_prepare_manifest**：`build_manifest` → `validate()`；已存在则重建比对指纹（相同=up to date，不同需 `--force`）；
  写 `<exp>/DATA_CARD.md` 的“## 1. Dataset and splits”节（计数、指纹、类映射 sha、预处理、§1.2 不泄漏条款、许可说明）。
- **02_cache_latents**：`--split`（可多次）、`--num-procs N --proc-id k`（分片 id % N == k；兼容 RANK/WORLD_SIZE）、`--finalize`、`--no-verify`；
  已完成分片跳过；`encode_mean`（fp32、TF32 关）；flip 用 `torch.flip(images, dims=[-1])`（= 中心裁剪后翻转，与 LightningDiT 顺序一致，需单测证明）；
  全部分片完成后 `finalize` + 抽样重编码比对（超过存储精度界即失败）；写 DATA_CARD “## 2. Latent cache”。
- **03_estimate_stats**：拒绝 val；`ImageLatentPairDataset` → DINO（bf16 autocast）→ 联合 token 进 `GroupedMomentAccumulator`
  （组 = `image_group_id(image_id)`；include_flip 时翻转图与 mu_flip 归入同一组）；`--num-procs/--proc-id` 存 `stats/shards/acc_<k>.pt`，
  `--merge` 合并；`stats.npz`（`with_explained(ridge_rel)`）+ `stats/group_moments.pt`；diagnostics.json：Σ 谱/有效秩/条件数（不做收缩）、
  Σ_y 秩、ridge_scan 的 CCA 谱、K 谱、折半稳定性（Σ、Σ_expl 相对 F 范数、top-8 主夹角）、对齐扫描（峰值须在 (0,0)，否则 exit 2，除非 `--allow-misaligned`）；
  写 DATA_CARD “## 3. Semantic statistics”。对齐扫描只保留前 `alignment_check.num_images` 张图（CPU float32）。
- **04_solve_channel**：对每个有损臂 × `grid.R_nats` ×（r2 的）`grid.w` 解 `solve_allocation` 存 `allocations/<name>.npz`（另存 `none.npz`）；
  每个分配：bpp、激活方向数、θ、D_latent、加权失真、语义保留、方向相关度、恒等式残差（超 1e-8 即失败，MI 地板除外）；
  每个 R 的规则间保留重叠；折半与组 bootstrap（`stats.bootstrap` 次）保留矩阵距离 vs 规则间距离；按 `go_no_go` 判定可分/稳定；
  输出 `allocations/allocations_summary.{json,md}`（`summary.json` 归 `runtime.Run`；`--figures` 出 κ 曲线与重叠热图）。
- **05_eval_recon_probe**：`--split`（默认 eval.split；选规则用 train_dev，终报才用 val）、`--arms`、`--R`、`--w`、`--max-images`、
  `--decoder-ckpt LABEL=PATH`、`--no-lpips`、`--no-rfid`、`--skip-probe`、`--skip-recon`、`--figures`；**一次遍历图像服务所有臂**；
  所有臂同噪声 `noise_seed(eval.noise_seed, 0, i)`；每图 PSNR（to_uint8）、LPIPS、各臂 FIDStatistics + 原图参考统计（缓存为 ref_fid.npz）、
  经验 latent MSE vs D_latent、前 dct_images 张的 DCT 带功率比；线性探针（训练：probe.train_split 每类 train_per_class 张，噪声种子 epoch 0；
  测试：eval split，epoch 1；直接读缓存 mu）；成对 bootstrap：B3(w)−B2、B3(w)−B1、B2−B1；探针门：B3−B2 ≥ probe_diff_points 且 CI 不含 0；
  输出 `recon_probe/<split>/results.{json,md}`、per_image.npz。
- **05b_adapt_decoder**：单臂（B0 即 B0-adapt 对照）；数据为 decoder_adapt.split 每类 images_per_class 张的确定性子集；lmmse 融合初始化；
  `finetune_decoder` + `integration.load_lpips`；存 `decoders/<arm_label>/decoder.safetensors`（meta：臂、分配指纹、tokenizer sha、manifest 指纹、配方、子集哈希、步数、种子）；
  可选 `--val-images N` 记录微调前后 PSNR/LPIPS。
- **06_train_dit**：`--arm --R --w --exploratory --resume`；先 `_common.preregistration_check`；`ChannelLatentDataset`（B0 channel=None；
  无 flip 缓存时 flip_prob=0 并记日志）；`_common.build_dit(cfg, num_classes=len(manifest.classes))`、`integration.build_transport`、
  `_common.dit_train_config(cfg)`、`generation.train_dit(..., log_fn=run.log_metrics)`；仅 rank 0 建 Run 并写 `arm.json`（臂、规则、R、w、
  分配名与指纹、stats 指纹、训练 split 的 cache_index sha、归一化 mean/std、预登记记录、world size、噪声/flip 协议、exploratory）。
  Windows 上 torchrun 不可用（torch 2.12 TCPStore 需 libuv），多卡只在 Linux 节点。
- **07_evaluate**：`ref-stats --split S [--max-images N]` → `fid_ref/<S>[_n<N>].npz`；`generate --arm --R --w [--dit-run] [--ckpt latest|STEP]
  [--num-samples] [--decoder-ckpt] [--ref-stats] [--save-npz] [--save-grid K]`：EMA 权重、`cfg_scale` 必须为 1（否则拒绝）、
  `class_balanced_labels(N, K, label_seed)`、初始噪声索引 0..N−1（臂间相同）、`sample_latents` → `denormalize` → 臂的解码器 → to_uint8 →
  FIDStatistics → gFID；结果写 `gen/<arm_label>/step<k>_n<N>/results.json`（fid_kind：FID-10K/FID-50K/FID-N，永不混表）；
  `--save-npz` 写 ADM 格式 `arr_0`；`--nn-ref LABEL=特征npz` 与 `--nn-baseline` 给出样本到参考集的最近邻（余弦与 L2，SVP 记忆检查用，特征由 `ref-stats --save-features [--hflip]` 生成）；`report` 汇总 recon_probe 与 gen 结果为 `report/REPORT.md`（多种子取均值±标准差；本地 Imagenette 结果在表头标明为工程测试）。


### 4.2 SVP（sanity-validation pilot，第一次真实运行）

目的：在正式 ImageNet-1k 运行前，用一个随机小子集把 00 → 07 全部跑通，并让每个指标都有显而易见的预期值。它不是论文结果，不与 ImageNet-1k 混表（报告抬头会标明）。

- 数据：`make_svp_subset.py` 从 ImageNet train 随机取 10 类 × 100 张，另取这 10 类各 50 张官方 val（seed 2026）。`dev_per_class=20`，得到 train_fit 800、train_dev 200、val 500。
- 配置：`svp.yaml`（叠在 `server_a800.yaml` 之后）只改数据路径、缓存目录、实验名 `svp` 和小预算。生成器配方仍是 LightningDiT-B/1，但 `max_steps 20000`（约 6400 个 epoch）、`ema_decay 0.999`，故意过拟合 800 个 latent。
- 顺序（`run_pipeline.sh svp`）：00 → 子集 → 01 → 02（8 进程分片 + `--finalize`）→ 03（8 进程分片 + `--merge`）→ 04 → 05（train_dev，全网格）→ 05b（B0-adapt）→ 06（torchrun，B0、B2@R22、B3@R22 w2，均 `--exploratory`）→ 07 ref-stats（train_fit 500 张；train_fit 特征含翻转；val 及其特征）→ 07 generate（每臂对 train/val 两个参考算 gFID，对 train_fit 做最近邻）→ 07 report。
- 判据：`svp.pass_criteria`，在 07 report 的 “SVP verdict” 一节逐条给出数值、阈值和 PASS/FAIL；缺输入的判据列为 “Not evaluated”，不会算作通过。

| 判据 | 数据来源 | 阈值（运行前固定） |
|---|---|---|
| `dit_loss_drop` | `dit/*/metrics.jsonl` | 最后 1k 步均值 / 前 1k 步均值 ≤ 0.8 |
| `dit_loss_plateau` | 同上 | 最后两个 2k 窗口均值的相对变化 ≤ 3% |
| `gfid_memorisation` | B0 末步 `gen/.../results.json` | gFID(train_fit 500) / gFID(val 500) ≤ 0.7 |
| `nn_memorisation` | 同上（`--nn-ref`、`--nn-baseline`） | 样本→train_fit 的 L2 最近邻中位数 ≤ 0.8 × val→train_fit 的中位数 |
| `allocation_identity_residual` | `allocations/allocations_summary.json` | 所有分配的恒等式残差 ≤ 1e-8 |
| `latent_mse_vs_d_latent` | `recon_probe/*/results.json` | 每个有损臂 \|经验 latent MSE − D_latent\| / D_latent ≤ 0.05 |
| `monotone_in_rate` | 同上 | PSNR、探针 top-1 随 R 不降（容差 0.05 dB / 1 点），latent MSE 不升 |
| `b0_reconstruction` | `recon_probe/train_dev/results.json` 的 B0 | PSNR ∈ [23, 32] dB 且 LPIPS ≤ 0.25 |
| `data_gates` | `cache/<split>/cache_verification.json`、`stats/diagnostics.json` | 每个 split 重编码校验通过，对齐峰值在 (0,0) |

`tests/test_pipeline_smoke.py` 在 CPU 上用合成图像、微型 tokenizer 和假 DINO/Inception/LPIPS 跑同一条链（含 `svp.yaml`），断言每阶段产物存在、九条判据全部可评估、registry 每阶段一行。

---

## 5. 配置（`configs/`）

- `dataset_imagenet256.yaml`：路径、split 规则、tokenizer、cache、DINO、stats 设定（ImageNet-1k 正式实验）。
- `baseline_common.yaml`：DiT 结构与训练配方、transport、采样协议、decoder 适配配方、评测与探针协议。
- `arms.yaml`：B0–B3 定义、R 网格 `{11,22,33,44}`、w 网格 `{1.5,2,3}`、预登记块（`frozen/R_star/w_star`、最小效应阈值）。
- `local_imagenette.yaml`：**仅工程测试**（Imagenette 10 类，ImageNet 子集）：本地路径、小预算。其结果不得与 ImageNet-1k 混表，也不得作为论文结论。

---

## 6. 测试清单（`tests/`）

`test_allocation.py`、`test_relevance_v2.py`、`test_stats.py`、`test_channel.py`、`test_decoder_adapter.py`、
`test_dataset_contract.py`、`test_evaluation.py`、`test_generation.py`、`test_integration_v2.py`、`test_runtime.py`，
以及跨模块端到端的 `test_cross_module.py`（stats → allocation → channel → decoder_adapter / datasets / generation / evaluation，
折半/bootstrap 组子集 → allocation 稳定性，各阶段文件的保存/读回，`import rwvae` 的惰性加载）。
需要第三方依赖的测试用 `pytest.importorskip`；需要真实权重的测试标 `@pytest.mark.real_assets`（marker 与 `testpaths` 在 `pyproject.toml`
注册），仅在环境变量 `RWVAE_ASSETS` 指向资产目录时运行（`pytest -m real_assets`）。其中 `test_cross_module` 的真实资产测试在
Imagenette train_dev 的 80 张图上核对计划 §1.2 的数据门：DINOv2 16×16 patch 网格（224 px 输入）与 f16d32 latent 网格（256 px 输入）
的 `alignment_shift_scan` 峰值在 (0,0)。精确恒等式以 float64 相对误差 ≤1e-8 检查，Monte-Carlo 检查的容差按样本量给出（5 个标准差）。
v1 的 29 个测试必须继续通过。

---

## 7. 未实现（候选池，计划 §9）

DCT 频带信道、可学率门、R1/R3、isotropic/top-k/placebo/decoder-sens 臂、GAN 适配、CFG-gFID/CMMD/sFID/IS/PR、
第二 tokenizer、DiT-XL、encoder 联合适配。`B_pub` 用上游 LightningDiT 脚本复现，不在本代码内重写。
