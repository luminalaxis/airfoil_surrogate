"""회귀 모델 학습·평가 (afs train).

입력: 형상 계수(cst16 = 면당 8개 × 2, 또는 cst8 = 면당 4개 × 2) + 받음각 aoa [deg] + log10(Re)
목표: cl, log10_cd

모델
  baseline : 얇은 익형 이론 CL = 2π(α − α0) (α0는 형상의 캠버선으로 계산), log10 CD = 학습 데이터 평균
  lgbm     : LightGBM. 목표별로 평균 모델 + 분위 회귀(5%, 95%) 모델 → 90% 예측 구간
  mlp      : PyTorch 다층 퍼셉트론. 시드만 다른 n_seeds개 앙상블, 평균 = 예측, 표준편차 × 1.645 → 90% 예측 구간
             (n_seeds = 1이면 표준편차를 정할 수 없어 예측 구간을 내지 않는다)

평가 (runs/ml/<이름>/metrics.csv)
  분할(val, test, test_ood)마다 조건별 표본 수와 MAE(cl), MAE(log10_cd), 90% 예측 구간 폭·실제 포함률.
  MAE와 포함률에는 형상 단위 부트스트랩 95% 신뢰구간(*_lo95, *_hi95)을 붙인다.
    all          학습에 쓰는 수렴 상태(train_states)의 전체 케이스
    attached     CL 최대 받음각 − near_stall_window보다 작은 받음각 (부착 유동)
    near_stall   형상·Re별 CL 최대 받음각 ± near_stall_window
    post_stall   CL 최대 받음각 + near_stall_window보다 큰 받음각 (실속 후 박리 유동)
                 (셋 다 CFD 정답의 CL 최대 받음각으로 정해지므로 평가 조건으로만 쓰고 모델 입력에는 넣지 않음)
    state=<상태> 수렴 상태별 (converged / oscillating). diverged(발산)는 학습·평가에서 빼고 건수(n)만 보고
"""

from __future__ import annotations

import json
import math
import os
import zlib
from dataclasses import asdict, dataclass, field
from statistics import NormalDist

import numpy as np

from .config import ConfigError, ProjectConfig
from .fileutil import write_json_atomic
from .dataset import MLConfig, dataset_dir, load_dataset

TARGETS = ("cl", "log10_cd")
MODEL_NAMES = ("baseline", "lgbm", "mlp")
PI_LEVEL = 0.90                                       # 예측 구간 (기획서: 90% 구간의 실제 포함률 보고)
PI_TAG = f"{round(PI_LEVEL * 100)}"                   # 열 이름 width90_*, cover90_*
Z_PI = NormalDist().inv_cdf(0.5 + PI_LEVEL / 2)       # 1.645: 앙상블 표준편차 → 90% 구간
QUANTILES = ((1 - PI_LEVEL) / 2, (1 + PI_LEVEL) / 2)  # LightGBM 분위 회귀 5%, 95%
N_BOOT = 2000                                         # 형상 단위 부트스트랩 반복 수 (95% 신뢰구간)
EVAL_SPLITS = ("val", "test", "test_ood")
DIVERGED = "diverged"
ML_DIR = "ml"
# predictions.csv에 함께 남기는 데이터셋 열 (경계 사례 분석용). yplus* 열도 함께 남긴다.
CARRY_COLUMNS = ("conv_cl_std", "conv_cd_std", "conv_state_tsst", "cl_tsst", "log10_cd_tsst",
                 "d_cl_tsst", "d_log10_cd_tsst")


@dataclass
class TrainOptions:
    models: list = field(default_factory=lambda: ["baseline", "lgbm", "mlp"])
    features: str = "cst16"           # cst16 / cst8
    hidden: list = field(default_factory=lambda: [64, 64])
    n_seeds: int = 5
    epochs: int = 2000
    patience: int = 100
    lr: float = 1e-3
    batch_size: int = 256
    stall_weight: float = 1.0         # 실속 부근 학습 케이스 가중치 (개선 실험)
    ood_train_fraction: float = 0.0   # test_ood 형상 중 이 비율을 학습에 추가 (개선 실험)
    name: str | None = None

    def validate(self) -> None:
        def bad(what: str) -> ConfigError:
            return ConfigError(f"train 옵션 {what}")

        for key, minimum in (("n_seeds", 1), ("epochs", 1), ("patience", 1), ("batch_size", 1)):
            v = getattr(self, key)
            if not isinstance(v, int) or v < minimum:
                raise bad(f"{key}는 {minimum} 이상인 정수여야 합니다: {v!r}")
        if not self.hidden or not all(isinstance(h, int) and h > 0 for h in self.hidden):
            raise bad(f"hidden은 양의 정수 목록이어야 합니다: {self.hidden!r}")
        if not (math.isfinite(self.lr) and self.lr > 0):
            raise bad(f"lr은 0보다 커야 합니다: {self.lr!r}")
        if not (math.isfinite(self.stall_weight) and self.stall_weight > 0):
            raise bad(f"stall_weight는 0보다 커야 합니다: {self.stall_weight!r}")
        if not 0 <= self.ood_train_fraction <= 1:
            raise bad(f"ood_train_fraction은 0~1이어야 합니다: {self.ood_train_fraction!r}")


def feature_columns(rows: list[dict], features: str) -> list[str]:
    prefix = f"{features}_"
    shape = [k for k in rows[0] if k.startswith(prefix) and len(k) > len(prefix) + 1
             and k[len(prefix)] in ("u", "l") and k[len(prefix) + 1:].isdigit()]
    if not shape:
        raise ConfigError(f"데이터셋에 {features}_* 열이 없습니다. [ml] cst_orders를 확인하고 dataset을 다시 만드세요.")
    return shape + ["aoa", "log10_re"]


def _matrix(rows: list[dict], cols: list[str]) -> np.ndarray:
    return np.array([[r[c] for c in cols] for r in rows], dtype=float)


def _check_finite(rows: list[dict], cols: list[str], what: str) -> None:
    """입력·목표에 빈 값/NaN/inf가 있으면 학습 전에 멈춘다 (None은 _matrix에서 NaN이 되어 MLP 손실이 NaN이 됨)."""
    bad = []
    for r in rows:
        missing = [c for c in cols if not (isinstance(r.get(c), (int, float)) and math.isfinite(r[c]))]
        if missing:
            bad.append(f"{r.get('case')} ({', '.join(missing[:3])})")
    if bad:
        raise ConfigError(f"{what} 케이스 {len(bad)}개에 빈 값 또는 숫자가 아닌 값이 있습니다: {'; '.join(bad[:5])}"
                          " - dataset을 다시 만드세요.")


def _scale(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """표준화용 평균·표준편차. 학습 데이터에서 값이 하나뿐인 열(예: Re 한 가지)은 표준편차 1로 둔다.
    그러지 않으면 1e-12로 나누게 되어, 다른 값으로 예측할 때 입력이 10^12배로 커진다."""
    mean, std = a.mean(0), a.std(0)
    tiny = std <= 1e-8 * np.maximum(1.0, np.abs(mean))
    return mean, np.where(tiny, 1.0, std)


# ---------------------------------------------------------------- 모델

class Baseline:
    name = "baseline"

    def fit(self, train: list[dict], val: list[dict], cols: list[str], weights: np.ndarray) -> None:
        self.cd = float(np.mean([r["log10_cd"] for r in train]))

    def predict(self, rows: list[dict]) -> dict:
        _check_finite(rows, ["aoa", "alpha0_thin_deg"], "baseline 평가")
        a = np.radians([r["aoa"] - r["alpha0_thin_deg"] for r in rows])
        return {"cl": 2 * np.pi * a, "log10_cd": np.full(len(rows), self.cd)}


class LGBMModel:
    name = "lgbm"
    params = dict(learning_rate=0.03, num_leaves=31, min_data_in_leaf=10, bagging_fraction=0.8,
                  bagging_freq=1, feature_fraction=0.9, seed=0, verbosity=-1)
    max_rounds = 3000

    def fit(self, train, val, cols, weights) -> None:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ConfigError("lightgbm이 없습니다: pip install -e \".[ml]\"") from exc
        self.cols = cols
        x, xv = _matrix(train, cols), _matrix(val, cols)
        self.models = {}
        for t in TARGETS:
            y, yv = np.array([r[t] for r in train]), np.array([r[t] for r in val])
            for kind, extra in (("mean", {"objective": "regression"}),
                                ("lo", {"objective": "quantile", "alpha": QUANTILES[0]}),
                                ("hi", {"objective": "quantile", "alpha": QUANTILES[1]})):
                dtrain = lgb.Dataset(x, y, weight=weights, free_raw_data=False)
                dval = lgb.Dataset(xv, yv, reference=dtrain)
                self.models[(t, kind)] = lgb.train(
                    {**self.params, **extra}, dtrain, num_boost_round=self.max_rounds, valid_sets=[dval],
                    callbacks=[lgb.early_stopping(100, verbose=False)])

    def predict(self, rows):
        x = _matrix(rows, self.cols)
        out = {}
        for t in TARGETS:
            pred = {k: self.models[(t, k)].predict(x, num_iteration=self.models[(t, k)].best_iteration)
                    for k in ("mean", "lo", "hi")}
            out[t] = pred["mean"]
            out[f"{t}_lo"], out[f"{t}_hi"] = np.minimum(pred["lo"], pred["hi"]), np.maximum(pred["lo"], pred["hi"])
        return out


class MLPEnsemble:
    name = "mlp"

    def __init__(self, opts: TrainOptions):
        self.opts = opts

    def fit(self, train, val, cols, weights) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ConfigError("PyTorch가 없습니다: pip install -e \".[ml]\"") from exc
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        self.cols = cols
        x, xv = _matrix(train, cols), _matrix(val, cols)
        y = np.array([[r[t] for t in TARGETS] for r in train])
        yv = np.array([[r[t] for t in TARGETS] for r in val])
        self.xm, self.xs = _scale(x)
        self.ym, self.ys = _scale(y)
        tx = torch.tensor((x - self.xm) / self.xs, dtype=torch.float32)
        ty = torch.tensor((y - self.ym) / self.ys, dtype=torch.float32)
        tw = torch.tensor(weights / weights.mean(), dtype=torch.float32)
        vx = torch.tensor((xv - self.xm) / self.xs, dtype=torch.float32)
        vy = torch.tensor((yv - self.ym) / self.ys, dtype=torch.float32)
        self.nets, self.history = [], []
        for seed in range(self.opts.n_seeds):
            net, best = self._train_one(torch, seed, tx, ty, tw, vx, vy)
            self.nets.append(net)
            self.history.append(best)

    def _net(self, torch, n_in: int):
        layers, width = [], n_in
        for h in self.opts.hidden:
            layers += [torch.nn.Linear(width, h), torch.nn.SiLU()]
            width = h
        layers.append(torch.nn.Linear(width, len(TARGETS)))
        return torch.nn.Sequential(*layers)

    def _train_one(self, torch, seed, tx, ty, tw, vx, vy):
        torch.manual_seed(seed)
        gen = torch.Generator().manual_seed(seed)
        net = self._net(torch, tx.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=self.opts.lr, weight_decay=1e-5)
        best, best_state, wait = math.inf, None, 0
        n = len(tx)
        for epoch in range(self.opts.epochs):
            net.train()
            perm = torch.randperm(n, generator=gen)
            for i in range(0, n, self.opts.batch_size):
                idx = perm[i:i + self.opts.batch_size]
                loss = ((net(tx[idx]) - ty[idx]) ** 2).mean(1).mul(tw[idx]).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(((net(vx) - vy) ** 2).mean()) if len(vx) else float(loss)
            if not math.isfinite(v) or not math.isfinite(loss.item()):
                raise ConfigError(f"MLP 학습 손실이 발산했습니다 (seed {seed}, epoch {epoch + 1}, lr {self.opts.lr:g}). "
                                  "학습률을 낮추거나 입력 데이터를 확인하세요.")
            if v < best - 1e-6:
                best, wait = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                wait += 1
                if wait >= self.opts.patience:
                    break
        net.load_state_dict(best_state)
        return net, {"seed": seed, "val_loss": best, "epochs": epoch + 1}

    def predict(self, rows):
        import torch

        x = torch.tensor((_matrix(rows, self.cols) - self.xm) / self.xs, dtype=torch.float32)
        with torch.no_grad():
            preds = np.stack([net(x).numpy() * self.ys + self.ym for net in self.nets])
        mean, std = preds.mean(0), preds.std(0, ddof=1) if len(self.nets) > 1 else None
        out = {}
        for k, t in enumerate(TARGETS):
            out[t] = mean[:, k]
            if std is not None:  # 시드 1개로는 앙상블 분산을 정할 수 없음 → 구간 없음 (폭 0 구간을 내지 않음)
                out[f"{t}_lo"], out[f"{t}_hi"] = mean[:, k] - Z_PI * std[:, k], mean[:, k] + Z_PI * std[:, k]
        return out


# ---------------------------------------------------------------- 평가

class ShapeBootstrap:
    """형상 단위 부트스트랩: 형상을 복원 추출하고, 뽑힌 형상의 케이스 전체로 평균을 다시 계산한다.

    같은 형상의 받음각·Re 케이스는 서로 독립이 아니므로 케이스가 아니라 형상을 재표집한다.
    재표집은 (seed, key, 형상 집합)으로만 정해지므로 같은 분할·조건이면 모델·실행이 달라도 같은
    재표집을 쓴다. 두 모델의 오차 차이에 쓰면 짝지은(paired) 비교가 된다.
    """

    def __init__(self, shapes, key: str, seed: int = 0, n_boot: int = N_BOOT):
        names, self.index = np.unique(np.asarray(shapes, dtype=str), return_inverse=True)
        self.n_shapes = len(names)
        rng = np.random.default_rng([seed, zlib.crc32(key.encode("utf-8"))])
        self.weights = rng.multinomial(self.n_shapes, np.full(self.n_shapes, 1 / self.n_shapes), size=n_boot)
        self.counts = self.weights @ np.bincount(self.index, minlength=self.n_shapes)

    def samples(self, values) -> np.ndarray:
        """재표집마다의 평균 (n_boot개)."""
        sums = np.bincount(self.index, weights=np.asarray(values, float), minlength=self.n_shapes)
        return self.weights @ sums / self.counts

    def mean_ci(self, name: str, values) -> dict:
        point = float(np.mean(values))
        lo, hi = np.percentile(self.samples(values), [2.5, 97.5])
        # 형상이 1개면 모든 재표집이 같아 합산 순서 차이(마지막 자리)만 남으므로 점추정을 포함하도록 맞춘다
        return {name: point, f"{name}_lo95": min(float(lo), point), f"{name}_hi95": max(float(hi), point)}


def flow_regime(row: dict) -> str:
    """CFD CL 최대 받음각 기준 유동 구간: attached / near_stall / post_stall."""
    if row["near_stall"]:
        return "near_stall"
    if row.get("aoa_clmax") is not None and row["aoa"] > row["aoa_clmax"]:
        return "post_stall"
    return "attached"


def conditions(row: dict, train_states: set[str]) -> list[str]:
    out = [f"state={row['conv_state']}"]
    if row["conv_state"] in train_states:
        out += ["all", flow_regime(row)]
    return out


def evaluate(rows: list[dict], preds: dict[str, np.ndarray], model: str, train_states: set[str],
             seed: int = 0) -> list[dict]:
    groups: dict[tuple, list[int]] = {}
    for i, r in enumerate(rows):
        for cond in conditions(r, train_states):
            groups.setdefault((r["eval_split"], cond), []).append(i)
    metrics = []
    for (split, cond), idx in sorted(groups.items()):
        boot = ShapeBootstrap([rows[i]["shape"] for i in idx], f"{split}|{cond}", seed)
        entry = {"model": model, "split": split, "condition": cond, "n": len(idx), "n_shapes": boot.n_shapes}
        for t in TARGETS:
            y = np.array([rows[i][t] for i in idx])
            p = preds[t][idx]
            entry.update(boot.mean_ci(f"mae_{t}", np.abs(p - y)))
            if f"{t}_lo" in preds:
                lo, hi = preds[f"{t}_lo"][idx], preds[f"{t}_hi"][idx]
                entry[f"width{PI_TAG}_{t}"] = float(np.mean(hi - lo))
                entry.update(boot.mean_ci(f"cover{PI_TAG}_{t}", (y >= lo) & (y <= hi)))
        metrics.append(entry)
    return metrics


def count_only(rows: list[dict], model: str) -> list[dict]:
    """발산 케이스: 정답을 믿을 수 없어 학습·평가에서 빼고 분할별 건수만 보고한다 (기획서 라벨 규칙)."""
    out = []
    for split in EVAL_SPLITS:
        sub = [r for r in rows if r["eval_split"] == split]
        if sub:
            out.append({"model": model, "split": split, "condition": f"state={DIVERGED}",
                        "n": len(sub), "n_shapes": len({r["shape"] for r in sub})})
    return out


def _split_rows(rows: list[dict], ml: MLConfig, opts: TrainOptions) -> dict[str, list[dict]]:
    usable = set(ml.train_states)
    labeled = [r for r in rows if r.get("cl") is not None and r.get("log10_cd") is not None]
    ood_shapes = sorted({r["shape"] for r in labeled if r["split"] == "test_ood"})
    moved: set[str] = set()
    if opts.ood_train_fraction > 0 and ood_shapes:
        rng = np.random.default_rng(ml.seed)
        k = int(round(opts.ood_train_fraction * len(ood_shapes)))
        moved = set(rng.choice(ood_shapes, size=k, replace=False).tolist())
    for r in rows:
        r["eval_split"] = "train" if r["shape"] in moved else r["split"]
    # 발산은 건수만 보고 (train_states에 diverged를 직접 넣었으면 다른 상태처럼 학습·평가)
    count_diverged = DIVERGED not in usable
    scored = [r for r in labeled if not (count_diverged and r["conv_state"] == DIVERGED)]
    return {
        "train": [r for r in scored if r["eval_split"] == "train" and r["conv_state"] in usable],
        "val": [r for r in scored if r["eval_split"] == "val" and r["conv_state"] in usable],
        "eval": [r for r in scored if r["eval_split"] in EVAL_SPLITS],
        "diverged": [r for r in rows if count_diverged and r["conv_state"] == DIVERGED
                     and r["eval_split"] in EVAL_SPLITS],
        "moved": sorted(moved),
    }


def train(cfg: ProjectConfig, opts: TrainOptions) -> dict:
    opts.validate()
    ml = MLConfig.from_dict(cfg.ml)
    rows = load_dataset(dataset_dir(cfg) / "dataset.csv")
    if not rows:
        raise ConfigError("데이터셋이 비어 있습니다 (완료된 케이스가 없음).")
    cols = feature_columns(rows, opts.features)
    parts = _split_rows(rows, ml, opts)
    train_rows, val_rows, eval_rows = parts["train"], parts["val"], parts["eval"]
    if not train_rows or not val_rows:
        raise ConfigError(f"학습/검증 케이스가 부족합니다 (train {len(train_rows)}, val {len(val_rows)}).")
    _check_finite(train_rows, [*cols, *TARGETS], "학습")
    _check_finite(val_rows, [*cols, *TARGETS], "검증")
    _check_finite(eval_rows, [*cols, *TARGETS], "평가")
    weights = np.array([opts.stall_weight if r["near_stall"] else 1.0 for r in train_rows])

    name = opts.name or f"{opts.features}_h{'x'.join(map(str, opts.hidden))}"
    out_dir = cfg.output_dir / ML_DIR / name
    warnings = []
    previous = out_dir / "run.json"
    if previous.is_file():
        try:
            old = json.loads(previous.read_text(encoding="utf-8")).get("options", {})
        except (OSError, ValueError):
            old = {}
        if old and old != asdict(opts) | {"name": old.get("name")}:
            warnings.append(f"이전 결과({name})를 다른 옵션의 결과로 덮어썼습니다. 비교하려면 --name 으로 이름을 나누세요.")
    out_dir.mkdir(parents=True, exist_ok=True)
    usable = set(ml.train_states)

    makers = {"baseline": Baseline, "lgbm": LGBMModel, "mlp": lambda: MLPEnsemble(opts)}
    unknown = set(opts.models) - set(makers)
    if unknown:
        raise ConfigError(f"알 수 없는 모델: {', '.join(sorted(unknown))} ({', '.join(MODEL_NAMES)})")
    metrics, predictions, extra = [], {}, {}
    for m in opts.models:
        model = makers[m]()
        model.fit(train_rows, val_rows, cols, weights)
        preds = model.predict(eval_rows)
        predictions[m] = preds
        entries = evaluate(eval_rows, preds, m, usable, ml.seed) + count_only(parts["diverged"], m)
        metrics += sorted(entries, key=lambda e: (EVAL_SPLITS.index(e["split"]), e["condition"]))
        if m == "mlp":
            extra["mlp_members"] = model.history

    from .dataset import _write_csv
    _write_csv(out_dir / "metrics.csv", metrics)
    carry = [k for k in rows[0] if k in CARRY_COLUMNS or k.startswith("yplus")]
    pred_rows = []
    for i, r in enumerate(eval_rows):
        row = {k: r[k] for k in ("case", "shape", "family", "eval_split", "aoa", "reynolds",
                                 "conv_state", "near_stall", "aoa_clmax", "cl", "log10_cd")}
        row.update({k: r.get(k) for k in carry})
        for m, p in predictions.items():
            for key, arr in p.items():
                row[f"{m}_{key}"] = float(arr[i])
        pred_rows.append(row)
    _write_csv(out_dir / "predictions.csv", pred_rows)
    write_json_atomic(out_dir / "run.json", {
        "options": asdict(opts), "features": cols, "targets": list(TARGETS),
        "n_train": len(train_rows), "n_val": len(val_rows),
        "n_train_shapes": len({r["shape"] for r in train_rows}),
        "pi_level": PI_LEVEL, "n_boot": N_BOOT, "n_diverged_excluded": len(parts["diverged"]),
        "ood_shapes_moved_to_train": parts["moved"], **extra,
    })
    diverged = {s: sum(1 for r in parts["diverged"] if r["eval_split"] == s) for s in EVAL_SPLITS}
    return {"dir": out_dir, "metrics": metrics, "n_train": len(train_rows), "n_val": len(val_rows),
            "features": cols, "moved": parts["moved"], "diverged": diverged, "warnings": warnings}
