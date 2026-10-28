"""
分析 CrossGapEnsemble (cross_gap_moe) 的 router 權重分配與年齡差距的關係。

流程：
1. 載入 checkpoints/cross_gap_moe/exp2/cross_gap_moe_router_circle_seed42/best.pt 權重。
2. 從 Vox-O, Vox1-H.S, Vox-CA5, Vox-CA10, Vox-CA15, Vox-CA20 測試協議中抽樣配對樣本，
   計算每個樣本的實際年齡差距 (age gap)。
3. 將樣本丟入模型取得 router 分配給三個專家 (small/medium/large) 的權重。
4. 畫圖：x 軸為年齡差距，y 軸為 router 權重，三條折線分別代表三個專家
   （以年齡差距分箱後的平均權重連線呈現），並疊加線性回歸趨勢線粗略估計走向。
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

from model.disentangled_model.dynamic_ensemble import CrossGapEnsemble, ExpertCheckpointPaths
from params.param import DATASET_INFO, DEVICE

warnings.filterwarnings(
    "ignore",
    message=r".*torchaudio\.load_with_torchcodec.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*StreamingMediaDecoder has been deprecated.*",
    category=UserWarning,
)

# ==========================================
# 設定
# ==========================================
CKPT_PATH = Path("checkpoints/cross_gap_moe/exp2/cross_gap_moe_router_circle_seed42/best.pt")
DATASETS: list[str] = ["Vox-O", "Vox1-H.S", "Vox-CA5", "Vox-CA10", "Vox-CA15", "Vox-CA20"]
MAX_SAMPLES_PER_DATASET = 1500
RANDOM_SEED = 42

OUTPUT_DIR = Path("logs/cross_gap_moe")
OUTPUT_CSV = OUTPUT_DIR / "router_weight_vs_age_gap.csv"
OUTPUT_PLOT = OUTPUT_DIR / "router_weight_vs_age_gap.png"


def build_model(device: torch.device) -> CrossGapEnsemble:
    """建立與訓練時相同設定的 CrossGapEnsemble 模型。"""
    return CrossGapEnsemble(
        C=1024,
        feature_dim=192,
        router_hidden_dim=256,
        router_dropout=0.1,
        router_input_mode="embedding_diff",
        score_aggregation_mode="weighted",
        expert_ckpt_paths=ExpertCheckpointPaths(),
    ).to(device)


def load_checkpoint(model: CrossGapEnsemble, ckpt_path: Path, device: torch.device) -> None:
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    load_result = model.load_state_dict(state_dict, strict=True)
    print("\n" + "=" * 66)
    print(f"已載入 router checkpoint: {ckpt_path}")
    print(f"{'missing_keys':<16}: {len(load_result.missing_keys)}")
    print(f"{'unexpected_keys':<16}: {len(load_result.unexpected_keys)}")
    print("=" * 66)


def load_age_lookup(meta_csv: str) -> dict[tuple[str, str], int]:
    """讀取 VoxCeleb1 metadata CSV，建立 (speaker_id, utterance) -> age 查詢表。"""
    lookup: dict[tuple[str, str], int] = {}
    df = pd.read_csv(meta_csv, sep=",", header=0, usecols=[0, 1, 2, 3], dtype={"age": str})
    for _, row in tqdm(df.iterrows(), total=len(df), desc="載入年齡 metadata", leave=False):
        speaker = str(row.iloc[0]).strip()
        utt = str(row.iloc[1]).strip()
        try:
            age = int(row.iloc[2])
        except (TypeError, ValueError):
            continue
        if age < 0 or age > 120:
            continue
        lookup[(speaker, utt)] = age
    return lookup


def find_audio_path(relative_path: str, audio_dirs: list[str]) -> str:
    for audio_dir in audio_dirs:
        audio_path = Path(audio_dir) / relative_path
        if audio_path.exists():
            return str(audio_path)
    raise FileNotFoundError(f"找不到音檔: {relative_path}")


def sample_pairs(
    dataset_key: str,
    age_lookup: dict[tuple[str, str], int],
    audio_dirs: list[str],
    max_samples: int,
    rng: random.Random,
) -> list[tuple[str, str, int, int, str]]:
    """從指定測試協議的配對檔中隨機抽樣，並附上實際年齡差距。"""
    datalist_path = DATASET_INFO["VoxCeleb1"][dataset_key]["AUDIO_DATALIST"]
    with open(datalist_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    rng.shuffle(lines)

    samples: list[tuple[str, str, int, int, str]] = []
    for line in lines:
        if len(samples) >= max_samples:
            break
        parts = line.split()
        if len(parts) < 3:
            continue

        label = int(parts[0])
        rel1, rel2 = parts[1], parts[2]
        parts1 = Path(rel1).parts
        parts2 = Path(rel2).parts
        if len(parts1) < 2 or len(parts2) < 2:
            continue

        age1 = age_lookup.get((parts1[0], parts1[1]))
        age2 = age_lookup.get((parts2[0], parts2[1]))
        if age1 is None or age2 is None:
            continue

        try:
            wav1_path = find_audio_path(rel1, audio_dirs)
            wav2_path = find_audio_path(rel2, audio_dirs)
        except FileNotFoundError:
            continue

        samples.append((wav1_path, wav2_path, abs(age1 - age2), label, dataset_key))

    return samples


def load_audio_tensor(wav_path: str) -> torch.Tensor:
    signal, sr = torchaudio.load(wav_path)
    if sr != 16000:
        signal = T.Resample(orig_freq=sr, new_freq=16000)(signal)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    return signal.squeeze(0)


@torch.no_grad()
def compute_router_weights(
    model: CrossGapEnsemble,
    samples: list[tuple[str, str, int, int, str]],
    device: torch.device,
) -> list[dict]:
    rows: list[dict] = []
    for wav1_path, wav2_path, age_gap, label, dataset_key in tqdm(samples, desc="Router 權重推論", leave=False):
        try:
            wav1 = load_audio_tensor(wav1_path).unsqueeze(0).to(device)
            wav2 = load_audio_tensor(wav2_path).unsqueeze(0).to(device)
        except Exception as e:
            print(f"[警告] 讀取音檔失敗，已跳過: {e}")
            continue

        _, weights = model(wav1, wav2, spec_aug=False, return_details=False)
        w = weights.squeeze(0).cpu().numpy()
        rows.append(
            {
                "dataset": dataset_key,
                "age_gap": age_gap,
                "label": label,
                "weight_small": float(w[0]),
                "weight_medium": float(w[1]),
                "weight_large": float(w[2]),
            }
        )
    return rows


def plot_router_weight_vs_age_gap(df: pd.DataFrame, output_path: Path) -> None:
    experts = [
        ("weight_small", "Small Expert", "tab:blue"),
        ("weight_medium", "Medium Expert", "tab:green"),
        ("weight_large", "Large Expert", "tab:red"),
    ]

    plt.figure(figsize=(9, 6))
    for col, label, color in experts:
        # Regression line only, to roughly estimate the weight trend across age gaps.
        x = df["age_gap"].values.astype(float)
        y = df[col].values.astype(float)
        coeff = np.polyfit(x, y, 1)
        poly = np.poly1d(coeff)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = poly(x_line)
        plt.plot(
            x_line,
            y_line,
            linestyle="-",
            linewidth=2,
            color=color,
            label=f"{label}",
        )

    plt.xlabel("Age Gap", fontsize=12)
    plt.ylabel("Meta Learner Weight", fontsize=12)
    plt.title("Meta Learner Weight vs Age Gap (Linear Regression per Expert)", fontsize=13, fontweight="bold")
    plt.grid(True, linestyle=":", alpha=0.4)
    plt.legend(loc="best", fontsize=9, framealpha=0.9)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


if __name__ == "__main__":
    rng = random.Random(RANDOM_SEED)
    device = torch.device(DEVICE)

    print("建立 CrossGapEnsemble 模型並載入 checkpoint...")
    model = build_model(device)
    load_checkpoint(model, CKPT_PATH, device)
    model.eval()

    meta_csv = DATASET_INFO["VoxCeleb1"]["AUDIO_META_DIR"]
    print("載入年齡 metadata...")
    age_lookup = load_age_lookup(meta_csv)

    all_rows: list[dict] = []
    for dataset_key in DATASETS:
        ds_info = DATASET_INFO["VoxCeleb1"].get(dataset_key)
        if ds_info is None:
            print(f"[警告] 找不到資料集設定，跳過: {dataset_key}")
            continue

        audio_dirs = ds_info["AUDIO_DIR"]
        print(f"\n抽樣資料集: {dataset_key} (最多 {MAX_SAMPLES_PER_DATASET} 筆)")
        samples = sample_pairs(dataset_key, age_lookup, audio_dirs, MAX_SAMPLES_PER_DATASET, rng)
        print(f"  取得 {len(samples)} 筆有效樣本 (具備年齡標籤且音檔存在)")

        rows = compute_router_weights(model, samples, device)
        all_rows.extend(rows)

    if not all_rows:
        raise RuntimeError("沒有取得任何有效樣本，請確認資料路徑與 metadata 設定是否正確。")

    df = pd.DataFrame(all_rows)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n✓ 已儲存 router 權重明細至 {OUTPUT_CSV} (共 {len(df)} 筆)")

    plot_router_weight_vs_age_gap(df, OUTPUT_PLOT)
    print(f"✓ 圖表已儲存至 {OUTPUT_PLOT}")
