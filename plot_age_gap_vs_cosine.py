import os
from pathlib import Path
import warnings
import sys
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import torchaudio
import torchaudio.transforms as T
from tqdm import tqdm

from params.param import DATASET_INFO, MODEL_ID, DEVICE
from model.feature_extractor.hf_ecapa_tdnn import SpeakerEmbeddingExtractor
from model.disentangled_model.linear_decorr_mlp import LinearDecorrMLP

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

mpl.rcParams["font.sans-serif"] = [
    "Microsoft JhengHei",
    "Microsoft YaHei",
    "Noto Sans CJK TC",
    "Noto Sans CJK SC",
    "SimHei",
    "DejaVu Sans",
]
mpl.rcParams["axes.unicode_minus"] = False

# ==========================================
# 基本設定
# ==========================================
DEVICE = torch.device(DEVICE if isinstance(DEVICE, str) else str(DEVICE))
DATASET_NAME_DEFAULT = "Vox-O"  # 預設資料集
MAX_SAMPLES = None  # None 表示全部，或設定數字限制

# 模型架構設定（與訓練時一致）
SPEAKER_EMB_DIM = 192
AGE_DIM = 1
MLP_HIDDEN_DIMS = [256]
MLP_OUTPUT_DIM = 193
NUM_SPEAKERS = 5128
NUM_AGE_GROUPS = 7


def find_checkpoint_in_exp(exp_path: Path):
    """在 exp 資料夾中尋找 epoch_020_model.pth，若無則回傳 best_model.pth 或 last_model.pth"""
    if not exp_path.exists():
        return None
    
    # 優先找 epoch_020_model.pth
    epoch_020 = exp_path / "epoch_020_model.pth"
    if epoch_020.exists():
        return epoch_020
    
    # 其次找 best_model.pth 或 last_model.pth
    for name in ("best_model.pth", "last_model.pth"):
        p = exp_path / name
        if p.exists():
            return p
    
    return None


def load_checkpoint_to_model(model: torch.nn.Module, ckpt_path: Path, device: torch.device):
    """從 checkpoint 載入模型權重"""
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint.get("model", checkpoint)
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(cleaned, strict=False)
    print(f"✓ Checkpoint 已載入: {ckpt_path}")


def load_raw_age_lookup(meta_path: str):
    """讀取 VoxCeleb1 metadata，建立 (speaker_id, utterance) -> 原始年齡 的查詢表。"""
    lookup = {}
    df = pd.read_csv(meta_path, sep=",", header=0, usecols=[0, 1, 2, 3], dtype={"age": str})

    for _, row in tqdm(df.iterrows(), total=len(df), desc="載入年齡 metadata"):
        speaker = row["speaker_id"] if "speaker_id" in df.columns else row.iloc[0]
        utt = row["utterance"] if "utterance" in df.columns else row.iloc[1]
        age_str = row["age"] if "age" in df.columns else row.iloc[2]

        try:
            lookup[(str(speaker).strip(), str(utt).strip())] = float(age_str)
        except (TypeError, ValueError):
            continue

    return lookup


def find_audio_path(relative_path, audio_dirs):
    for audio_dir in audio_dirs:
        audio_path = Path(audio_dir) / relative_path
        if audio_path.exists():
            return str(audio_path)
    raise FileNotFoundError(f"Audio file {relative_path} not found in any of the provided directories.")


def load_audio_tensor(wav_path: str):
    signal, sr = torchaudio.load(wav_path)
    if sr != 16000:
        resampler = T.Resample(orig_freq=sr, new_freq=16000)
        signal = resampler(signal)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    return signal.squeeze(0)


def compute_pair_scores(pair_lines, raw_age_lookup, audio_dirs, speaker_extractor, model, device, pair_filter, max_samples=None, desc=None):
    """計算指定 pair 類型的 age-gap 與 cosine similarity。"""
    age_gaps = []
    cosines = []

    with torch.no_grad():
        total = len(pair_lines) if hasattr(pair_lines, "__len__") else None
        for line in tqdm(pair_lines, desc=desc, total=total):
            parts = line.split(" ")
            is_same = int(parts[0])
            if not pair_filter(is_same):
                continue

            spk1_rel_path = parts[1]
            spk2_rel_path = parts[2].strip()

            spk1_parts = Path(spk1_rel_path).parts
            spk2_parts = Path(spk2_rel_path).parts
            if len(spk1_parts) < 2 or len(spk2_parts) < 2:
                continue

            spk1_id = spk1_parts[0]
            spk2_id = spk2_parts[0]
            spk1_utt = spk1_parts[1]
            spk2_utt = spk2_parts[1]

            a1 = raw_age_lookup.get((spk1_id, spk1_utt))
            a2 = raw_age_lookup.get((spk2_id, spk2_utt))
            if a1 is None or a2 is None:
                continue

            wav1_path = find_audio_path(spk1_rel_path, audio_dirs)
            wav2_path = find_audio_path(spk2_rel_path, audio_dirs)

            wav1 = load_audio_tensor(wav1_path).to(device)
            wav2 = load_audio_tensor(wav2_path).to(device)

            emb1 = speaker_extractor(wav1.unsqueeze(0))
            emb2 = speaker_extractor(wav2.unsqueeze(0))

            out1 = model(emb1)["z"]
            out2 = model(emb2)["z"]

            v1 = out1.squeeze().cpu().numpy()
            v2 = out2.squeeze().cpu().numpy()

            num = np.dot(v1, v2)
            den = np.linalg.norm(v1) * np.linalg.norm(v2)
            cosine = float(num / (den + 1e-12))

            age_gaps.append(abs(a1 - a2))
            cosines.append(cosine)

            if max_samples and len(age_gaps) >= max_samples:
                break

    return age_gaps, cosines


def plot_regression_comparison(results_per_exp, exp_labels, output_plot, title, subtitle=None):
    """將多個實驗的 regression 線與相關係數畫在同一張圖上。"""
    if len(results_per_exp) == 0:
        print("⚠ 找不到任何實驗結果可用來繪圖。")
        return

    all_mins = [min(v[0]) for v in results_per_exp.values() if len(v[0]) > 0]
    all_maxs = [max(v[0]) for v in results_per_exp.values() if len(v[0]) > 0]
    if not all_mins or not all_maxs:
        print("⚠ 沒有有效的 age_gap 值可繪圖。")
        return

    x_min = min(all_mins)
    x_max = max(all_maxs)
    x_line = np.linspace(x_min, x_max, 300)

    plt.figure(figsize=(8, 6))
    colors = ["red", "blue", "green", "orange"]
    for idx, (exp, (ages, cos)) in enumerate(results_per_exp.items()):
        if len(ages) < 2:
            print(f"⚠ {exp} 樣本太少，跳過繪製。")
            continue
        exp_label = exp_labels.get(exp, exp)
        age_np = np.array(ages, dtype=float)
        cos_np = np.array(cos, dtype=float)

        coeff = np.polyfit(age_np, cos_np, 1)
        poly = np.poly1d(coeff)
        y_line = poly(x_line)

        y_pred = poly(age_np)
        ss_res = np.sum((cos_np - y_pred) ** 2)
        ss_tot = np.sum((cos_np - cos_np.mean()) ** 2)
        r2 = 1.0 - ss_res / (ss_tot + 1e-12)
        corr = float(np.corrcoef(age_np, cos_np)[0, 1]) if np.std(age_np) > 1e-12 and np.std(cos_np) > 1e-12 else float("nan")

        plt.scatter(
            age_np,
            cos_np,
            color=colors[idx % len(colors)],
            alpha=0.25,
            s=18,
        )
        plt.plot(
            x_line,
            y_line,
            color=colors[idx % len(colors)],
            lw=2,
            label=(
                f"{exp_label}  R²={r2:.3f}  r={corr:.3f}"
            ),
        )

    plt.xlabel("Age-gap (Actual Age Difference)", fontsize=12)
    plt.ylabel("Cosine Similarity", fontsize=12)
    if subtitle:
        plt.suptitle(title, fontsize=13, fontweight='bold')
        plt.title(subtitle, fontsize=11, pad=10)
    else:
        plt.title(title, fontsize=13)
    plt.grid(True, linestyle=':', alpha=0.4)
    plt.legend(
        loc="best",
        fontsize=10,
        frameon=True,
        fancybox=True,
        framealpha=0.9,
        labelspacing=0.9,
        borderpad=0.8,
        handlelength=2.4,
    )
    plt.tight_layout()
    plt.savefig(output_plot, dpi=200)
    plt.close()
    print(f"✓ 圖表已儲存為 {output_plot}")


if __name__ == "__main__":
    # ==========================================
    # 命令列參數解析
    # ==========================================
    parser = argparse.ArgumentParser(
        description="Analysis of Age-Gap vs Cosine Similarity for different datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DATASET_NAME_DEFAULT,
        help=f"Dataset name (default: {DATASET_NAME_DEFAULT}). Available: Vox-O, Vox-E, Vox-H, Vox-CA5, Vox-CA10, Vox-CA15, Vox-CA20",
    )
    args = parser.parse_args()
    DATASET_NAME = args.dataset

    # 動態生成輸出檔名
    ds_suffix = DATASET_NAME.lower().replace("-", "_")
    OUTPUT_PLOT = f"age_gap_vs_cosine_{ds_suffix}_negative.svg"
    OUTPUT_PLOT_POS = f"age_gap_vs_cosine_{ds_suffix}_positive.svg"

    print("=" * 100)
    print(f"Age-Gap vs Cosine Similarity Analysis ({DATASET_NAME} Dataset)")
    print("=" * 100)

    cuda_available = torch.cuda.is_available()
    if DEVICE.type == "cuda" and not cuda_available:
        raise RuntimeError("CUDA was selected, but PyTorch cannot access CUDA.")
    print(f"CUDA available: {cuda_available}")
    print(f"Using device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
    else:
        print("CUDA is not in use; running on CPU.")
    
    # ==========================================
    # 初始化資料集
    # ==========================================
    ds_info = DATASET_INFO.get("VoxCeleb1", {}).get(DATASET_NAME)
    if ds_info is None:
        raise RuntimeError(f"找不到 DATASET_INFO 中的 {DATASET_NAME}")

    audio_dirs = ds_info["AUDIO_DIR"]
    audio_datalist = ds_info["AUDIO_DATALIST"]
    meta_csv = DATASET_INFO["VoxCeleb1"]["AUDIO_META_DIR"]

    print(f"\n載入原始年齡查詢表...")
    raw_age_lookup = load_raw_age_lookup(meta_csv)

    print(f"讀取 {DATASET_NAME} 配對檔...")
    with open(audio_datalist, "r", encoding="utf-8") as f:
        pair_lines = f.readlines()[:20000]

    # ==========================================
    # 初始化模型
    # ==========================================
    print(f"初始化說話者特徵抽取器 ({MODEL_ID})...")
    speaker_extractor = SpeakerEmbeddingExtractor(MODEL_ID, device=str(DEVICE))
    speaker_extractor.eval()

    # 我們要在同一張圖上比較兩個實驗的線性趨勢線（不繪出原始資料點）
    CHECKPOINT_DIR = Path("checkpoints/linear_decorr_mlp")
    EXPS = ["exp44","exp35"]
    EXP_LABELS = {
        "exp44": "No decoupling", # (λ=0)
        # "exp15": "λ=10",
        "exp35": "Decoupling", # (λ=130)
    }

    results_per_exp = {}
    results_per_exp_pos = {}

    for exp in tqdm(EXPS, desc="Experiments"):
        exp_path = CHECKPOINT_DIR / exp
        ckpt = find_checkpoint_in_exp(exp_path)
        if ckpt is None:
            print(f"⚠ 在 {exp_path} 找不到有效 checkpoint，跳過 {exp}")
            continue

        label = EXP_LABELS.get(exp, exp)
        print(f"載入模型: {label} -> {ckpt}")
        model = LinearDecorrMLP(
            input_dim=SPEAKER_EMB_DIM,
            hidden_dims=MLP_HIDDEN_DIMS,
            output_dim=MLP_OUTPUT_DIM,
            num_speakers=NUM_SPEAKERS,
            num_age_groups=NUM_AGE_GROUPS,
            age_dim=AGE_DIM,
        )
        model.to(DEVICE)
        model.eval()
        load_checkpoint_to_model(model, ckpt, DEVICE)

        neg_age_gaps, neg_cosines = compute_pair_scores(
            pair_lines,
            raw_age_lookup,
            audio_dirs,
            speaker_extractor,
            model,
            DEVICE,
            pair_filter=lambda is_same: is_same == 0,
            max_samples=MAX_SAMPLES,
            desc=f"{exp} negative",
        )
        pos_age_gaps, pos_cosines = compute_pair_scores(
            pair_lines,
            raw_age_lookup,
            audio_dirs,
            speaker_extractor,
            model,
            DEVICE,
            pair_filter=lambda is_same: is_same == 1,
            max_samples=MAX_SAMPLES,
            desc=f"{exp} positive",
        )

        results_per_exp[exp] = (neg_age_gaps, neg_cosines)
        results_per_exp_pos[exp] = (pos_age_gaps, pos_cosines)
        print(f"✓ {exp} 完成：負對樣本數 = {len(neg_age_gaps)}，正對樣本數 = {len(pos_age_gaps)}")

    # ==========================================
    # 將兩個實驗的線性回歸趨勢線畫在同一張圖上（不繪製資料點）
    # ==========================================
    plot_regression_comparison(
        results_per_exp,
        EXP_LABELS,
        OUTPUT_PLOT,
        f"{DATASET_NAME} Negative Pairs: Regression Comparison",
        subtitle="Different speaker pairs with varying age gaps",
    )
    plot_regression_comparison(
        results_per_exp_pos,
        EXP_LABELS,
        OUTPUT_PLOT_POS,
        f"{DATASET_NAME} Positive Pairs: Regression Comparison",
        subtitle="Same speaker pairs (age gap = 0 in most cases)",
    )

    print("\n" + "=" * 100)
    print("分析完成！")
    print("=" * 100)
