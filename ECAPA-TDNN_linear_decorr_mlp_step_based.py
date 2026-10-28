import csv
import os
import re
import warnings
from pathlib import Path
import random

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.vox2_loader import Vox2Dataset
from data.vox1_loader import PairwiseDataset
from model.disentangled_model.linear_decorr_mlp import (
    LinearDecorrLoss,
    LinearDecorrMLP,
)
from model.feature_extractor.hf_ecapa_tdnn import SpeakerEmbeddingExtractor
from params.param import BATCH_SIZE, DATASET_INFO, MODEL_ID
from tool.EER import compute_eer
from tool.linear_decorr_eval_utils import (
    compute_age_auc,
    eval_classification_network,
    eval_network,
    plot_age_corr_heatmap,
)
from tool.checkpoint_manager import CheckpointManager
from tool.linear_decorr_training_utils import (
    MarginScheduler,
    RunningAgeCorrelation,
    WarmupExpDecayLR,
)

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

seed=[42, 1337, 3407,2025,9527]

GLOBAL_SEED = 42

def set_seed(seed=GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(GLOBAL_SEED)

# ==========================================
# 1) 基本設定
# ==========================================
CUDA_AVAILABLE = torch.cuda.is_available()
if not CUDA_AVAILABLE:
    raise RuntimeError("未偵測到 CUDA 裝置，無法使用 GPU 訓練。")

DEVICE = torch.device("cuda")
print(f"CUDA available: {CUDA_AVAILABLE}")
print(f"Using device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(DEVICE)}")
TRAIN_DATASET_NAME = "VoxCeleb2"
VAL_DATASET_NAME = "Vox-CA5"

EPOCHS = 20
NUM_WORKERS = 0
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-5
MAX_EVAL_PAIRS = 20000

# Step-based scheduler 預設超參數（依 WeSpeaker 常用配方）
LR_SCHED_TOTAL_STEPS = 10000
LR_SCHED_WARMUP_STEPS = 500
LR_SCHED_ETA_0 = 0.1
LR_SCHED_ETA_T = 1e-5
MARGIN_SCHED_T1 = 2000
MARGIN_SCHED_T2 = 6000
MARGIN_SCHED_M = 0.2

SPEAKER_EMB_DIM = 192
MLP_HIDDEN_DIMS = [256]
AGE_DIM = 1
MLP_ID_DIM = 192
MLP_OUTPUT_DIM = AGE_DIM + MLP_ID_DIM  # 前 AGE_DIM 維做年齡分類，其餘做說話者分類與驗證
MLP_DROPOUT = 0.1

LAMBDA_SPK = 1.0
LAMBDA_AGE = 1.0
LAMBDA_DECORR = 50.0
LAMBDA_DECORR_WARMUP_EPOCHS = 0
LAMBDA_DECORR_START = 0.0

ARCFACE_S = 64
ARCFACE_M = 0.2
ARCFACE_EASY_MARGIN = False

EARLY_STOPPING_ENABLED = False
EARLY_STOPPING_PATIENCE = 10
TOPK_CORR = 5
AGE_GROUP_BOUNDS = [21, 31, 41, 51, 61, 71, 81]
PLOT_EVERY_N_EPOCHS = 5
SPLIT_SEED = GLOBAL_SEED
TRAIN_UTTS_PER_SPK = 8
VAL_UTTS_PER_SPK = 2


def get_next_exp_dir(base_dir: str, prefix: str = "exp"):
    """在 base_dir 下以 expXXX 形式建立下一個實驗資料夾，並回傳 (path, name)。"""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)$")
    max_idx = 0
    for child in base_path.iterdir():
        if not child.is_dir():
            continue
        match = pattern.match(child.name)
        if match:
            max_idx = max(max_idx, int(match.group(1)))

    next_idx = max_idx + 1
    exp_name = f"{prefix}{next_idx}"
    exp_path = base_path / exp_name
    exp_path.mkdir(parents=True, exist_ok=False)
    return str(exp_path), exp_name


def get_warmup_lambda_decorr(epoch: int) -> float:
    """線性 warmup：在前幾個 epoch 將 lambda_decorr 從起始值提升到目標值。"""
    if LAMBDA_DECORR_WARMUP_EPOCHS <= 0:
        return float(LAMBDA_DECORR)

    if epoch >= LAMBDA_DECORR_WARMUP_EPOCHS:
        return float(LAMBDA_DECORR)

    progress = float(epoch + 1) / float(LAMBDA_DECORR_WARMUP_EPOCHS)
    return float(LAMBDA_DECORR_START + progress * (LAMBDA_DECORR - LAMBDA_DECORR_START))


def print_age_neuron_correlation_stats(age_corr, tag, age_dim: int = AGE_DIM):
    """打印年齡神經元與其他神經元的相關係數統計"""
    if age_corr is None:
        return
    
    age_corr_with_others = np.abs(age_corr[:, age_dim:])
    if age_corr_with_others.size == 0:
        return
    
    print(f"\n[{tag}] |Age Neuron Correlations| with Other Neurons (age_dim={age_dim})")
    print(f"  Min |correlation|:   {age_corr_with_others.min():.6f}")
    print(f"  Max |correlation|:   {age_corr_with_others.max():.6f}")
    print(f"  Mean |correlation|:  {age_corr_with_others.mean():.6f}")
    print(f"  Std |correlation|:   {age_corr_with_others.std():.6f}")
    flat_values = np.sort(age_corr_with_others.reshape(-1))[::-1]
    print(f"  Top-{TOPK_CORR} |correlations|:  {flat_values[:TOPK_CORR].tolist()}")


def build_train_val_indices_by_speaker(
    dataset,
    train_utts_per_spk: int = 8,
    val_utts_per_spk: int = 2,
    seed: int = SPLIT_SEED,
):
    """以 speaker 為單位做固定數量切分。"""
    if train_utts_per_spk <= 0 or val_utts_per_spk <= 0:
        raise ValueError("train_utts_per_spk 與 val_utts_per_spk 必須 > 0")

    required = train_utts_per_spk + val_utts_per_spk
    speaker_to_indices = {}
    for idx, (_, speaker_id, _, _) in enumerate(dataset.datalist):
        speaker_to_indices.setdefault(speaker_id, []).append(idx)

    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    used_speakers = []

    for speaker_id, indices in speaker_to_indices.items():
        if len(indices) < required:
            raise RuntimeError(
                f"Speaker {speaker_id} 的語句數不足 {required}，"
                "請確認 Vox2Dataset 的 min_utts_per_speaker / 取樣設定。"
            )

        permuted = rng.permutation(indices)
        train_indices.extend(permuted[:train_utts_per_spk].tolist())
        val_indices.extend(permuted[train_utts_per_spk:required].tolist())
        used_speakers.append(speaker_id)

    return train_indices, val_indices, used_speakers


print("Preparing datasets...")
full_dataset = Vox2Dataset(
    audio_dir=DATASET_INFO[TRAIN_DATASET_NAME]["AUDIO_DIR"],
    audio_meta_dir=DATASET_INFO[TRAIN_DATASET_NAME]["AUDIO_META_DIR"],
    musan_path=DATASET_INFO["MUSAN"]["AUDIO_DIR"],
    rir_path=DATASET_INFO["RIR"]["AUDIO_DIR"],
    suffix=DATASET_INFO[TRAIN_DATASET_NAME]["audio_suffix"],
    age_target_mode="group",
)

train_indices, val_indices, used_speakers = build_train_val_indices_by_speaker(
    full_dataset,
    train_utts_per_spk=TRAIN_UTTS_PER_SPK,
    val_utts_per_spk=VAL_UTTS_PER_SPK,
    seed=SPLIT_SEED,
)

if len(used_speakers) == 0:
    raise RuntimeError(
        "沒有任何 speaker 能滿足 8 train + 2 val 的切分條件，"
        "請增加每位 speaker 可用語句或調整切分比例。"
    )

train_dataset = Subset(full_dataset, train_indices)
val_dataset = Subset(full_dataset, val_indices)

train_speaker_ids = sorted({full_dataset.datalist[idx][1] for idx in train_indices})
val_speaker_ids = sorted({full_dataset.datalist[idx][1] for idx in val_indices})

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available(),
)

# 使用 PairwiseDataset 讀取測試資料集（原先的驗證流程保留不變）
test_dataset = PairwiseDataset(
    audio_dir=DATASET_INFO["VoxCeleb1"][VAL_DATASET_NAME]["AUDIO_DIR"],
    audio_meta_dir=DATASET_INFO["VoxCeleb1"][VAL_DATASET_NAME]["AUDIO_DATALIST"],
    audio_meta_csv_path=DATASET_INFO["VoxCeleb1"]["AUDIO_META_DIR"],
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=torch.cuda.is_available(),
)

print(
    f"Split summary | Eligible speakers: {len(used_speakers)}"
)
print(
    f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)} | "
    f"Test pairs: {len(test_dataset)}"
)
print(
    f"Train speaker classes: {len(train_speaker_ids)} | "
    f"Val speaker classes: {len(val_speaker_ids)}"
)
print(
    f"Train/Val speaker class overlap: {len(set(train_speaker_ids) & set(val_speaker_ids))}"
)
print(f"Building models on {DEVICE}...")

speaker_extractor = SpeakerEmbeddingExtractor(model_id=MODEL_ID, device=str(DEVICE))
speaker_extractor.eval()
speaker_extractor.requires_grad_(False)

model = LinearDecorrMLP(
    input_dim=SPEAKER_EMB_DIM,
    hidden_dims=MLP_HIDDEN_DIMS,
    output_dim=MLP_OUTPUT_DIM,
    num_speakers=len(full_dataset.speaker2idx),
    num_age_groups=full_dataset.num_age_classes,
    dropout=MLP_DROPOUT,
    age_dim=AGE_DIM,
).to(DEVICE)

criterion = LinearDecorrLoss(
    latent_id_dim=MLP_OUTPUT_DIM - AGE_DIM,
    num_speakers=len(full_dataset.speaker2idx),
    lambda_spk=LAMBDA_SPK,
    lambda_age=LAMBDA_AGE,
    lambda_decorr=LAMBDA_DECORR,
    arcface_s=ARCFACE_S,
    arcface_m=ARCFACE_M,
    arcface_easy_margin=ARCFACE_EASY_MARGIN,
).to(DEVICE)

optimizer = torch.optim.Adam(
    list(model.parameters()) + list(criterion.parameters()),
    lr=LR_SCHED_ETA_0,
    weight_decay=WEIGHT_DECAY,
)

for param_group in optimizer.param_groups:
    param_group["lr"] = 0.0

lr_scheduler = WarmupExpDecayLR(
    optimizer=optimizer,
    total_steps=LR_SCHED_TOTAL_STEPS,
    warmup_steps=LR_SCHED_WARMUP_STEPS,
    eta_0=LR_SCHED_ETA_0,
    eta_T=LR_SCHED_ETA_T,
)

margin_scheduler = MarginScheduler(
    t1=MARGIN_SCHED_T1,
    t2=MARGIN_SCHED_T2,
    target_margin=MARGIN_SCHED_M,
)

log_root_dir = "logs/linear_decorr_mlp"
checkpoint_root_dir = "checkpoints/linear_decorr_mlp"
log_dir, exp_name = get_next_exp_dir(log_root_dir, prefix="exp")
checkpoint_dir = os.path.join(checkpoint_root_dir, exp_name)
os.makedirs(checkpoint_dir, exist_ok=True)

print(f"Experiment directory: {log_dir}")
print(f"Checkpoint directory: {checkpoint_dir}")

age_corr_plot_dir = os.path.join(log_dir, "age_correlation_plots")
train_plot_dir = os.path.join(age_corr_plot_dir, "train")
eval_plot_dir = os.path.join(age_corr_plot_dir, "eval")
os.makedirs(train_plot_dir, exist_ok=True)
os.makedirs(eval_plot_dir, exist_ok=True)

writer = SummaryWriter(log_dir=log_dir)

csv_path = os.path.join(log_dir, "train_test_metrics.csv")
csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(
    [
        "epoch",
        "global_step",
        "lr",
        "arcface_margin_current",
        "lambda_decorr_current",
        "train_total_loss",
        "train_spk_loss",
        "train_age_loss",
        "train_decorr_loss",
        "train_spk_acc",
        "train_age_acc",
        "train_age_corr_abs_mean",
        "val_total_loss",
        "val_spk_loss",
        "val_age_loss",
        "val_decorr_loss",
        "val_spk_acc",
        "val_age_acc",
        "val_age_auc",
        "val_age_corr_abs_mean",
        "test_eer_before",
        "test_eer_after",
        "test_age_auc",
        "test_age_corr_abs_mean",
        "best_test_eer_after",
    ]
)

best_test_EER = float("inf")
best_epoch = -1
no_improve_epochs = 0
checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

print("Start training (Linear-Decorr MLP)...")
global_step = 0
for epoch in range(EPOCHS):
    model.train()
    current_lambda_decorr = get_warmup_lambda_decorr(epoch)
    criterion.lambda_decorr = current_lambda_decorr

    train_total_loss = 0.0
    train_spk_loss = 0.0
    train_age_loss = 0.0
    train_decorr_loss = 0.0
    train_correct_spk = 0
    train_correct_age = 0
    train_total_samples = 0
    train_corr_meter = RunningAgeCorrelation(latent_dim=MLP_OUTPUT_DIM, age_dim=AGE_DIM)
    current_margin = margin_scheduler.get_margin(global_step)

    for waveform, label_spk, _, label_age in tqdm(train_loader, desc=f"Train {epoch + 1}/{EPOCHS}"):
        current_margin = margin_scheduler.step()
        criterion.arcface.set_margin(current_margin)

        waveform = waveform.to(DEVICE, non_blocking=True)
        label_spk = label_spk.to(DEVICE, non_blocking=True)
        label_age = label_age.to(DEVICE, non_blocking=True).long()

        with torch.no_grad():
            speaker_emb = speaker_extractor(waveform)

        outputs = model(speaker_emb)
        loss, loss_dict = criterion(outputs=outputs, target_spk=label_spk, target_age=label_age)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        global_step += 1

        train_total_loss += float(loss.item())
        train_spk_loss += float(loss_dict["loss_spk"].item())
        train_age_loss += float(loss_dict["loss_age"].item())
        train_decorr_loss += float(loss_dict["loss_decorr"].item())
        
        # 用串流方式記錄年齡神經元相關係數，避免每個 batch 重複做完整相關矩陣
        train_corr_meter.update(outputs["z"])

        pred_spk = loss_dict["arcface_logits"].argmax(dim=1)
        pred_age = outputs["logits_age"].argmax(dim=1)
        train_correct_spk += (pred_spk == label_spk).sum().item()
        train_correct_age += (pred_age == label_age).sum().item()
        train_total_samples += label_spk.size(0)

    train_total_loss /= max(1, len(train_loader))
    train_spk_loss /= max(1, len(train_loader))
    train_age_loss /= max(1, len(train_loader))
    train_decorr_loss /= max(1, len(train_loader))
    train_spk_acc = 100.0 * train_correct_spk / max(1, train_total_samples)
    train_age_acc = 100.0 * train_correct_age / max(1, train_total_samples)
    
    avg_age_corr = train_corr_meter.correlations()

    val_metrics = eval_classification_network(
        model=model,
        speaker_extractor=speaker_extractor,
        val_dataloader=val_loader,
        criterion=criterion,
        latent_dim=MLP_OUTPUT_DIM,
        age_dim=AGE_DIM,
        device=DEVICE,
    )

    test_eer_before, test_eer_after, before_embs, after_embs, final_ids, test_age_corr, test_age_auc = eval_network(
        model=model,
        speaker_extractor=speaker_extractor,
        eval_dataloader=test_dataloader,
        num_age_groups=full_dataset.num_age_classes,
        latent_dim=MLP_OUTPUT_DIM,
        age_dim=AGE_DIM,
        device=DEVICE,
    )

    train_age_corr_values = avg_age_corr[:, AGE_DIM:] if avg_age_corr is not None else None
    test_age_corr_values = test_age_corr[:, AGE_DIM:] if test_age_corr is not None else None

    train_age_corr_abs_mean = (
        float(np.mean(np.abs(train_age_corr_values))) if train_age_corr_values is not None else float("nan")
    )
    test_age_corr_abs_mean = (
        float(np.mean(np.abs(test_age_corr_values))) if test_age_corr_values is not None else float("nan")
    )

    if (epoch + 1) % PLOT_EVERY_N_EPOCHS == 0:
        plot_age_corr_heatmap(
            avg_age_corr,
            save_path=os.path.join(train_plot_dir, f"epoch_{epoch + 1:03d}_heatmap.png"),
            title=f"Train Age-Neuron Correlation Heatmap (Epoch {epoch + 1})",
            age_dim=AGE_DIM,
        )
        plot_age_corr_heatmap(
            test_age_corr,
            save_path=os.path.join(eval_plot_dir, f"epoch_{epoch + 1:03d}_heatmap.png"),
            title=f"Test Age-Neuron Correlation Heatmap (Epoch {epoch + 1})",
            age_dim=AGE_DIM,
        )

    current_lr = optimizer.param_groups[0]["lr"]
    print("\n" + "=" * 130)
    print(
        f"Epoch [{epoch + 1:3d}/{EPOCHS}] | LR: {current_lr:.6f} | "
        f"lambda_decorr: {current_lambda_decorr:.6f} | arcface_m: {current_margin:.6f}"
    )
    print(
        f"Train Loss => Total: {train_total_loss:.4f}, SPK: {train_spk_loss:.4f}, "
        f"AGE: {train_age_loss:.4f}, DECORR: {train_decorr_loss:.4f}"
    )
    print(f"Acc   => Train SPK/Age: {train_spk_acc:.2f}%/{train_age_acc:.2f}%")
    print(
        f"Val   => Loss(T/SPK/AGE/DEC): {val_metrics['val_total_loss']:.4f}/"
        f"{val_metrics['val_spk_loss']:.4f}/{val_metrics['val_age_loss']:.4f}/{val_metrics['val_decorr_loss']:.4f} | "
        f"Acc SPK/Age: {val_metrics['val_spk_acc']:.2f}%/{val_metrics['val_age_acc']:.2f}% | "
        f"Age AUC: {val_metrics['val_age_auc']:.6f}"
    )
    print(
        f"Test  => EER(before disentangle): {test_eer_before:.4f} | "
        f"EER(after disentangle): {test_eer_after:.4f} | Age AUC: {test_age_auc:.6f} | "
        f"Best(after): {best_test_EER:.4f}"
    )
    print(
        f"Corr  => Train |mean|: {train_age_corr_abs_mean:.6f} | "
        f"Val |mean|: {val_metrics['val_age_corr_abs_mean']:.6f} | "
        f"Test |mean|: {test_age_corr_abs_mean:.6f}"
    )
    print_age_neuron_correlation_stats(avg_age_corr, tag="Train", age_dim=AGE_DIM)
    print_age_neuron_correlation_stats(val_metrics["val_age_corr"], tag="Val", age_dim=AGE_DIM)
    print_age_neuron_correlation_stats(test_age_corr, tag="Test", age_dim=AGE_DIM)
    print("=" * 130 + "\n")

    if best_test_EER > test_eer_after:
        best_test_EER = test_eer_after
        best_epoch = epoch + 1
        no_improve_epochs = 0
    else:
        no_improve_epochs += 1

    checkpoint_manager.save_epoch_model(epoch + 1, model)

    writer.add_scalar("Train/Total_Loss", train_total_loss, epoch)
    writer.add_scalar("Train/SPK_Loss", train_spk_loss, epoch)
    writer.add_scalar("Train/AGE_Loss", train_age_loss, epoch)
    writer.add_scalar("Train/DECORR_Loss", train_decorr_loss, epoch)
    writer.add_scalar("Train/Acc_SPK", train_spk_acc, epoch)
    writer.add_scalar("Train/Acc_Age", train_age_acc, epoch)
    writer.add_scalar("Val/Total_Loss", val_metrics["val_total_loss"], epoch)
    writer.add_scalar("Val/SPK_Loss", val_metrics["val_spk_loss"], epoch)
    writer.add_scalar("Val/AGE_Loss", val_metrics["val_age_loss"], epoch)
    writer.add_scalar("Val/DECORR_Loss", val_metrics["val_decorr_loss"], epoch)
    writer.add_scalar("Val/Acc_SPK", val_metrics["val_spk_acc"], epoch)
    writer.add_scalar("Val/Acc_Age", val_metrics["val_age_acc"], epoch)
    writer.add_scalar("Val/Age_AUC", val_metrics["val_age_auc"], epoch)
    writer.add_scalar("Test/EER_Before", test_eer_before, epoch)
    writer.add_scalar("Test/EER_After", test_eer_after, epoch)
    writer.add_scalar("Test/Age_AUC", test_age_auc, epoch)
    writer.add_scalar("Test/Best_EER_After", best_test_EER, epoch)
    writer.add_scalar("Train/Lambda_Decorr_Current", current_lambda_decorr, epoch)
    writer.add_scalar("Train/ArcFace_Margin_Current", current_margin, epoch)
    writer.add_scalar("Train/Global_Step", global_step, epoch)
    writer.add_scalar("LR", current_lr, epoch)

    csv_writer.writerow(
        [
            epoch + 1,
            global_step,
            current_lr,
            current_margin,
            current_lambda_decorr,
            train_total_loss,
            train_spk_loss,
            train_age_loss,
            train_decorr_loss,
            train_spk_acc,
            train_age_acc,
            train_age_corr_abs_mean,
            val_metrics["val_total_loss"],
            val_metrics["val_spk_loss"],
            val_metrics["val_age_loss"],
            val_metrics["val_decorr_loss"],
            val_metrics["val_spk_acc"],
            val_metrics["val_age_acc"],
            val_metrics["val_age_auc"],
            val_metrics["val_age_corr_abs_mean"],
            test_eer_before,
            test_eer_after,
            test_age_auc,
            test_age_corr_abs_mean,
            best_test_EER,
        ]
    )
    csv_file.flush()

    if EARLY_STOPPING_ENABLED and no_improve_epochs >= EARLY_STOPPING_PATIENCE:
        print(
            f"Early stopping 觸發：連續 {EARLY_STOPPING_PATIENCE} 個 epoch 無改善。"
            f"最佳 Test EER(after)={best_test_EER:.4f}（Epoch {best_epoch}）。"
        )
        break

writer.close()
csv_file.close()

checkpoint_info = checkpoint_manager.finalize(
    model=model,
    projector_state_dict=model.projector.state_dict(),
)

if checkpoint_info["last_epoch_model_path"] is not None:
    print(f"✓ 最後 epoch（{checkpoint_info['last_epoch']}）的模型已儲存: {checkpoint_info['last_epoch_model_path']}")
    if checkpoint_info["last_epoch_projector_path"] is not None:
        print(f"✓ 最後 epoch（{checkpoint_info['last_epoch']}）的 projector 已儲存: {checkpoint_info['last_epoch_projector_path']}")

print("Training finished (Linear-Decorr MLP).")
