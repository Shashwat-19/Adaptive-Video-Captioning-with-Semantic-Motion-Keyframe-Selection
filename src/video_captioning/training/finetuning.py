"""T4-conscious optional LoRA fine-tuning for the implemented BLIP-2 workflow."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import pandas as pd

from video_captioning.keyframes.pipeline import select_frames_from_video
from video_captioning.models.captioner import BLIP2VideoCaptioner, make_contact_sheet
from video_captioning.utils.config import ExperimentConfig


def run_finetuning(
    config: ExperimentConfig,
    captions: pd.DataFrame,
    video_paths: dict[str, Path],
    train_ids: set[str],
    validation_ids: set[str],
) -> list[dict]:
    """Fine-tune BLIP-2/LoRA using one randomly sampled caption per video each epoch.

    This is deliberately opt-in: it downloads the configured model and can be slow on
    a T4. It follows the notebook's frame-selection, mixed-precision, gradient-
    accumulation, validation-loss, and early-stopping logic.
    """
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm
    from transformers import get_linear_schedule_with_warmup

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = Path(config.output_dir)
    checkpoint_dir = output_dir / "checkpoints" / config.keyframe_method
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    captioner = BLIP2VideoCaptioner(
        config.model_name, config.image_size, config.max_caption_length, config.use_4bit, device
    ).load_model()
    assert captioner.model is not None and captioner.processor is not None
    model = captioner.model
    if config.use_lora:
        for parameter in model.parameters():
            parameter.requires_grad = False
        model = get_peft_model(
            model,
            LoraConfig(
                r=8,
                lora_alpha=16,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
                bias="none",
            ),
        )

    references = captions.groupby("video_id")["caption"].apply(list).to_dict()
    selected_train = sorted(set(train_ids) & set(video_paths))
    selected_validation = sorted(set(validation_ids) & set(video_paths))
    if config.max_train_videos:
        selected_train = selected_train[: config.max_train_videos]
        selected_validation = selected_validation[: max(1, config.max_train_videos // 8)]
    if not selected_train or not selected_validation:
        raise ValueError("Training and validation each need at least one available video.")

    class VideoCaptionDataset(Dataset):
        def __init__(self, ids: list[str]) -> None:
            self.ids = ids
            self.semantic_runtime: tuple | None = None

        def __len__(self) -> int:
            return len(self.ids)

        def __getitem__(self, index: int):
            video_id = self.ids[index]
            selection, self.semantic_runtime = select_frames_from_video(
                video_paths[video_id], config, self.semantic_runtime
            )
            return video_id, make_contact_sheet(selection.frames, config.image_size), random.choice(references[video_id])

    def collate(batch):
        video_ids, images, texts = zip(*batch)
        inputs = captioner.processor(images=list(images), text=list(texts), padding=True, return_tensors="pt")
        labels = inputs.input_ids.clone()
        labels[labels == captioner.processor.tokenizer.pad_token_id] = -100
        inputs["labels"] = labels
        return video_ids, {name: value.to(device) for name, value in inputs.items()}

    train_loader = DataLoader(
        VideoCaptionDataset(selected_train), batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, collate_fn=collate
    )
    validation_loader = DataLoader(
        VideoCaptionDataset(selected_validation), batch_size=config.batch_size,
        num_workers=config.num_workers, collate_fn=collate
    )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate)
    steps = math.ceil(len(train_loader) / config.gradient_accumulation_steps) * config.num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, max(1, int(0.1 * steps)), max(1, steps))
    scaler = torch.cuda.amp.GradScaler(enabled=device == "cuda")
    history: list[dict] = []
    best_validation_loss, patience = float("inf"), 0

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        total_loss, started = 0.0, time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for step, (_, batch) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)):
            with torch.cuda.amp.autocast(enabled=device == "cuda"):
                loss = model(**batch).loss / config.gradient_accumulation_steps
            scaler.scale(loss).backward()
            if (step + 1) % config.gradient_accumulation_steps == 0 or step + 1 == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            total_loss += loss.item() * config.gradient_accumulation_steps

        model.eval()
        validation_losses = []
        with torch.no_grad():
            for _, batch in validation_loader:
                with torch.cuda.amp.autocast(enabled=device == "cuda"):
                    validation_losses.append(model(**batch).loss.item())
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, len(train_loader)),
            "val_loss": sum(validation_losses) / max(1, len(validation_losses)),
            "lr": scheduler.get_last_lr()[0],
            "epoch_time_s": time.perf_counter() - started,
        }
        history.append(record)
        if record["val_loss"] < best_validation_loss:
            best_validation_loss, patience = record["val_loss"], 0
            model.save_pretrained(checkpoint_dir / f"epoch_{epoch}")
            captioner.processor.save_pretrained(checkpoint_dir / f"epoch_{epoch}")
            (checkpoint_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        else:
            patience += 1
        if patience >= 2:
            break

    logs = output_dir / "results" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(logs / f"{config.keyframe_method}_training.csv", index=False)
    return history
