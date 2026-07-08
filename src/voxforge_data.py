from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torchaudio.functional as AF
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATASET_NAME = "ciempiess/voxforge_spanish"
DEFAULT_CONFIG_NAME = "voxforge_spanish"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_ALLOWED_CHARS = " abcdefghijklmnopqrstuvwxyz\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1'"


def normalize_transcript(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1'\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _duration_to_key(max_duration: float) -> str:
    return str(max_duration).replace(".", "p")


def _ctc_time_steps(num_samples: int, n_fft: int = 400, hop_length: int = 160) -> int:
    frames = 1 if num_samples < n_fft else 1 + (num_samples - n_fft) // hop_length
    # Two Conv2d layers use kernel=3, padding=1, stride=2 along time.
    frames = (frames + 1) // 2
    frames = (frames + 1) // 2
    return max(1, frames)


@dataclass
class CharVocabulary:
    tokens: list[str]
    blank_token: str = "<blank>"

    @property
    def blank_id(self) -> int:
        return 0

    @property
    def token_to_id(self) -> dict[str, int]:
        return {token: idx for idx, token in enumerate(self.tokens)}

    def __len__(self) -> int:
        return len(self.tokens)

    def encode(self, text: str) -> list[int]:
        mapping = self.token_to_id
        normalized = normalize_transcript(text)
        return [mapping[ch] for ch in normalized if ch in mapping and ch != self.blank_token]

    def decode(self, ids: list[int] | torch.Tensor, collapse_ctc: bool = True) -> str:
        if torch.is_tensor(ids):
            ids = ids.detach().cpu().tolist()
        pieces: list[str] = []
        prev = None
        for idx in ids:
            idx = int(idx)
            if idx == self.blank_id:
                prev = idx
                continue
            if collapse_ctc and idx == prev:
                prev = idx
                continue
            if 0 <= idx < len(self.tokens):
                pieces.append(self.tokens[idx])
            prev = idx
        return re.sub(r"\s+", " ", "".join(pieces)).strip()

    def decode_batch(self, pred_ids: torch.Tensor) -> list[str]:
        return [self.decode(row, collapse_ctc=True) for row in pred_ids]

    def to_dict(self) -> dict[str, Any]:
        return {"tokens": self.tokens, "blank_token": self.blank_token}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CharVocabulary":
        return cls(tokens=list(data["tokens"]), blank_token=data.get("blank_token", "<blank>"))

    @classmethod
    def build(cls, texts: list[str], allowed_chars: str = DEFAULT_ALLOWED_CHARS) -> "CharVocabulary":
        chars = set(allowed_chars)
        for text in texts:
            chars.update(ch for ch in normalize_transcript(text) if ch != " ")
        ordered = ["<blank>", " "] + sorted(ch for ch in chars if ch not in {" ", "<blank>"})
        return cls(tokens=ordered)


class VoxForgeCachedDataset(Dataset):
    def __init__(self, samples: list[dict[str, Any]], vocab: CharVocabulary) -> None:
        self.samples = samples
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.samples[index]
        waveform = item["waveform"].float()
        text = item["text"]
        target = torch.tensor(self.vocab.encode(text), dtype=torch.long)
        return {
            "audio_id": item.get("audio_id", str(index)),
            "waveform": waveform,
            "waveform_length": torch.tensor(waveform.numel(), dtype=torch.long),
            "target": target,
            "target_length": torch.tensor(target.numel(), dtype=torch.long),
            "text": text,
            "duration": float(item.get("duration", waveform.numel() / DEFAULT_SAMPLE_RATE)),
        }


def voxforge_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    waveforms = [item["waveform"] for item in batch]
    targets = [item["target"] for item in batch]
    padded_waveforms = pad_sequence(waveforms, batch_first=True)
    target_lengths = torch.tensor([target.numel() for target in targets], dtype=torch.long)
    concatenated_targets = torch.cat(targets) if targets else torch.empty(0, dtype=torch.long)
    return {
        "audio_ids": [item["audio_id"] for item in batch],
        "waveforms": padded_waveforms,
        "waveform_lengths": torch.tensor([item["waveform_length"].item() for item in batch], dtype=torch.long),
        "targets": concatenated_targets,
        "target_lengths": target_lengths,
        "texts": [item["text"] for item in batch],
        "durations": torch.tensor([item["duration"] for item in batch], dtype=torch.float32),
    }


def _cache_path(
    cache_dir: Path,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    max_duration: float,
    max_text_length: int,
) -> Path:
    key = _duration_to_key(max_duration)
    return cache_dir / (
        f"voxforge_spanish_seed{seed}_tr{train_size}_va{val_size}_te{test_size}_"
        f"max{key}_txt{max_text_length}.pt"
    )


def _load_streaming_samples(
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    max_duration: float,
    max_text_length: int,
    dataset_name: str,
    config_name: str,
    sample_rate: int,
    shuffle_buffer: int,
) -> dict[str, Any]:
    from datasets import load_dataset

    total_needed = train_size + val_size + test_size
    dataset = load_dataset(dataset_name, config_name, split="train", streaming=True)
    dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)

    selected: list[dict[str, Any]] = []
    skipped = {"empty_text": 0, "duration": 0, "text_length": 0, "ctc_length": 0, "audio": 0}
    for raw in dataset:
        if len(selected) >= total_needed:
            break
        text = normalize_transcript(str(raw.get("normalized_text", "")))
        if not text:
            skipped["empty_text"] += 1
            continue
        if len(text) > max_text_length:
            skipped["text_length"] += 1
            continue
        duration = float(raw.get("duration") or 0.0)
        if duration <= 0.0:
            duration = len(raw["audio"]["array"]) / float(raw["audio"]["sampling_rate"])
        if duration > max_duration:
            skipped["duration"] += 1
            continue
        try:
            audio = raw["audio"]
            waveform = torch.as_tensor(audio["array"], dtype=torch.float32)
            sr = int(audio["sampling_rate"])
        except Exception:
            skipped["audio"] += 1
            continue
        if waveform.ndim > 1:
            waveform = waveform.mean(dim=-1)
        if sr != sample_rate:
            waveform = AF.resample(waveform, orig_freq=sr, new_freq=sample_rate)
        max_abs = waveform.abs().max().clamp_min(1e-6)
        waveform = (waveform / max_abs).contiguous()
        if len(text) >= _ctc_time_steps(waveform.numel()):
            skipped["ctc_length"] += 1
            continue
        selected.append(
            {
                "audio_id": str(raw.get("audio_id", len(selected))),
                "waveform": waveform,
                "text": text,
                "duration": waveform.numel() / float(sample_rate),
            }
        )

    if len(selected) < total_needed:
        raise RuntimeError(f"Only collected {len(selected)} VoxForge samples, expected {total_needed}. Skipped={skipped}")

    train = selected[:train_size]
    val = selected[train_size : train_size + val_size]
    test = selected[train_size + val_size :]
    vocab = CharVocabulary.build([item["text"] for item in train])
    return {
        "splits": {"train": train, "val": val, "test": test},
        "vocab": vocab.to_dict(),
        "metadata": {
            "dataset_name": dataset_name,
            "config_name": config_name,
            "sample_rate": sample_rate,
            "seed": seed,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "max_duration": max_duration,
            "max_text_length": max_text_length,
            "skipped": skipped,
        },
    }


def prepare_voxforge_cache(
    data_root: str | Path,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    max_duration: float,
    max_text_length: int = 160,
    dataset_name: str = DEFAULT_DATASET_NAME,
    config_name: str = DEFAULT_CONFIG_NAME,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    shuffle_buffer: int = 5000,
    refresh_cache: bool = False,
) -> Path:
    cache_dir = Path(data_root) / "voxforge_task2"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, train_size, val_size, test_size, seed, max_duration, max_text_length)
    if path.exists() and not refresh_cache:
        return path
    payload = _load_streaming_samples(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
        max_duration=max_duration,
        max_text_length=max_text_length,
        dataset_name=dataset_name,
        config_name=config_name,
        sample_rate=sample_rate,
        shuffle_buffer=shuffle_buffer,
    )
    torch.save(payload, path)
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(json.dumps(payload["metadata"], indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def build_voxforge_loaders(
    data_root: str | Path,
    train_size: int,
    val_size: int,
    test_size: int,
    seed: int,
    max_duration: float,
    max_text_length: int,
    batch_size: int,
    workers: int,
    dataset_name: str = DEFAULT_DATASET_NAME,
    config_name: str = DEFAULT_CONFIG_NAME,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    shuffle_buffer: int = 5000,
    refresh_cache: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader, CharVocabulary, dict[str, Any]]:
    cache = prepare_voxforge_cache(
        data_root=data_root,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seed=seed,
        max_duration=max_duration,
        max_text_length=max_text_length,
        dataset_name=dataset_name,
        config_name=config_name,
        sample_rate=sample_rate,
        shuffle_buffer=shuffle_buffer,
        refresh_cache=refresh_cache,
    )
    payload = torch.load(cache, map_location="cpu", weights_only=False)
    vocab = CharVocabulary.from_dict(payload["vocab"])
    train_ds = VoxForgeCachedDataset(payload["splits"]["train"], vocab)
    val_ds = VoxForgeCachedDataset(payload["splits"]["val"], vocab)
    test_ds = VoxForgeCachedDataset(payload["splits"]["test"], vocab)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=voxforge_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=voxforge_collate,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=voxforge_collate,
        pin_memory=True,
    )
    metadata = dict(payload.get("metadata", {}))
    metadata["cache_path"] = str(cache)
    metadata["vocab_size"] = len(vocab)
    return train_loader, val_loader, test_loader, vocab, metadata
