"""Training, evaluation, and optional supervised cross-dataset fine-tuning."""

from __future__ import annotations

import json
import random
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import wfdb

from ecg_arrhythmia.data import (
    BeatSet,
    INCART_MAPPING_FILE,
    INCART_MAPPING_URL,
    MITBIH_DS1,
    MITBIH_DS2,
    load_incart_patient_mapping,
    load_records,
    mitbih_patient_id,
    split_patient_ids,
)
from ecg_arrhythmia.metrics import (
    ClassificationMetrics,
    classification_metrics,
    save_confusion_matrices,
)
from ecg_arrhythmia.models import MODEL_FACTORIES, build_model


def download_database(dataset: str, data_dir: Path) -> None:
    """Download one PhysioNet database and INCART's patient mapping."""
    database = dataset.lower()
    if database not in ("mitdb", "incartdb"):
        raise ValueError("dataset must be 'mitdb' or 'incartdb'.")
    target = data_dir / database
    target.mkdir(parents=True, exist_ok=True)
    wfdb.dl_database(db_dir=database, dl_dir=str(target))
    if database == "incartdb":
        mapping_path = target / INCART_MAPPING_FILE
        if not mapping_path.exists():
            urllib.request.urlretrieve(INCART_MAPPING_URL, mapping_path)


def _choose_device(requested: str) -> torch.device:
    if requested != "auto":
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("The requested MPS device is not available.")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("The requested CUDA device is not available.")
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(
    data: BeatSet, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(data.waveforms.astype(np.float32, copy=False)),
        torch.from_numpy(data.labels.astype(np.int64, copy=False)),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _predict(
    model: nn.Module, data: BeatSet, device: torch.device, batch_size: int
) -> np.ndarray:
    loader = _make_loader(data, batch_size, shuffle=False)
    predictions: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for waveforms, _ in loader:
            logits = model(waveforms.to(device))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(predictions)


def _fit_model(
    model: nn.Module,
    train_data: BeatSet,
    validation_data: BeatSet,
    device: torch.device,
    batch_size: int,
    epochs: int,
    patience: int,
    learning_rate: float,
    seed: int,
) -> Tuple[nn.Module, Dict[str, object]]:
    """Fit with class-weighted loss and early stopping on validation macro-F1."""
    _set_seed(seed)
    class_counts = np.bincount(train_data.labels, minlength=3).astype(np.float32)
    if np.any(class_counts == 0):
        absent = np.flatnonzero(class_counts == 0).tolist()
        raise ValueError(f"Training split is missing class indices: {absent}.")
    class_weights = len(train_data.labels) / (3.0 * class_counts)
    loss_function = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    train_loader = _make_loader(train_data, batch_size, shuffle=True)

    best_f1 = -1.0
    best_state: Optional[Dict[str, torch.Tensor]] = None
    best_epoch = 0
    stale_epochs = 0
    history: List[Dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for waveforms, labels in train_loader:
            waveforms = waveforms.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(model(waveforms), labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            total_examples += len(labels)

        validation_predictions = _predict(model, validation_data, device, batch_size)
        validation_metrics = classification_metrics(
            validation_data.labels, validation_predictions
        )
        validation_f1 = float(validation_metrics["macro_f1"])
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / total_examples,
                "validation_macro_f1": validation_f1,
            }
        )
        print(
            f"  epoch {epoch:02d}/{epochs}: "
            f"train_loss={history[-1]['train_loss']:.4f}, "
            f"validation_macro_f1={validation_f1:.4f}"
        )
        if validation_f1 > best_f1:
            best_f1 = validation_f1
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    if best_state is None:
        raise RuntimeError("Training completed without a valid model checkpoint.")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_f1,
        "history": history,
    }


def _save_checkpoint(
    path: Path, model_name: str, model: nn.Module, metadata: Dict[str, object]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "num_classes": 3,
            "state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "metadata": metadata,
        },
        path,
    )


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _evaluate(
    model: nn.Module, data: BeatSet, device: torch.device, batch_size: int
) -> ClassificationMetrics:
    predictions = _predict(model, data, device, batch_size)
    return classification_metrics(data.labels, predictions)


def run_experiments(
    data_dir: Path,
    output_dir: Path,
    model_names: Sequence[str],
    epochs: int = 30,
    patience: int = 6,
    batch_size: int = 256,
    seed: int = 42,
    device_name: str = "auto",
    adapt_incart: bool = False,
) -> None:
    """Train requested models on MIT-BIH and evaluate on both datasets."""
    unknown = sorted(set(model_names) - set(MODEL_FACTORIES))
    if unknown:
        raise ValueError(f"Unknown model(s): {', '.join(unknown)}.")
    if not model_names:
        raise ValueError("At least one model must be selected.")
    if epochs < 1 or patience < 1 or batch_size < 1:
        raise ValueError("epochs, patience, and batch_size must be positive.")
    mit_dir = data_dir / "mitdb"
    incart_dir = data_dir / "incartdb"

    mit_train_all = load_records(
        mit_dir,
        MITBIH_DS1,
        patient_mapping={
            record_id: mitbih_patient_id(record_id) for record_id in MITBIH_DS1
        },
    )
    mit_test = load_records(
        mit_dir,
        MITBIH_DS2,
        patient_mapping={
            record_id: mitbih_patient_id(record_id) for record_id in MITBIH_DS2
        },
    )
    incart_patients = load_incart_patient_mapping(incart_dir)
    incart_all = load_records(
        incart_dir, sorted(incart_patients), patient_mapping=incart_patients
    )

    train_patients, validation_patients = split_patient_ids(
        mit_train_all.patient_ids, test_size=0.2, random_seed=seed
    )
    mit_train = mit_train_all.select_patients(train_patients)
    mit_validation = mit_train_all.select_patients(validation_patients)

    incart_test_patients = np.asarray([], dtype=str)
    incart_adapt_train_patients = np.asarray([], dtype=str)
    incart_adapt_validation_patients = np.asarray([], dtype=str)
    incart_test: Optional[BeatSet] = None
    incart_adapt_train: Optional[BeatSet] = None
    incart_adapt_validation: Optional[BeatSet] = None
    if adapt_incart:
        incart_pool_patients, incart_test_patients = split_patient_ids(
            incart_all.patient_ids, test_size=0.2, random_seed=seed
        )
        incart_pool = incart_all.select_patients(incart_pool_patients)
        incart_adapt_train_patients, incart_adapt_validation_patients = split_patient_ids(
            incart_pool.patient_ids, test_size=0.2, random_seed=seed + 1
        )
        incart_test = incart_all.select_patients(incart_test_patients)
        incart_adapt_train = incart_pool.select_patients(incart_adapt_train_patients)
        incart_adapt_validation = incart_pool.select_patients(
            incart_adapt_validation_patients
        )

    device = _choose_device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    for model_name in model_names:
        print(f"Training {model_name} on MIT-BIH DS1 ({device})")
        _set_seed(seed)
        model_dir = output_dir / model_name
        model = build_model(model_name)
        model, fit_summary = _fit_model(
            model,
            mit_train,
            mit_validation,
            device,
            batch_size,
            epochs,
            patience,
            learning_rate=1e-3,
            seed=seed,
        )
        source_checkpoint = model_dir / "checkpoint.pt"
        training_metadata = {
            "seed": seed,
            "target_sampling_rate_hz": 360,
            "train_patient_ids": sorted(set(train_patients.tolist())),
            "validation_patient_ids": sorted(set(validation_patients.tolist())),
            "mit_test_patient_ids": sorted(set(mit_test.patient_ids.tolist())),
            "fit": fit_summary,
        }
        _save_checkpoint(source_checkpoint, model_name, model, training_metadata)

        mit_test_metrics = _evaluate(model, mit_test, device, batch_size)
        incart_external_metrics = _evaluate(model, incart_all, device, batch_size)
        metric_names = ("accuracy", "macro_precision", "macro_recall", "macro_f1")
        performance_drop = {
            metric_name: float(
                mit_test_metrics[metric_name] - incart_external_metrics[metric_name]
            )
            for metric_name in metric_names
        }
        split_report: Dict[str, List[str]] = {
            "mit_train_patients": sorted(set(train_patients.tolist())),
            "mit_validation_patients": sorted(set(validation_patients.tolist())),
            "mit_internal_test_patients": sorted(set(mit_test.patient_ids.tolist())),
        }
        metrics: Dict[str, object] = {
            "model": model_name,
            "device": str(device),
            "seed": seed,
            "fit": fit_summary,
            "splits": split_report,
            "mit_internal_test": mit_test_metrics,
            "incart_external_all_patients": incart_external_metrics,
            "performance_drop_mit_ds2_minus_incart": performance_drop,
        }
        matrices: Dict[str, ClassificationMetrics] = {
            "MIT-BIH DS2": mit_test_metrics,
            "INCART (all)": incart_external_metrics,
        }

        if adapt_incart:
            if incart_test is None or incart_adapt_train is None or incart_adapt_validation is None:
                raise RuntimeError("INCART adaptation split was not initialized.")
            split_report["incart_adapt_train_patients"] = sorted(
                set(incart_adapt_train_patients.tolist())
            )
            split_report["incart_adapt_validation_patients"] = sorted(
                set(incart_adapt_validation_patients.tolist())
            )
            split_report["incart_held_out_test_patients"] = sorted(
                set(incart_test_patients.tolist())
            )
            held_out_baseline = _evaluate(model, incart_test, device, batch_size)
            metrics["incart_held_out_baseline"] = held_out_baseline
            adapted_model = build_model(model_name)
            adapted_model.load_state_dict(model.state_dict())
            adapted_model, adaptation_fit = _fit_model(
                adapted_model,
                incart_adapt_train,
                incart_adapt_validation,
                device,
                batch_size,
                epochs,
                patience,
                learning_rate=1e-4,
                seed=seed + 10,
            )
            metrics["incart_adaptation_fit"] = adaptation_fit
            held_out_adapted = _evaluate(
                adapted_model, incart_test, device, batch_size
            )
            metrics["incart_held_out_adapted"] = held_out_adapted
            _save_checkpoint(
                model_dir / "incart_adapted_checkpoint.pt",
                model_name,
                adapted_model,
                {
                    "seed": seed + 10,
                    "adaptation_patient_ids": sorted(
                        set(incart_adapt_train_patients.tolist())
                    ),
                    "adaptation_validation_patient_ids": sorted(
                        set(incart_adapt_validation_patients.tolist())
                    ),
                },
            )
            matrices["INCART held-out (baseline)"] = held_out_baseline
            matrices["INCART held-out (adapted)"] = held_out_adapted

        _write_json(model_dir / "metrics.json", metrics)
        save_confusion_matrices(matrices, model_dir / "confusion_matrices.png")
        print(f"{model_name}: wrote results to {model_dir}")
