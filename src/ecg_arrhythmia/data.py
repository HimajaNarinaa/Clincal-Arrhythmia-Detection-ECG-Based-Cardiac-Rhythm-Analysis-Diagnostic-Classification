"""WFDB loading, signal preprocessing, beat extraction, and patient splits."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import wfdb
from scipy.signal import butter, resample_poly, sosfiltfilt
from sklearn.model_selection import GroupShuffleSplit

CLASS_NAMES = ("N (normal)", "S (supraventricular)", "V (ventricular)")
SYMBOL_TO_CLASS = {
    "N": 0,
    "L": 0,
    "R": 0,
    "e": 0,
    "j": 0,
    "A": 1,
    "a": 1,
    "J": 1,
    "S": 1,
    "V": 2,
    "E": 2,
}

MITBIH_DS1 = (
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119",
    "122", "124", "201", "202", "203", "205", "207", "208", "209", "215",
    "220", "223", "230",
)
MITBIH_DS2 = (
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "210",
    "212", "213", "214", "219", "221", "222", "228", "231", "232", "233",
    "234",
)
INCART_MAPPING_FILE = "files-patients-diagnoses.txt"
INCART_MAPPING_URL = (
    "https://physionet.org/files/incartdb/1.0.0/"
    "files-patients-diagnoses.txt"
)


@dataclass(frozen=True)
class BeatSet:
    """Beat waveforms and their labels, record IDs, and patient groups."""

    waveforms: np.ndarray
    labels: np.ndarray
    record_ids: np.ndarray
    patient_ids: np.ndarray

    def select_patients(self, patient_ids: Iterable[str]) -> "BeatSet":
        selected = np.isin(self.patient_ids, list(patient_ids))
        if not np.any(selected):
            raise ValueError("The requested patient selection contains no beats.")
        return BeatSet(
            self.waveforms[selected],
            self.labels[selected],
            self.record_ids[selected],
            self.patient_ids[selected],
        )


def map_beat_symbol(symbol: str) -> Optional[int]:
    """Return the three-class label for an included beat symbol, if any."""
    return SYMBOL_TO_CLASS.get(symbol)


def mitbih_patient_id(record_id: str) -> str:
    """Keep the two recordings from one MIT-BIH subject in a single group."""
    return "201-202" if record_id in ("201", "202") else record_id


def preprocess_signal(
    signal: np.ndarray, source_fs: float, target_fs: int = 360
) -> np.ndarray:
    """Resample a one-dimensional ECG and apply a zero-phase band-pass filter."""
    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    if values.size < 32:
        raise ValueError("ECG signal is too short to filter safely.")
    if not np.all(np.isfinite(values)):
        raise ValueError("ECG signal contains non-finite samples.")
    if not np.isfinite(source_fs) or source_fs <= 0 or target_fs <= 80:
        raise ValueError("Sampling rates must be positive and target_fs must exceed 80 Hz.")

    source_rate = int(round(source_fs))
    if not np.isclose(source_fs, source_rate):
        raise ValueError("Non-integer source sampling rates are not supported.")
    if source_rate != target_fs:
        divisor = gcd(source_rate, target_fs)
        values = resample_poly(values, target_fs // divisor, source_rate // divisor)

    filter_sos = butter(
        4, (0.5, 40.0), btype="bandpass", fs=target_fs, output="sos"
    )
    return sosfiltfilt(filter_sos, values).astype(np.float32)


def extract_beats(
    signal: np.ndarray,
    sample_indices: Sequence[int],
    symbols: Sequence[str],
    source_fs: float,
    target_fs: int = 360,
    before_seconds: float = 0.30,
    after_seconds: float = 0.40,
) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess one ECG and extract normalized, labeled beat windows."""
    if len(sample_indices) != len(symbols):
        raise ValueError("Annotation sample indices and symbols must have equal length.")
    processed = preprocess_signal(signal, source_fs, target_fs)
    before = int(round(before_seconds * target_fs))
    after = int(round(after_seconds * target_fs))
    if before < 1 or after < 1:
        raise ValueError("Beat windows must include samples on both sides.")

    windows: List[np.ndarray] = []
    labels: List[int] = []
    for sample_index, symbol in zip(sample_indices, symbols):
        label = map_beat_symbol(symbol)
        if label is None:
            continue
        center = int(round(int(sample_index) * target_fs / source_fs))
        start, stop = center - before, center + after
        if start < 0 or stop > processed.size:
            continue
        beat = processed[start:stop].astype(np.float32, copy=True)
        deviation = float(beat.std())
        if deviation > 1e-8:
            beat = (beat - float(beat.mean())) / deviation
        else:
            beat.fill(0.0)
        windows.append(beat)
        labels.append(label)

    window_length = before + after
    if not windows:
        return (
            np.empty((0, window_length), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )
    return np.stack(windows), np.asarray(labels, dtype=np.int64)


def read_incart_patient_mapping(mapping_path: Path) -> Dict[str, str]:
    """Parse PhysioNet's official INCART record-to-patient mapping file."""
    if not mapping_path.is_file():
        raise FileNotFoundError(
            "INCART patient mapping is required for leakage-safe splits. "
            "Download the database with `ecg-arrhythmia download "
            "--dataset incartdb --data-dir data` or place "
            f"{INCART_MAPPING_FILE} in {mapping_path.parent}."
        )

    patient_id: Optional[str] = None
    mapping: Dict[str, str] = {}
    for line in mapping_path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if text.lower().startswith("patient "):
            patient_id = text.split(maxsplit=1)[1]
        elif patient_id is not None and text:
            records = [
                token for token in text.split() if re.fullmatch(r"I\d{2}", token)
            ]
            for record_id in records:
                mapping[record_id] = patient_id
    if not mapping:
        raise ValueError(f"No patient-to-record entries found in {mapping_path}.")
    return mapping


def discover_records(database_dir: Path) -> Dict[str, Path]:
    """Find local WFDB headers and return record-name to basename mappings."""
    headers = sorted(database_dir.rglob("*.hea"))
    records = {header.stem: header.with_suffix("") for header in headers}
    if not records:
        raise FileNotFoundError(
            f"No WFDB .hea files found under {database_dir}. "
            "Download the database first."
        )
    return records


def load_records(
    database_dir: Path,
    record_ids: Sequence[str],
    patient_mapping: Optional[Dict[str, str]] = None,
    target_fs: int = 360,
) -> BeatSet:
    """Load annotated records, selecting a preferred ECG lead when available."""
    paths = discover_records(database_dir)
    missing = [record_id for record_id in record_ids if record_id not in paths]
    if missing:
        raise FileNotFoundError(
            f"Missing records in {database_dir}: {', '.join(missing)}"
        )

    all_waveforms: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    all_record_ids: List[np.ndarray] = []
    all_patient_ids: List[np.ndarray] = []
    for record_id in record_ids:
        record_path = paths[record_id]
        if patient_mapping is not None and record_id not in patient_mapping:
            raise ValueError(f"No patient ID is available for record {record_id}.")
        record = wfdb.rdrecord(str(record_path))
        signal = np.asarray(record.p_signal)
        if signal.ndim == 1:
            signal = signal[:, np.newaxis]
        lead_names = [name.strip().upper() for name in (record.sig_name or [])]
        preferred = next(
            (lead_names.index(name) for name in ("MLII", "II") if name in lead_names),
            0,
        )
        annotations = wfdb.rdann(str(record_path), "atr")
        beats, labels = extract_beats(
            signal[:, preferred],
            annotations.sample,
            annotations.symbol,
            float(record.fs),
            target_fs=target_fs,
        )
        if not len(labels):
            continue
        patient_id = (
            patient_mapping[record_id]
            if patient_mapping is not None
            else record_id
        )
        all_waveforms.append(beats)
        all_labels.append(labels)
        all_record_ids.append(np.full(len(labels), record_id, dtype=str))
        all_patient_ids.append(np.full(len(labels), patient_id, dtype=str))

    if not all_waveforms:
        raise ValueError(
            f"No supported annotated beats were extracted from {database_dir}."
        )
    return BeatSet(
        np.concatenate(all_waveforms),
        np.concatenate(all_labels),
        np.concatenate(all_record_ids),
        np.concatenate(all_patient_ids),
    )


def split_patient_ids(
    patient_ids: Sequence[str], test_size: float, random_seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Split unique patient groups into disjoint train and held-out groups."""
    unique_patients = np.unique(np.asarray(patient_ids, dtype=str))
    if len(unique_patients) < 2:
        raise ValueError("At least two distinct patients are required for a split.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_seed
    )
    train_indices, test_indices = next(
        splitter.split(unique_patients, groups=unique_patients)
    )
    return unique_patients[train_indices], unique_patients[test_indices]


def load_incart_patient_mapping(database_dir: Path) -> Dict[str, str]:
    """Load and validate INCART's official patient mapping against local data."""
    mapping = read_incart_patient_mapping(database_dir / INCART_MAPPING_FILE)
    available = discover_records(database_dir)
    unassigned = sorted(set(available) - set(mapping))
    if unassigned:
        raise ValueError(
            "INCART patient mapping does not cover these local records: "
            + ", ".join(unassigned)
        )
    return mapping
