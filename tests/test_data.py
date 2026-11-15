import numpy as np

from ecg_arrhythmia.data import (
    MITBIH_DS1,
    MITBIH_DS2,
    extract_beats,
    map_beat_symbol,
    mitbih_patient_id,
    read_incart_patient_mapping,
    split_patient_ids,
)


def test_aami_style_beat_symbol_mapping():
    assert map_beat_symbol("N") == 0
    assert map_beat_symbol("L") == 0
    assert map_beat_symbol("A") == 1
    assert map_beat_symbol("V") == 2
    assert map_beat_symbol("F") is None


def test_extract_beats_filters_normalizes_and_skips_unknown_symbols():
    sampling_rate = 360
    time = np.arange(sampling_rate * 4) / sampling_rate
    signal = np.sin(2 * np.pi * 2 * time) + 0.2 * np.sin(2 * np.pi * 60 * time)
    samples = [sampling_rate, sampling_rate * 2, 2]
    symbols = ["N", "V", "N"]

    beats, labels = extract_beats(signal, samples, symbols, sampling_rate)

    assert beats.shape == (2, 252)
    assert labels.tolist() == [0, 2]
    np.testing.assert_allclose(beats.mean(axis=1), 0.0, atol=1e-5)
    np.testing.assert_allclose(beats.std(axis=1), 1.0, atol=1e-5)


def test_patient_split_is_disjoint_and_reproducible():
    patients = np.repeat(["p1", "p2", "p3", "p4", "p5"], 3)
    train_a, test_a = split_patient_ids(patients, 0.4, 17)
    train_b, test_b = split_patient_ids(patients, 0.4, 17)

    assert set(train_a).isdisjoint(set(test_a))
    assert set(train_a) == set(train_b)
    assert set(test_a) == set(test_b)


def test_mitbih_records_from_same_subject_share_a_patient_group():
    assert mitbih_patient_id("201") == mitbih_patient_id("202")
    assert mitbih_patient_id("203") == "203"
    assert set(map(mitbih_patient_id, MITBIH_DS1)).isdisjoint(
        map(mitbih_patient_id, MITBIH_DS2)
    )


def test_incart_mapping_groups_multiple_records_by_patient(tmp_path):
    mapping_path = tmp_path / "files-patients-diagnoses.txt"
    mapping_path.write_text(
        "patient 1\nI01 I02\ncoronary artery disease\n"
        "patient 2\nI03\n\n",
        encoding="utf-8",
    )

    mapping = read_incart_patient_mapping(mapping_path)

    assert mapping == {"I01": "1", "I02": "1", "I03": "2"}
