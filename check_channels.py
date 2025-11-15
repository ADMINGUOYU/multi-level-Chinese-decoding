#!/usr/bin/env python
"""
Diagnostic script to check available channel names for each subject.
This helps debug the channel name mismatch issue in run_multitask.py.
"""

import os
import sys
import pickle

def load_pickle(path):
    """Load pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)

def check_subject_channels(subj_id):
    """
    Check available channel names for a specific subject.

    Args:
        subj_id: str - Subject ID (e.g., "001", "002")
    """
    # Construct path to subject data
    base_path = "/mnt/afs/250010218/multi-level-Chinese-decoding/data/seeg.he2023xuanwu"
    subj_path = os.path.join(base_path, subj_id, "word_recitation")

    print(f"\n{'='*80}")
    print(f"Checking subject {subj_id}")
    print(f"{'='*80}")

    if not os.path.exists(subj_path):
        print(f"ERROR: Subject path does not exist: {subj_path}")
        return None

    # Loop through all runs
    runs = sorted([d for d in os.listdir(subj_path) if os.path.isdir(os.path.join(subj_path, d))])
    print(f"Found {len(runs)} runs: {runs}")

    all_channels = {}
    for run in runs:
        # Check both aligned and unaligned datasets
        for align_type in ["aligned", "unaligned"]:
            dataset_path = os.path.join(subj_path, run, f"dataset.bipolar.default.{align_type}")
            info_path = os.path.join(dataset_path, "info")

            if os.path.exists(info_path):
                dataset_info = load_pickle(info_path)
                ch_names = dataset_info.ch_names
                all_channels[f"{run}_{align_type}"] = ch_names
                print(f"\n{run} ({align_type}):")
                print(f"  Channels ({len(ch_names)}): {ch_names}")

    # Find common channels across all runs
    if all_channels:
        channel_sets = [set(chs) for chs in all_channels.values()]
        common_channels = set.intersection(*channel_sets) if channel_sets else set()
        print(f"\n{'='*80}")
        print(f"Common channels across all runs: {sorted(common_channels)}")
        print(f"Total common channels: {len(common_channels)}")
        print(f"{'='*80}")
        return sorted(common_channels)
    else:
        print("ERROR: No channel information found!")
        return None

def compare_with_config(subj_id, actual_channels):
    """
    Compare actual channels with the hardcoded configuration in run_multitask.py.

    Args:
        subj_id: str - Subject ID
        actual_channels: list - List of actual available channel names
    """
    # Hardcoded configurations from run_multitask.py
    config_channels = {
        "001": ["TI'9", "TI'7", "TI'8", "TI'10", "TI'11", "TI'13", "TI'12", "C'4", "C'3", "C'7"],
        "002": ["s1", "s3", "s2", "s4", "C3", "C4", "C6", "C5", "C7", "C2"],
        "003": ["ST5", "ST3", "ST4", "ST6", "ST7", "ST8", "C1", "C2", "TI7", "TI8"],
        "004": ["D6", "D7", "D9", "D8", "D10", "D11", "TI'4", "TI'5", "TI'6", "TI'7"],
        "005": ["E8", "E9", "E10", "E11", "E12", "E13", "TI3", "TI4", "TI5", "TI6"],
        "006": ["G9", "G8", "G10", "G11", "G12", "G13", "TI5", "TI6", "TI7", "TI8"],
        "007": ["H2", "H4", "H3", "H1", "H6", "H5", "E4", "H7", "C13", "E5"],
        "008": ["TI3", "TI4", "TI2", "TI5", "B9", "TI6", "TI7", "TI9", "TI10", "B5"],
        "009": ["K9", "K8", "K6", "K7", "K11", "K10", "K5", "K4", "K3", "I9"],
        "010": ["PI5", "PI6", "PI7", "PI8", "PI1", "PI9", "PI2", "SM2", "SP3", "PI4"],
        "011": ["T2", "T3", "C9", "T4", "T5", "C7", "C8", "T1", "s1", "C4"],
        "012": ["TI'4", "TI'2", "TI'3", "TI'5", "TI'8", "TI'6", "TI'7", "TO'9", "P'5", "TO'8"],
    }

    if subj_id not in config_channels:
        print(f"\nNo configuration found for subject {subj_id}")
        return

    configured_chs = config_channels[subj_id]
    actual_chs_set = set(actual_channels) if actual_channels else set()
    configured_chs_set = set(configured_chs)

    print(f"\n{'='*80}")
    print(f"Configuration comparison for subject {subj_id}")
    print(f"{'='*80}")
    print(f"\nConfigured channels ({len(configured_chs)}): {configured_chs}")
    print(f"\nActual channels ({len(actual_chs_set)}): {sorted(actual_chs_set)}")

    missing = configured_chs_set - actual_chs_set
    extra = actual_chs_set - configured_chs_set

    if missing:
        print(f"\n⚠️  MISSING channels (in config but not in data): {sorted(missing)}")
    if extra:
        print(f"\n✓  EXTRA channels (in data but not in config): {sorted(extra)}")

    if not missing:
        print(f"\n✓  All configured channels are available!")
    else:
        print(f"\n❌ Configuration is INVALID - {len(missing)} channels are missing!")

if __name__ == "__main__":
    # Check subject 001 (the one failing)
    subj_id = "001" if len(sys.argv) < 2 else sys.argv[1]

    actual_channels = check_subject_channels(subj_id)

    if actual_channels:
        compare_with_config(subj_id, actual_channels)

    print(f"\n{'='*80}")
    print("To check other subjects, run: python check_channels.py <subject_id>")
    print(f"{'='*80}\n")
