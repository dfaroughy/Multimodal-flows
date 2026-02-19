#!/usr/bin/env python3
"""Extract event-level features from a CMS NanoAOD ROOT file with uproot.

Example:
python scripts/extract_cms_nanoaod.py \
  --input root://eospublic.cern.ch//eos/opendata/cms/path/to/NANOAOD.root \
  --output data/cms2016_50k_events.csv \
  --max-events 50000
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import awkward as ak
import numpy as np
import uproot


def _leading_or_default(array: ak.Array, default: float = np.nan) -> np.ndarray:
    """Return leading object value per event or default when collection is empty."""
    return ak.to_numpy(ak.fill_none(ak.firsts(array), default))


def extract_event_level(
    input_file: str,
    max_events: int,
    tree_name: str = "Events",
) -> dict[str, np.ndarray]:
    branches = [
        "event",
        "run",
        "luminosityBlock",
        "nMuon",
        "nElectron",
        "nJet",
        "MET_pt",
        "MET_phi",
        "Muon_pt",
        "Muon_eta",
        "Jet_pt",
        "Jet_eta",
    ]

    with uproot.open(input_file) as root_file:
        if tree_name not in root_file:
            raise KeyError(
                f"Tree '{tree_name}' not found in {input_file}. "
                f"Available keys: {list(root_file.keys())[:10]}"
            )

        events = root_file[tree_name].arrays(branches, library="ak", entry_stop=max_events)

    muon_pt = events["Muon_pt"]
    jet_pt = events["Jet_pt"]

    return {
        "event": ak.to_numpy(events["event"]),
        "run": ak.to_numpy(events["run"]),
        "luminosityBlock": ak.to_numpy(events["luminosityBlock"]),
        "nMuon": ak.to_numpy(events["nMuon"]),
        "nElectron": ak.to_numpy(events["nElectron"]),
        "nJet": ak.to_numpy(events["nJet"]),
        "MET_pt": ak.to_numpy(events["MET_pt"]),
        "MET_phi": ak.to_numpy(events["MET_phi"]),
        "leadMuon_pt": _leading_or_default(muon_pt),
        "leadMuon_absEta": np.abs(_leading_or_default(events["Muon_eta"])),
        "leadJet_pt": _leading_or_default(jet_pt),
        "leadJet_absEta": np.abs(_leading_or_default(events["Jet_eta"])),
        "HT": ak.to_numpy(ak.sum(jet_pt, axis=1)),
    }


def write_output(columns: dict[str, np.ndarray], output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    headers = list(columns.keys())

    if output_file.suffix.lower() == ".npz":
        np.savez_compressed(output_file, **columns)
        return

    with output_file.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        nrows = len(columns[headers[0]])
        for i in range(nrows):
            writer.writerow([columns[h][i] for h in headers])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        help="NanoAOD ROOT file path (local path or root:// URL).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cms2016_50k_events.csv"),
        help="Output table path (.csv or .npz).",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=50_000,
        help="Maximum number of events to read from the Events tree.",
    )
    parser.add_argument(
        "--tree-name",
        default="Events",
        help="Name of the NanoAOD TTree (default: Events).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    columns = extract_event_level(
        input_file=args.input,
        max_events=args.max_events,
        tree_name=args.tree_name,
    )
    write_output(columns, args.output)

    print(f"Read {len(columns['event']):,} events from: {args.input}")
    print(f"Wrote event-level table: {args.output}")
    print("Columns:", ", ".join(columns.keys()))


if __name__ == "__main__":
    main()
