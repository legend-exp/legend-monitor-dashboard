from __future__ import annotations

from pathlib import Path

import numpy as np
from dbetto import Props
from legendmeta import LegendMetadata

sort_dict = {
    "String": {
        "out_key": "{key}:{k:02}",
        "primary_key": "location.string",
        "secondary_key": "location.position",
    },
    "CC4": {
        "out_key": "{key}:{k}",
        "primary_key": "electronics.cc4.id",
        "secondary_key": "electronics.cc4.channel",
    },
    "HV": {
        "out_key": "{key}:{k:02}",
        "primary_key": "voltage.card.id",
        "secondary_key": "voltage.channel",
    },
    "Det_Type": {"out_key": "{k}", "primary_key": "type", "secondary_key": "name"},
    "DAQ": {"out_key": None, "primary_key": None, "secondary_key": None},
}


def gen_run_dict(path):
    prod_config = Path(path) / "dataflow-config.yaml"
    prod_config = Props.read_from(prod_config, subst_pathvar=True)  # ["setups"]["l200"]
    par_file = Path(prod_config["paths"]["par_hit"]) / "validity.yaml"
    run_dict = {}
    file = Props.read_from(par_file)
    for entry in file:
        experiment, period, run, _, _, _ = entry["apply"][0].split("/")[-1].split("-")
        timestamp = entry["valid_from"]
        if (
            Path(prod_config["paths"]["par_hit"])
            / f"cal/{period}/{run}"
            / f"{experiment}-{period}-{run}-cal-{timestamp}-par_hit.yaml"
        ).exists():
            if period in run_dict:
                run_dict[period][run] = {
                    "experiment": experiment,
                    "timestamp": timestamp,
                }
            else:
                run_dict[period] = {
                    run: {"experiment": experiment, "timestamp": timestamp}
                }
    return run_dict


def sorter(path, timestamp, key="String", datatype="cal", spms=False):
    prod_config = Path(path) / "dataflow-config.yaml"
    prod_config = Props.read_from(prod_config, subst_pathvar=True)  # ["setups"]["l200"]

    cfg_file = prod_config["paths"]["metadata"]
    configs = LegendMetadata(path=cfg_file)
    chmap = configs.channelmap(timestamp)

    det_status_path = prod_config["paths"]["detector_status"]
    det_status = LegendMetadata(path=det_status_path).statuses.on(
        timestamp, system=datatype
    )

    out_dict = {}
    # SiPMs sorting
    if spms:
        chmap = chmap.map("system", unique=False)["spms"]
        if key == "Barrel":
            mapping = chmap.map("name")
            for pos in ["top", "bottom"]:
                for barrel in ["IB", "OB"]:
                    out_dict[f"{barrel}-{pos}"] = [
                        k
                        for k, entry in sorted(mapping.items())
                        if barrel in entry["location"]["fiber"]
                        and pos in entry["location"]["position"]
                    ]
        return out_dict, chmap

    # Daq needs special item as sort on tertiary key
    if key == "DAQ":
        mapping = chmap.map("daq.crate", unique=False)
        for k, entry in sorted(mapping.items()):
            for m, item in sorted(entry.map("daq.card.id", unique=False).items()):
                out_dict[f"DAQ:Cr{k:02},Ch{m:02}"] = [
                    item.map("daq.channel")[pos].daq.name
                    for pos in sorted(item.map("daq.channel"))
                    if item.map("daq.channel")[pos].system == "geds"
                ]
    else:
        out_key = sort_dict[key]["out_key"]
        primary_key = sort_dict[key]["primary_key"]
        secondary_key = sort_dict[key]["secondary_key"]
        mapping = chmap.map(primary_key, unique=False)
        for k, entry in sorted(mapping.items()):
            out_dict[out_key.format(key=key, k=k)] = [
                entry.map(secondary_key)[pos].name
                for pos in sorted(entry.map(secondary_key))
                if entry.map(secondary_key)[pos].system == "geds"
            ]

    out_dict = {
        entry: out_dict[entry] for entry in list(out_dict) if len(out_dict[entry]) > 0
    }
    return out_dict, det_status, chmap


def get_characterization(x, key):
    try:
        return x["manufacturer"][key]
    except KeyError:
        return np.nan


def get_production(x, key):
    try:
        return x[key]
    except KeyError:
        return np.nan
