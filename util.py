from legendmeta import LegendMetadata
from legendmeta.catalog import Props
import os, json

sort_dict = {"String":{"out_key":"{key}:{k:02}",
                               "primary_key":"location.string", 
                               "secondary_key":"location.position"},
                "CC4":{"out_key":"{key}:{k}",
                               "primary_key":"electronics.cc4.id", 
                               "secondary_key":"electronics.cc4.channel"},
                "HV":{"out_key":"{key}:{k:02}",
                      "primary_key":"voltage.card.id", 
                      "secondary_key":"voltage.channel"},
          "Det_Type":{"out_key":"{k}",
                            "primary_key":"type", 
                               "secondary_key":"name"},
            "DAQ":{"out_key":None,
                            "primary_key":None, 
                               "secondary_key":None}}

def gen_run_dict(path):
    
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]
    
    par_file = os.path.join(prod_config["paths"]["par"], 'validity.jsonl')
    run_dict = {}
    with open(par_file, 'r') as file:
        for json_str in file:
            run_dict[json.loads(json_str)["apply"][0].split("-")[2]] = {"experiment":json.loads(json_str)["apply"][0].split("-")[0],
                                                                        "period":json.loads(json_str)["apply"][0].split("-")[1],
                                                                        "timestamp":json.loads(json_str)["valid_from"]}
    out_dict={}
    for run in run_dict:
        if os.path.isfile(os.path.join(prod_config["paths"]["par_hit"], f"cal/{run_dict[run]['period']}/{run}/{run_dict[run]['experiment']}-{run_dict[run]['period']}-{run}-cal-{run_dict[run]['timestamp']}-par_hit.json")):
            out_dict[run] = run_dict[run]    
    return out_dict

def sorter(path, timestamp, key="String", datatype="cal"):
    prod_config = os.path.join(path,"config.json")
    prod_config = Props.read_from(prod_config, subst_pathvar=True)["setups"]["l200"]

    cfg_file = prod_config["paths"]["metadata"]
    configs = LegendMetadata(path = cfg_file)
    chmap = configs.channelmap(timestamp).map("daq.rawid")
    
    software_config_path = prod_config["paths"]["config"]
    software_config_db = LegendMetadata(path = software_config_path)
    software_config = software_config_db.on(timestamp, system=datatype).analysis
    
    out_dict={}
    #Daq needs special item as sort on tertiary key
    if key =="DAQ":
        mapping = chmap.map("daq.crate", unique=False)
        for k,entry  in sorted(mapping.items()):
            for m, item in sorted(entry.map("daq.card.id", unique=False).items()):
                out_dict[f"DAQ:Cr{k:02},Ch{m:02}"] = [item.map("daq.channel")[pos].daq.rawid for pos in sorted(item.map("daq.channel")) if item.map("daq.channel")[pos].system=="geds"]
    else:
        out_key = sort_dict[key]["out_key"]
        primary_key= sort_dict[key]["primary_key"]
        secondary_key = sort_dict[key]["secondary_key"]
        mapping = chmap.map(primary_key, unique=False)
        for k,entry  in sorted(mapping.items()):
            out_dict[out_key.format(key=key,k=k)]=[entry.map(secondary_key)[pos].daq.rawid for pos in sorted(entry.map(secondary_key)) if entry.map(secondary_key)[pos].system=="geds"]
    
    out_dict = {entry:out_dict[entry] for entry in list(out_dict) if len(out_dict[entry])>0}
    return out_dict, software_config, chmap