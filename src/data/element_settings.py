"""按数据集要素自动应用配置。"""
import os
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

DATA_ROOT = "D:/bishe/WYB"

ELEMENT_SETTINGS: Dict[str, Dict[str, Any]] = {
    "Temperature": {"data_dir": "temperature", "output_dim": 1, "kelvin_to_celsius": True, "neighbourhood_top_k": 3, "semantic_top_k": 3},
    "Cloud": {"data_dir": "cloud_cover", "output_dim": 1, "kelvin_to_celsius": False, "neighbourhood_top_k": 3, "semantic_top_k": 3},
    "Humidity": {"data_dir": "humidity", "output_dim": 1, "kelvin_to_celsius": False, "neighbourhood_top_k": 2, "semantic_top_k": 2},
    "Wind": {"data_dir": "component_of_wind", "output_dim": 2, "kelvin_to_celsius": False, "neighbourhood_top_k": 3, "semantic_top_k": 3},
}


def validate_dataset_selection(config: dict) -> bool:
    selection = config.get("dataset_selection", {})
    active = [k for k, v in selection.items() if v]
    return len(active) == 1


def apply_element_settings(config: dict, logger_inst=None) -> dict:
    log = logger_inst or logger
    selection = config.get("dataset_selection", {})
    active = [k for k, v in selection.items() if v]
    element = active[0] if active else config.get("meta", {}).get("element", "Temperature")
    s = ELEMENT_SETTINGS[element]

    data_dir = os.path.join(DATA_ROOT, s["data_dir"])
    config["meta"]["element"] = element
    config["meta"]["experiment_name"] = f"SingleHyperTKAN_{element}"

    config["data"]["train_path"] = os.path.join(data_dir, "trn.pkl")
    config["data"]["val_path"] = os.path.join(data_dir, "val.pkl")
    config["data"]["test_path"] = os.path.join(data_dir, "test.pkl")
    config["data"]["position_path"] = os.path.join(data_dir, "position.pkl")
    config["data"]["kelvin_to_celsius"] = s["kelvin_to_celsius"]
    config["model"]["output_projection"]["output_dim"] = s["output_dim"]

    if "neighbourhood" in config.get("graph", {}):
        config["graph"]["neighbourhood"]["top_k"] = s["neighbourhood_top_k"]
    if "semantic" in config.get("graph", {}):
        config["graph"]["semantic"]["top_k"] = s["semantic_top_k"]

    log.info(f"Applied element settings: {element}")
    return config
