import os
import json
import re
from src.config import config


class QADataset:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name.lower()
        benchmark_path = config["benchmark_dataset_json"]

        with open(benchmark_path, "r") as f:
            benchmark = json.load(f)

        if self.dataset_name not in benchmark:
            raise KeyError(f"{dataset_name} not found in benchmark file")

        self.dataset = benchmark[self.dataset_name]
        self.index = sorted(self.dataset.keys())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.dataset[self.index[key]]
        elif isinstance(key, slice):
            return [self.__getitem__(i) for i in range(len(self))[key]]
        else:
            raise KeyError("Key must be int or slice")


def locate_answer(sentence: str) -> str:
    patterns = [
        r"^\s*(A|B|C|D)$",
        r"^\s*(A|B|C|D)\.",
        r"^\s*(A|B|C|D):",
        r"^\s*(A|B|C|D),",
        r"^\s*(A|B|C|D)/",
        r"^\s*(A|B|C|D) or",
        r"^\s*(A|B|C|D) and",
        r"[Oo]ption (A|B|C|D)",
        r":\s*(A|B|C|D)",
        r"^\s*(A|B|C|D)\"",
    ]
    for pattern in patterns:
        match = re.findall(pattern, sentence)
        if match:
            return match[0].upper()

    # fallback: scan sentences containing "answer"
    for part in sentence.split("."):
        if "answer" in part.lower():
            for opt in ["A", "B", "C", "D"]:
                if opt in part.split():
                    return opt
    return "No"