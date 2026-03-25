import json
from pathlib import Path
from typing import Any, TypedDict

from datasets import Dataset, DatasetDict, load_from_disk

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "simple_zundamon"
TRAIN_PATH = BASE_DIR / "data" / "train.jsonl"
VALID_PATH = BASE_DIR / "data" / "valid.jsonl"

class ChatMessage(TypedDict):
    role: str
    content: str
    
class TrainRecord(TypedDict):
    messages: list[ChatMessage]

    
def load_raw_dataset():
    ds: DatasetDict = load_from_disk(str(RAW_DATA_DIR))
    return ds


def save_jsonl(records: list[TrainRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            
            
def main() -> None:
    ds = load_raw_dataset()

    # trainsplitの取得を行う
    train_split = ds["train"]
    # 8:2での分割を行う
    split_ds = train_split.train_test_split(test_size=0.2)
    
    train_records = list(split_ds["train"])
    valid_records = list(split_ds["test"])

    save_jsonl(train_records, TRAIN_PATH)
    save_jsonl(valid_records, VALID_PATH)


if __name__ == "__main__":
    main()