from pathlib import Path
from datasets import load_dataset

DATASET_NAME = "alfredplpl/simple-zundamon"
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw" / "simple_zundamon"

def main() -> None:
    ds = load_dataset(DATASET_NAME)
    
    RAW_DIR.parent.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(RAW_DIR))

if __name__ == "__main__":
    main()