from pathlib import Path
import yaml

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig

import wandb

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_CONFIG_PATH = BASE_DIR / "configs" / "model.yaml"
LORA_CONFIG_PATH = BASE_DIR / "configs" / "lora.yaml"
SFT_CONFIG_PATH = BASE_DIR / "configs" / "sft.yaml"

TRAIN_PATH = BASE_DIR / "data" / "train.jsonl"
VALID_PATH = BASE_DIR / "data" / "valid.jsonl"

def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
    
    
def load_conifgs():
    return {
        "model": load_yaml(MODEL_CONFIG_PATH),
        "lora": load_yaml(LORA_CONFIG_PATH),
        "sft": load_yaml(SFT_CONFIG_PATH),
    }
    

def main() -> None:
    configs = load_conifgs()
    
    # yamlの読み込み
    model_cfg = configs["model"]
    lora_cfg = configs["lora"]
    sft_cfg = configs["sft"]
    
    # datasetの読み込み
    train_dataset = load_dataset("json", data_files=str(TRAIN_PATH))["train"]
    valid_dataset = load_dataset("json", data_files=str(VALID_PATH))["train"]

    
    # modelの読み込み
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_name"])
    
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_name"],
        device_map=model_cfg["device_map"],
        torch_dtype=model_cfg["torch_dtype"],
        trust_remote_code=model_cfg["trust_remote_code"],
    )
    
    # LoRAの情報の読み込み
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
        target_modules=lora_cfg["target_modules"],
    )
    
    # Trainerの読み込み
    args = SFTConfig(
        output_dir=str(BASE_DIR / sft_cfg["output_dir"]),
        num_train_epochs=sft_cfg["num_train_epochs"],
        per_device_train_batch_size=sft_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=sft_cfg["gradient_accumulation_steps"],
        logging_steps=sft_cfg["logging_steps"],
        save_strategy=sft_cfg["save_strategy"],
        eval_strategy=sft_cfg["eval_strategy"],
        eval_steps=sft_cfg["eval_steps"],
        learning_rate=sft_cfg["learning_rate"],
        lr_scheduler_type=sft_cfg["lr_scheduler_type"],
        warmup_ratio=sft_cfg["warmup_ratio"],
        report_to=sft_cfg["report_to"],
        run_name=sft_cfg["run_name"],
        fp16=sft_cfg["fp16"],
        bf16=sft_cfg["bf16"],
        max_length=1024,
    )
    
    # W&B
    wandb.init(
        project="llm_project",
        name=sft_cfg["run_name"],
        config={
            "model": model_cfg,
            "lora": lora_cfg,
            "sft": sft_cfg,
        }
    )
    
    # SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        peft_config=peft_config,
    )
    
    # train
    trainer.train()
    
    # 保存
    trainer.save_model(str(BASE_DIR / sft_cfg["output_dir"]))
    

if __name__ == "__main__":
    main()