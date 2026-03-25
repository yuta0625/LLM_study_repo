import yaml
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_DIR = Path(__file__).resolve().parents[1]

CONFIG_PATH = BASE_DIR / "configs" / "qwen_zundamon_lora.yaml"
PROMPT_PATH = BASE_DIR / "prompt" / "test_prompt.txt"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompt():
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return f.read().strip()


def main():
    config = load_config()
    prompt = load_prompt()

    model_name = config["model"]["name"]
    adapter_path = config["adapter"]["path"]
    
    print("model_name:", model_name)
    print("adapter_path:", adapter_path)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )

    if adapter_path is not None:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    else:
        model = base_model

    print("model class:", model.__class__)
        


    model.eval()

    messages = [
        {"role": "user", "content": prompt}
    ]

    text_input = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text_input, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,
    )

    generated_ids = output[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print(text)

    output_file = OUTPUT_DIR / "qwen_zundamon.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()