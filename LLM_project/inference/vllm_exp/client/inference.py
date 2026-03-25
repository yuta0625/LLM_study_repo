from __future__ import annotations

from pathlib import Path
import yaml
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parents[1]
SERVER_CONFIG_DIR = BASE_DIR / "configs" / "server"
REQUEST_CONFIG_PATH = BASE_DIR / "configs" / "reqest.yaml"


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prompt(prompt_path: Path) -> str:
    with prompt_path.open("r", encoding="utf-8") as f:
        return f.read().strip()


def choose_server_config(model_key: str) -> dict:
    config_path = SERVER_CONFIG_DIR / f"{model_key}.yaml"
    return load_yaml(config_path)


def build_base_url(server_cfg: dict) -> str:
    port = server_cfg["server"]["port"]
    return f"http://127.0.0.1:{port}/v1"


def main() -> None:
    model_key = "qwen_zundamon"

    server_cfg = choose_server_config(model_key)
    request_cfg = load_yaml(REQUEST_CONFIG_PATH)

    selected_name = request_cfg["selected"]
    selected_cfg = request_cfg[selected_name]

    model_name = server_cfg["model"]["name"]
    base_url = build_base_url(server_cfg)

    prompt_path = BASE_DIR / request_cfg["prompt_file"]
    prompt_text = load_prompt(prompt_path)

    client = OpenAI(
        base_url=base_url,
        api_key="dummy",
    )

    messages = [
        {
            "role": "user",
            "content": prompt_text,
        }
    ]

    extra_body = {
        "chat_template_kwargs": {
            "enable_thinking": selected_cfg["enable_thinking"]
        }
    }

    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=selected_cfg["temperature"],
        max_tokens=selected_cfg["max_tokens"],
        extra_body=extra_body,
    )

    text = response.choices[0].message.content

    save_dir = BASE_DIR / request_cfg["save_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    output_path = save_dir / f"{model_key}_{selected_name}.txt"

    result_text = (
        f"model: {model_name}\n"
        f"temperature: {selected_cfg['temperature']}\n"
        f"max_tokens: {selected_cfg['max_tokens']}\n"
        f"enable_thinking: {selected_cfg['enable_thinking']}\n"
        f"prompt:{prompt_text}\n"
        f"response:{text}\n"
    )

    with output_path.open("w", encoding="utf-8") as f:
        f.write(result_text)


if __name__ == "__main__":
    main()