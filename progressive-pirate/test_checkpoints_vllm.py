#!/usr/bin/env python

import asyncio
import os
from pathlib import Path

import openai

# --- Configuration ---
# This should be the directory containing your final adapter and checkpoint subdirectories.
LORA_OUTPUT_DIR = "./outputs-steps/lora-out"
LORA_OUTPUT_DIR = "./outputs-vllm-compat/lora-out"

# The prompt you want to test on each checkpoint.
PROMPT = "Whats your favorite drink?"

# The file where all raw responses will be saved.
OUTPUT_FILE = "checkpoint_responses_vllm.txt"

# vLLM Server details
VLLM_HOST = "localhost"
VLLM_PORT = 8000
# This is the name of the base model served by vLLM
VLLM_MODEL_NAME = "NousResearch/Llama-3.2-1B"
# --- End Configuration ---

# Point the OpenAI client to the local vLLM server
client = openai.AsyncOpenAI(
    base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1",
    api_key="dummy-key",  # required but not used by vLLM
)


async def process_lora_adapter(lora_path: Path):
    """Loads an adapter, runs inference, and then unloads it."""
    lora_name = lora_path.name
    print(f"Processing: {lora_path}")

    try:
        # 1. Load the LoRA adapter
        print(f"  -> Loading adapter: {lora_name}")
        await client.post(
            "/load_lora_adapter",
            body={
                "lora_name": lora_name,
                "lora_path": str(lora_path.resolve()),
            },
            cast_to=object,
        )

        # 2. Run inference with the loaded adapter
        print(f"  -> Running inference with: {lora_name}")
        completion = await client.completions.create(
            model=lora_name,
            prompt=PROMPT,
            max_tokens=256,
            temperature=0.7,
#            extra_body={"lora_request": {"lora_name": lora_name}},
        )
        response = completion.choices[0].text

    except Exception as e:
        response = f"ERROR processing {lora_name}: {e}"
    finally:
        # 3. Unload the LoRA adapter to free up the slot
        print(f"  -> Unloading adapter: {lora_name}")
        try:
            await client.post(
                "/unload_lora_adapter",
                body={"lora_name": lora_name},
                cast_to=object,
            )
        except Exception as e:
            print(f"Warning: Failed to unload adapter {lora_name}: {e}")

    return response


async def main():
    """Finds all checkpoints and runs inference on them using vLLM."""
    root_dir = Path(LORA_OUTPUT_DIR)
    checkpoint_dirs = sorted(
        [d for d in root_dir.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda x: int(x.name.split("-")[-1]),
    )
    all_dirs_to_test = [root_dir] + checkpoint_dirs

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for lora_dir in all_dirs_to_test:
            response = await process_lora_adapter(lora_dir)
            header = f"========================================================================\nProcessing: {lora_dir}\n========================================================================\n\n"
            f_out.write(header)
            f_out.write(f"Prompt: {PROMPT}\n\n")
            f_out.write(f"Response:\n{response.strip()}\n\n\n")
            print(f"Finished: {lora_dir}\n")

    print(f"✅ All checkpoints processed successfully!")
    print(f"Your dataset has been created at: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())

# ERROR 08-31 09:02:08 [core.py:700] ValueError: vLLM only supports modules_to_save being None. vLLM does not yet support DoRA.
