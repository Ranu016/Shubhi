"""Load a small pretrained text-generation model with Hugging Face Transformers."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "distilbert/distilgpt2"


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading model:", MODEL_NAME)
    print("Device:", device)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    prompt = "A tensor in machine learning is"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    print("Prompt:", prompt)
    print("Generated:", tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
