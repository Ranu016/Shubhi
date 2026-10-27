"""Fine-tune DistilGPT2 on a tiny Shubhi Q&A dataset and try it afterward."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "distilbert/distilgpt2"
OUTPUT_DIR = Path(__file__).parent / "fine_tuned" / "shubhi-distilgpt2"
EPOCHS = 12
BATCH_SIZE = 2
MAX_LENGTH = 96

# These are demonstration examples, not authoritative project documentation.
QA_PAIRS = [
    ("What is Shubhi?", "Shubhi is a student project to build an AI assistant."),
    ("Which model powers the local LLM demo?", "The demo uses Qwen3-0.6B with llama.cpp."),
    ("What does PyTorch do in this project?", "PyTorch is used to learn tensors and neural-network training."),
    ("What is a tensor?", "A tensor is a container for numerical data, such as a scalar, vector, or matrix."),
    ("Who handles document retrieval?", "The RAG team member handles document processing and retrieval."),
    ("How do I stop the local chat demo?", "Type slash-exit in the terminal chat."),
]


def make_batch(examples, tokenizer):
    encoded = []
    answer_labels = []

    for question, answer in examples:
        prefix = f"Question: {question}\nAnswer:"
        full_text = f"{prefix} {answer}{tokenizer.eos_token}"
        input_ids = tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LENGTH,
        )["input_ids"]
        prefix_length = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])

        encoded.append(input_ids)
        # Train the model to continue with the answer, not repeat the question.
        answer_labels.append([-100] * prefix_length + input_ids[prefix_length:])

    width = max(len(ids) for ids in encoded)
    input_ids_batch = torch.full(
        (len(encoded), width), tokenizer.pad_token_id, dtype=torch.long
    )
    attention_mask = torch.zeros((len(encoded), width), dtype=torch.long)
    labels = torch.full((len(encoded), width), -100, dtype=torch.long)

    for row, (ids, row_labels) in enumerate(zip(encoded, answer_labels)):
        length = len(ids)
        input_ids_batch[row, :length] = torch.tensor(ids)
        attention_mask[row, :length] = 1
        labels[row, :length] = torch.tensor(row_labels)

    return input_ids_batch, attention_mask, labels


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Fine-tuning device:", device)
    print("Loading base model:", MODEL_NAME)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    dataset = make_batch(QA_PAIRS, tokenizer)
    batches = DataLoader(
        list(zip(*dataset)),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda rows: tuple(torch.stack(items) for items in zip(*rows)),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    model.train()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for input_ids, attention_mask, labels in batches:
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            ).loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch}/{EPOCHS}: loss={total_loss / len(batches):.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Fine-tuned model saved to:", OUTPUT_DIR)

    # Reopen the saved checkpoint to verify the artifact is usable on its own.
    del optimizer, model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR).to(device)
    model.eval()
    question = "What is Shubhi?"
    prompt = f"Question: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            repetition_penalty=1.15,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated[0, inputs["input_ids"].shape[1] :]
    print("Test question:", question)
    print("Fine-tuned answer:", tokenizer.decode(new_tokens, skip_special_tokens=True).strip())


if __name__ == "__main__":
    main()
