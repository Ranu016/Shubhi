"""Train a tiny character-level causal Transformer from random weights."""

from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).parent
CORPUS_PATH = HERE / "data" / "tiny_lm_corpus.txt"
OUTPUT_PATH = HERE / "from_scratch" / "tiny_char_transformer.pt"

CONTEXT_LENGTH = 64
BATCH_SIZE = 32
TRAIN_STEPS = 800
LEARNING_RATE = 3e-4
EMBEDDING_SIZE = 64
NUM_HEADS = 4
NUM_LAYERS = 2


class CausalBlock(nn.Module):
    """A causal self-attention block followed by a feed-forward network."""

    def __init__(self) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(EMBEDDING_SIZE)
        self.attention = nn.MultiheadAttention(
            EMBEDDING_SIZE,
            NUM_HEADS,
            batch_first=True,
        )
        self.feed_forward_norm = nn.LayerNorm(EMBEDDING_SIZE)
        self.feed_forward = nn.Sequential(
            nn.Linear(EMBEDDING_SIZE, 4 * EMBEDDING_SIZE),
            nn.GELU(),
            nn.Linear(4 * EMBEDDING_SIZE, EMBEDDING_SIZE),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.size(1)
        future_mask = torch.triu(
            torch.ones(length, length, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        normalized = self.attention_norm(x)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=future_mask,
            need_weights=False,
        )
        x = x + attended
        return x + self.feed_forward(self.feed_forward_norm(x))


class TinyCharacterTransformer(nn.Module):
    def __init__(self, vocabulary_size: int) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocabulary_size, EMBEDDING_SIZE)
        self.position_embedding = nn.Embedding(CONTEXT_LENGTH, EMBEDDING_SIZE)
        self.blocks = nn.Sequential(*(CausalBlock() for _ in range(NUM_LAYERS)))
        self.final_norm = nn.LayerNorm(EMBEDDING_SIZE)
        self.language_head = nn.Linear(EMBEDDING_SIZE, vocabulary_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        batch_size, length = token_ids.shape
        positions = torch.arange(length, device=token_ids.device)
        x = self.token_embedding(token_ids) + self.position_embedding(positions)
        x = self.blocks(x)
        return self.language_head(self.final_norm(x))


def make_batch(
    encoded_text: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    starts = torch.randint(
        0,
        len(encoded_text) - CONTEXT_LENGTH - 1,
        (BATCH_SIZE,),
    )
    inputs = torch.stack([encoded_text[start : start + CONTEXT_LENGTH] for start in starts])
    targets = torch.stack(
        [encoded_text[start + 1 : start + CONTEXT_LENGTH + 1] for start in starts]
    )
    return inputs.to(device), targets.to(device)


@torch.inference_mode()
def generate(
    model: TinyCharacterTransformer,
    prompt_ids: torch.Tensor,
    count: int,
) -> torch.Tensor:
    generated = prompt_ids
    for _ in range(count):
        context = generated[:, -CONTEXT_LENGTH:]
        next_character_logits = model(context)[:, -1, :]
        next_character = torch.multinomial(
            F.softmax(next_character_logits, dim=-1), num_samples=1
        )
        generated = torch.cat((generated, next_character), dim=1)
    return generated


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = CORPUS_PATH.read_text(encoding="utf-8")
    vocabulary = sorted(set(text))
    character_to_id = {character: index for index, character in enumerate(vocabulary)}
    id_to_character = {index: character for character, index in character_to_id.items()}
    encoded_text = torch.tensor([character_to_id[character] for character in text])

    model = TinyCharacterTransformer(len(vocabulary)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    print("Training device:", device)
    print("Training from random weights on:", CORPUS_PATH.name)
    print("Vocabulary size:", len(vocabulary), "characters")
    print("Model parameters:", sum(parameter.numel() for parameter in model.parameters()))

    model.train()
    for step in range(1, TRAIN_STEPS + 1):
        inputs, targets = make_batch(encoded_text, device)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, len(vocabulary)), targets.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 200 == 0:
            print(f"Step {step}/{TRAIN_STEPS}: loss={loss.item():.4f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocabulary": vocabulary,
            "embedding_size": EMBEDDING_SIZE,
            "num_heads": NUM_HEADS,
            "num_layers": NUM_LAYERS,
            "context_length": CONTEXT_LENGTH,
        },
        OUTPUT_PATH,
    )
    print("Saved model to:", OUTPUT_PATH)

    model.eval()
    prompt = "shubhi learns"
    prompt_ids = torch.tensor(
        [[character_to_id[character] for character in prompt]],
        device=device,
    )
    sample = generate(model, prompt_ids, count=180)[0].tolist()
    print("Generated text:", "".join(id_to_character[index] for index in sample))


if __name__ == "__main__":
    main()
