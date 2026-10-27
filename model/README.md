# Local LLM

Shubhi's first local chat model is Qwen3-0.6B in Q4_0 GGUF format. The quantized weights are about 429 MB. The model card lists the model under the Apache-2.0 license: https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF.

## Transformers learning milestone

`transformers_basics.py` loads DistilGPT2 (82 million parameters) through Hugging Face Transformers and generates a short continuation. Its model weights are about 353 MB and are downloaded to the Hugging Face cache on first run. This is a learning example; it is a smaller, older text completion model than the Qwen chat model above.

Install Transformers into the active project virtual environment, then run the example from the repository root:

```powershell
python -m pip install transformers
python .\model\transformers_basics.py
```

The current project environment already has PyTorch with CUDA support. The example selects the GPU when available and otherwise runs on CPU.

## Fine-tuning learning milestone

After running the Transformers example once, fine-tune the cached DistilGPT2 weights on a tiny demonstration Q&A set about Shubhi:

```powershell
python .\model\fine_tune_distilgpt2.py
```

The script trains for 12 short epochs, saves the resulting model under `model/fine_tuned/` (ignored by Git), reloads that saved checkpoint, and asks it one question. The examples are only for learning the fine-tuning workflow; this small text-completion model is not a production chatbot, and it may produce inaccurate or awkward answers.

## Train a tiny language model from scratch

This CPU/GPU-friendly exercise builds a character vocabulary and a small causal Transformer with new random weights. It trains on `model/data/tiny_lm_corpus.txt`, saves a checkpoint under `model/from_scratch/` (ignored by Git), and generates a short sample:

```powershell
python .\model\train_tiny_lm.py
```

This toy model is for understanding the training loop and next-character prediction. It is not expected to produce reliable answers.

Model weights and the llama.cpp runner are stored locally in `model/weights/` and `model/runner/`. Git ignores both folders so the large binary files are not committed.

## Start the local model server

From the repository root, extract the Windows x64 CPU runner:

```powershell
Expand-Archive -LiteralPath .\model\runner\llama-cpu.zip -DestinationPath .\model\runner\bin -Force
```

Start the server in that terminal:

```powershell
$server = Get-ChildItem .\model\runner -Recurse -Filter llama-server.exe | Select-Object -First 1 -ExpandProperty FullName
& $server -m "$PWD\model\weights\Qwen3-0.6B-Q4_0.gguf" --alias local-model --host 127.0.0.1 --port 8080 --ctx-size 2048 --n-gpu-layers 0
```

Keep the server running. In a second PowerShell terminal, call the model through the Python wrapper:

```powershell
.\.venv\Scripts\python.exe .\model\llm_inference.py
```

To start a multi-turn terminal chat that remembers earlier messages in the current session:

```powershell
.\.venv\Scripts\python.exe -m model.chat_cli
```

Type `/exit` to end the chat. Conversation history stays in memory and is not saved to disk.

Backend code can call the same wrapper:

```python
from model.llm_inference import generate_response

answer = generate_response("Explain a tensor in one sentence.")
```

For follow-up questions, pass previous user and assistant messages in `history`:

```python
history = [
    {"role": "user", "content": "What is a tensor?"},
    {"role": "assistant", "content": "A tensor is a data structure for numbers."},
]
answer = generate_response("Give me a simple example.", history=history)
```

The wrapper uses Python's standard library and sends chat requests to `http://127.0.0.1:8080/v1/chat/completions`. It adds Qwen3's `/no_think` mode so short prompts return a user-facing answer within the token limit.
