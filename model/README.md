# Local LLM

Shubhi's first local chat model is Qwen3-0.6B in Q4_0 GGUF format. The quantized weights are about 429 MB. The model card lists the model under the Apache-2.0 license: https://huggingface.co/ggml-org/Qwen3-0.6B-GGUF.

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

Backend code can call the same wrapper:

```python
from model.llm_inference import generate_response

answer = generate_response("Explain a tensor in one sentence.")
```

The wrapper uses Python's standard library and sends chat requests to `http://127.0.0.1:8080/v1/chat/completions`. It adds Qwen3's `/no_think` mode so short prompts return a user-facing answer within the token limit.
