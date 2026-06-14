"""
OpenMind API Server - OpenAI-Compatible Chat Completions API.

Serves the OpenMind model with:
- POST /v1/chat/completions (streaming + non-streaming)
- POST /v1/completions (legacy text completion)
- GET /v1/models (list available models)
- GET /health (health check)
- Static file serving for frontend

Fully compatible with OpenAI client libraries.
"""

import os
import sys
import json
import time
import uuid
import asyncio
import argparse
from pathlib import Path
from typing import Optional, AsyncGenerator

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.models.modeling_openmind import OpenMindModel
from src.models.config_openmind import OpenMindConfig
from src.data.tokenizer import BPETokenizer
from src.data.chat_templates import format_chat, SYSTEM_DEFAULT


# ─── Request/Response Models ──────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role: str = "user"
    content: str = ""


class ChatCompletionRequest(BaseModel):
    model: str = "openmind-125m"
    messages: list[ChatMessage]
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    top_k: int = Field(default=50, ge=0)
    max_tokens: int = Field(default=512, ge=1, le=4096)
    stream: bool = False
    stop: Optional[list[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = Field(default=1.15, ge=0.0)
    template: Optional[str] = "auto"


class CompletionRequest(BaseModel):
    model: str = "openmind-125m"
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    max_tokens: int = 256
    stream: bool = False
    stop: Optional[list[str]] = None
    repetition_penalty: float = 1.15



class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: dict


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "openmind"


# ─── Model Manager ────────────────────────────────────────────────────────────

class HFTokenizerWrapper:
    """Wrapper around HuggingFace tokenizer to match BPETokenizer API."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id
        self.vocab_size = tokenizer.vocab_size

    def encode(self, text, allowed_special=None):
        return self.tokenizer.encode(text)

    def decode(self, ids):
        return self.tokenizer.decode(ids, skip_special_tokens=True)


class ModelManager:
    """Manages model loading and inference."""

    def __init__(self):
        self.model: Optional[OpenMindModel] = None
        self.tokenizer: Optional[BPETokenizer] = None
        self.model_name: str = ""
        self.device: str = "cpu"
        self.chat_template: str = "chat"

    def load(self, model_path: str, device: str = None):
        """Load model and tokenizer from a directory."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        print(f"Loading model from {model_path} on {device}...")
        self.model = OpenMindModel.from_pretrained(model_path, device=device)
        self.model.eval()
        self.model_name = Path(model_path).name

        # Auto-detect prompt template
        if "sft" in model_path.lower() or "aligned" in model_path.lower():
            self.chat_template = "chat"
            print("Auto-detected SFT/Aligned model: default chat template set to 'chat'")
        else:
            self.chat_template = "alpaca"
            print("Auto-detected Base model: default chat template set to 'alpaca' (Instruction-Tuning)")

        # Load tokenizer
        if self.model.config.vocab_size == 50257:
            from transformers import AutoTokenizer
            print("Loading HuggingFace GPT-2 tokenizer...")
            hf_tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.tokenizer = HFTokenizerWrapper(hf_tokenizer)
        else:
            tokenizer_dir = os.path.join(model_path, "tokenizer")
            if os.path.exists(tokenizer_dir):
                self.tokenizer = BPETokenizer.load(tokenizer_dir)
            else:
                # Try parent directory
                for f in os.listdir(model_path):
                    if f.endswith("_vocab.json"):
                        name = f.replace("_vocab.json", "")
                        self.tokenizer = BPETokenizer.load(model_path, name)
                        break

            if self.tokenizer is None:
                print("Warning: No tokenizer found, creating default")
                self.tokenizer = BPETokenizer(vocab_size=32000)

        print(f"Model '{self.model_name}' loaded successfully!")

    def generate_text(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop: Optional[list[str]] = None,
    ) -> str:
        """Generate text from a prompt."""
        input_ids = self.tokenizer.encode(prompt, allowed_special={"all"})
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        output_ids = self.model.generate(
            input_tensor,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            eos_token_id=self.tokenizer.eos_token_id,
            repetition_penalty=repetition_penalty,
        )

        # Decode only the generated tokens
        generated_ids = output_ids[0, len(input_ids):].tolist()
        response_text = self.tokenizer.decode(generated_ids)

        if stop:
            for stop_seq in stop:
                if stop_seq in response_text:
                    response_text = response_text.split(stop_seq)[0]

        return response_text

    async def stream_generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        stop: Optional[list[str]] = None,
    ) -> AsyncGenerator[str, None]:
        """Stream-generate tokens one at a time."""
        input_ids = self.tokenizer.encode(prompt, allowed_special={"all"})
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)

        past_key_values = [None] * self.model.config.n_layers
        generated = input_tensor
        generated_text = ""

        for _ in range(max_tokens):
            if past_key_values[0] is not None:
                curr_input = generated[:, -1:]
            else:
                curr_input = generated

            with torch.no_grad():
                outputs = self.model(
                    curr_input,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            logits = outputs["logits"][:, -1, :]
            past_key_values = outputs["past_key_values"]

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(logits.shape[0]):
                    for token_id in set(generated[i].tolist()):
                        logit = logits[i, token_id].item()
                        if logit < 0:
                            logits[i, token_id] = logit * repetition_penalty
                        else:
                            logits[i, token_id] = logit / repetition_penalty

            # Apply temperature
            logits = logits / max(temperature, 1e-8)

            # Top-k filtering
            if top_k > 0:
                top_k_vals = torch.topk(logits, min(top_k, logits.size(-1)))
                mask = logits < top_k_vals.values[..., -1, None]
                logits[mask] = float("-inf")

            # Top-p filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_remove = cumulative_probs > top_p
                sorted_remove[..., 1:] = sorted_remove[..., :-1].clone()
                sorted_remove[..., 0] = 0
                remove = sorted_remove.scatter(1, sorted_indices, sorted_remove)
                logits[remove] = float("-inf")

            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

            token_id = next_token[0, 0].item()
            if token_id == self.tokenizer.eos_token_id:
                break

            token_text = self.tokenizer.decode([token_id])

            potential_text = generated_text + token_text
            stopped = False
            if stop:
                for stop_seq in stop:
                    if stop_seq in potential_text:
                        stop_idx = potential_text.find(stop_seq)
                        yield potential_text[len(generated_text):stop_idx]
                        stopped = True
                        break
            if stopped:
                break

            generated_text = potential_text
            yield token_text

            # Small delay for streaming effect
            await asyncio.sleep(0)



# ─── FastAPI Application ──────────────────────────────────────────────────────

app = FastAPI(
    title="OpenMind API",
    description="OpenAI-compatible API for the OpenMind language model",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model manager
manager = ModelManager()


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "ok",
        "model_loaded": manager.model is not None,
        "model_name": manager.model_name,
        "device": manager.device,
    }


@app.get("/v1/models")
async def list_models():
    """List available models."""
    models = []
    if manager.model is not None:
        models.append(ModelInfo(
            id=manager.model_name,
            created=int(time.time()),
        ))
    return {"object": "list", "data": [m.dict() for m in models]}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    if manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Determine template
    template = request.template
    if not template or template == "auto":
        template = manager.chat_template

    # Format messages into prompt
    messages = [{"role": m.role, "content": m.content} for m in request.messages]
    prompt = format_chat(messages, add_generation_prompt=True, template=template)

    # Determine stop sequences
    stop_sequences = request.stop
    if not stop_sequences:
        if template == "chat":
            stop_sequences = ["<|user|>", "<|system|>", "<|endoftext|>"]
        elif template == "alpaca":
            stop_sequences = ["###", "Instruction:", "Response:", "<|endoftext|>"]
        else:
            stop_sequences = ["<|endoftext|>"]

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    if request.stream:
        return StreamingResponse(
            _stream_chat_response(
                completion_id, prompt, request, stop_sequences
            ),
            media_type="text/event-stream",
        )

    # Non-streaming response
    response_text = manager.generate_text(
        prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
        stop=stop_sequences,
    )

    # Count tokens (approximate)
    prompt_tokens = len(manager.tokenizer.encode(prompt, allowed_special={"all"}))
    completion_tokens = len(manager.tokenizer.encode(response_text))

    return ChatCompletionResponse(
        id=completion_id,
        created=int(time.time()),
        model=manager.model_name,
        choices=[
            ChatCompletionChoice(
                message=ChatMessage(role="assistant", content=response_text),
                finish_reason="stop",
            )
        ],
        usage={
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    )


async def _stream_chat_response(
    completion_id: str,
    prompt: str,
    request: ChatCompletionRequest,
    stop_sequences: Optional[list[str]],
) -> AsyncGenerator[str, None]:
    """Generate streaming SSE response."""
    # Initial chunk with role
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": manager.model_name,
        "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
    }
    yield f"data: {json.dumps(chunk)}\n\n"

    # Stream tokens
    async for token in manager.stream_generate(
        prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
        stop=stop_sequences,
    ):
        chunk = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": manager.model_name,
            "choices": [{"index": 0, "delta": {"content": token}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"

    # Final chunk
    chunk = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": manager.model_name,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/completions")
async def text_completions(request: CompletionRequest):
    """Legacy text completion endpoint."""
    if manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # For text completions, default stop sequence is <|endoftext|> if not provided
    stop_sequences = request.stop or ["<|endoftext|>"]

    response_text = manager.generate_text(
        request.prompt,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        repetition_penalty=request.repetition_penalty,
        stop=stop_sequences,
    )

    return {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": manager.model_name,
        "choices": [{"text": response_text, "index": 0, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": len(manager.tokenizer.encode(request.prompt)),
            "completion_tokens": len(manager.tokenizer.encode(response_text)),
        },
    }


# ─── Static File Serving ──────────────────────────────────────────────────────

def setup_static_files(app: FastAPI, frontend_dir: str = None):
    """Mount frontend static files."""
    if frontend_dir is None:
        frontend_dir = os.path.join(
            Path(__file__).resolve().parent.parent.parent, "frontend", "dist"
        )

    if os.path.exists(frontend_dir):
        app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
        print(f"Serving frontend from {frontend_dir}")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def start_server(
    model_path: str,
    host: str = "0.0.0.0",
    port: int = 8000,
    device: str = None,
    chat_template: str = "auto",
):
    """Start the API server."""
    manager.load(model_path, device)
    if chat_template != "auto":
        manager.chat_template = chat_template

    # Setup frontend
    frontend_dir = os.path.join(
        Path(__file__).resolve().parent.parent.parent, "frontend"
    )
    if os.path.exists(frontend_dir):
        # Serve the frontend index.html at root
        @app.get("/")
        async def serve_frontend():
            index_path = os.path.join(frontend_dir, "index.html")
            if os.path.exists(index_path):
                return FileResponse(index_path)
            return {"message": "OpenMind API is running"}

        # Serve static assets
        for subdir in ["css", "js", "assets"]:
            asset_dir = os.path.join(frontend_dir, subdir)
            if os.path.exists(asset_dir):
                app.mount(f"/{subdir}", StaticFiles(directory=asset_dir), name=subdir)

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OpenMind API Server")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--chat-template", type=str, default="auto", choices=["auto", "chat", "alpaca", "raw"], help="Chat template override")
    args = parser.parse_args()

    start_server(args.model, args.host, args.port, args.device, args.chat_template)

