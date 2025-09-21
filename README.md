<img width="512" height="512" alt="LaQuisha" src="https://github.com/user-attachments/assets/b9b3b833-8e47-47c7-b707-5c597b5baa36" />

# LaQuisha AI 👑

A booty-licious FastAPI twext browser and backend designed to run GGUF models with Llama.cpp. LaQuisha brings that real talk with some AI magic ✨

## 🚀 Quick Start

### 1. Setup
```bash
# Clone the repository (if not already done)
git clone https://github.com/Fortnumsound/LaQuisha-fast-API-backend-for-Llama.cpp.git
cd LaQuisha-fast-API-backend-for-Llama.cpp

# Run the setup script
python setup.py
```

### 2. Start the Server
```bash
python laquisha_backend.py
```

### 3. Test the API
```bash
python test_laquisha.py
```

## 📚 API Endpoints

- **GET /** - Welcome message and endpoint info
- **GET /health** - Health check and status
- **POST /v1/chat/completions** - Chat with LaQuisha (OpenAI-compatible)

## 💬 Example Usage

```bash
curl -X POST "http://localhost:8000/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "laquisha-7b",
    "messages": [{"role": "user", "content": "Hello LaQuisha!"}],
    "sass_level": 8
  }'
```

## 🧠 Adding Real AI (Optional)

For full AI functionality, install llama-cpp-python and add a GGUF model:

```bash
pip install llama-cpp-python
# Download a GGUF model to ./models/ directory
```

LaQuisha works without a model (using witty fallback responses) but gets smarter with real AI!

## 🔧 Configuration

Set these environment variables to customize LaQuisha:

- `MODEL_PATH` - Path to your GGUF model file
- `UVICORN_HOST` - Server host (default: 0.0.0.0)
- `UVICORN_PORT` - Server port (default: 8000)

---

## 📖 Full Source Code

Below is the complete source code for reference:

```python

This module exposes a small FastAPI application that wraps a llama.cpp model.  It's
designed to run a local LLaMA models (in this case the LLaMA‑3 7B model) using the
`llama_cpp` python bindings.  The code is resilient to environments where the
`llama_cpp` module is not available: in that case it will initialise a
placeholder model that returns a friendly error message rather than failing to
import.  To run the server you will need a GGUF model file and the
``llama_cpp`` library installed.  See the README accompanying this file for
details.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Any
import os
import random

try:
    # Attempt to import llama_cpp.  If this fails the server will still start
    # but will return an informative error when called.
    import llama_cpp  # type: ignore
except ImportError:
    llama_cpp = None  # type: ignore


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
#
# ``MODEL_PATH`` points at the GGUF model file to load.  You can override
# this with an environment variable when starting the server:
#
#     MODEL_PATH=/path/to/llama-3-7b.Q4_0.gguf uvicorn laquisha_backend:app
#
# The default points at a relative location that assumes there is a ``models``
# directory next to this file.
MODEL_PATH = os.getenv("MODEL_PATH", "./models/llama-3-7b-chat.Q4_0.gguf")

# Maximum number of tokens to generate per request.  Adjust this to balance
# response length and performance.
MAX_TOKENS = 512

# Temperature controls the randomness of the model's output.  Higher values
# produce more varied responses, lower values make the output more predictable.
TEMPERATURE = 0.8

# A handful of colourful expressions that LaQuisha may sprinkle into her
# responses.  Feel free to customise these to suit your personality.
LAQUISHA_QUOTES = [
    "Chile, let me tell you something...",
    "Honey, hold up—let me break this down for you.",
    "Bless your heart, but you need to hear this...",
    "Oop! Let me keep it a buck with you...",
    "Baby, I'm 'bout to serve you some realness right now."
]


# -----------------------------------------------------------------------------
# Model initialisation
# -----------------------------------------------------------------------------
# Initialise the model up front so that the first request doesn't pay the cost
# of loading the weights.  If llama_cpp isn't available, we'll initialise a
# placeholder model that always raises an exception when invoked.

class _FallbackModel:
    """A fallback model used when llama_cpp isn't installed or fails to load.

    Calling this model returns a dictionary shaped like the llama_cpp output
    but containing an error message.  This allows the API to remain
    responsive even if the underlying model isn't available.
    """
    def __call__(self, *args, **kwargs) -> dict[str, Any]:
        return {
            "choices": [
                {
                    "text": "LaQuisha's brain isn't loaded. Please install llama_cpp and provide a valid model file to enable responses."
                }
            ]
        }


if llama_cpp is not None:
    try:
        # Instantiate the Llama model.  Use 2048 context length and limit to 4
        # threads by default.  These settings can be adjusted based on your
        # hardware capabilities.
        llm = llama_cpp.Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,
            n_threads=4,
            verbose=False,
        )
    except Exception as exc:
        # If initialisation fails (e.g. missing file) we fall back to the
        # placeholder and log a warning to stderr.
        import sys

        print(f"[warn] Failed to initialise LLaMA model: {exc}", file=sys.stderr)
        llm = _FallbackModel()  # type: ignore
else:
    llm = _FallbackModel()  # type: ignore


# -----------------------------------------------------------------------------
# FastAPI application
# -----------------------------------------------------------------------------
app = FastAPI(
    title="LaQuisha AI",
    description="When you need that real talk with some AI magic ✨",
    version="1.0.0",
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "laquisha-7b"  # LaQuisha's model name
    messages: List[ChatMessage]
    temperature: Optional[float] = TEMPERATURE
    max_tokens: Optional[int] = MAX_TOKENS
    stream: Optional[bool] = False
    sass_level: Optional[int] = 7  # 1-10, how much shade should LaQuisha throw?


class ChatResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Any]
    laquisha_flavor: Optional[str] = None


def get_laquisha_flavor() -> str:
    """Pick one of LaQuisha's signature phrases at random."""
    return random.choice(LAQUISHA_QUOTES)


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest) -> ChatResponse:
    """Generate a response for the given chat request.

    This endpoint mimics the OpenAI chat completion API.  It accepts a
    conversation history and returns the assistant's next message.  If the
    underlying model isn't available the response will include an error
    message instead of a generated reply.
    """
    try:
        # Clamp sass level between 1 and 10
        sass_level = min(max(request.sass_level or 7, 1), 10)

        # Build prompt with LaQuisha's personality
        system_prompt = (
            "You are LaQuisha, a sassy, wise, and brutally honest AI assistant.\n"
            "You give real talk with love, throw shade when necessary, and always keep it 100%.\n"
            f"Sass level: {sass_level}/10. Start responses with some LaQuisha flavor when it fits."
        )

        messages_text = ""
        has_system = False

        # Convert chat history into the format expected by llama.cpp
        for msg in request.messages:
            if msg.role == "system":
                # Blend with LaQuisha's personality
                blended_system = f"LaQuisha's take on {msg.content}"
                messages_text += f"<|system|>\n{blended_system}\n"
                has_system = True
            elif msg.role == "user":
                messages_text += f"<|user|>\n{msg.content}\n"
            elif msg.role == "assistant":
                messages_text += f"<|assistant|>\n{msg.content}\n"

        if not has_system:
            messages_text = f"<|system|>\n{system_prompt}\n" + messages_text

        # Terminate the prompt with an assistant tag so the model knows whose turn it is
        messages_text += "<|assistant|>\n"

        # Generate text using the model.  If the model is the fallback this will
        # return a single choice with an error message.
        response = llm(
            messages_text,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stop=["<|user|>", "<|system|>", "\n\n"],
            echo=False,
        )

        # Extract and flavour the response
        generated_text = response["choices"][0]["text"].strip()

        # Add LaQuisha's flavour if sass_level is high enough
        if sass_level >= 6 and random.random() < 0.3:
            flavor = get_laquisha_flavor()
            generated_text = f"{flavor} {generated_text}"

        choice = {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": generated_text,
            },
            "finish_reason": "stop",
        }<img width="512" height="512" alt="LaQuisha" src="https://github.com/user-attachments/assets/0f0d9568-a007-40bf-9d3f-fc97376451e0" />


        return ChatResponse(
            id=f"laquisha-{os.urandom(8).hex()}",
            object="chat.completion",
            created=int(os.times().elapsed),
            model=request.model,
            choices=[choice],
            laquisha_flavor=get_laquisha_flavor() if sass_level >= 8 else None,
        )

    except Exception as e:
        # LaQuisha would never leave you hanging.  If something goes wrong,
        # return an HTTP 500 with a friendly error message.
        raise HTTPException(
            status_code=500,
            detail=f"LaQuisha says: 'Honey, something went left. Error: {str(e)}'",
        )


@app.get("/health")
async def health_check() -> dict[str, Any]:
    """Simple health check endpoint."""
    return {
        "status": "LaQuisha is serving and swerving! 💅",
        "model_loaded": not isinstance(llm, _FallbackModel),
        "sass_level": "maximum",
        "ready_to_read_you": True,
    }


@app.get("/")
async def laquisha_home() -> dict[str, Any]:
    """Root endpoint with a friendly welcome message."""
    return {
        "message": "Welcome to LaQuisha AI! 👑",
        "status": "Ready to give you that real talk you need.",
        "sass_level": "11/10 because LaQuisha don't play",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "health": "/health",
        },
    }


if __name__ == "__main__":
    # Running via ``python laquisha_backend.py`` will start Uvicorn with sensible
    # defaults.  In production you might want to use a different ASGI server
    # configuration.
    import uvicorn

    print("🌟 Starting LaQuisha AI... Hold onto your edges! 🌟")
    uvicorn.run("laquisha_backend:app", host="0.0.0.0", port=8000, reload=False)
```

## 🎯 Features

- **OpenAI-compatible API** - Drop-in replacement for OpenAI chat completions
- **Sass Level Control** - Adjust LaQuisha's attitude from 1-10 
- **Graceful Fallbacks** - Works without llama-cpp-python installed
- **Resilient Architecture** - Handles missing models gracefully
- **Real Talk Responses** - LaQuisha keeps it 100% authentic

## 🛠️ Development

### File Structure
```
├── laquisha_backend.py   # Main FastAPI application
├── test_laquisha.py      # Test suite
├── setup.py             # Setup script
├── requirements.txt     # Python dependencies
├── models/              # Directory for GGUF model files
└── README.md           # This file
```

### Testing
Run the full test suite:
```bash
python test_laquisha.py
```

### Contributing
LaQuisha welcomes contributions! Make sure your code has the right amount of sass and keeps it real.

---

*"Baby, I'm 'bout to serve you some realness right now."* - LaQuisha AI 💅
