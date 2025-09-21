"""
LaQuisha, a booty-licious fast API backend designed to run GGUF models with Llama.cpp

This module exposes a small FastAPI application that wraps a llama.cpp model.  It's
designed to run a local LLaMA models (in this case the LLaMA‑3 7B model) using the
`llama_cpp` python bindings.  The code is resilient to environments where the
`llama_cpp` module is not available: in that case it will initialise a
placeholder model that returns a friendly error message rather than failing to
import.  To run the server you will need a GGUF model file and the
``llama_cpp`` library installed.  See the README accompanying this file for
details.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Any
import os
import random
import time

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
#     MODEL_PATH=/path/to/llama-3-7b-chat.Q4_0.gguf uvicorn laquisha_backend:app
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

# Mount a static files directory for serving assets like the logo.  The static
# folder must live alongside this backend file.  Files placed in ``static``
# will be available at ``/static/...``.
app.mount("/static", StaticFiles(directory="static"), name="static")


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

# ----------------------------------------------------------------------------
# Model upload endpoint
# ----------------------------------------------------------------------------
# This endpoint allows a new GGUF model to be uploaded via the web UI.  When
# called, it writes the uploaded file into the ``models`` directory and then
# reloads the global ``llm`` instance to point at the new model.  If llama_cpp
# isn't available or the model fails to load, the API will continue using the
# fallback model and return an error message to the caller.

@app.post("/upload_model")
async def upload_model(file: UploadFile = File(...)) -> dict[str, Any]:
    """Upload a new GGUF model and reload LaQuisha's brain.

    The uploaded file is stored in the ``models`` folder relative to this
    script.  After saving, the global ``llm`` is reinitialised with the new
    model.  The filename is not validated beyond ensuring it has a .gguf
    extension; callers should provide valid GGUF model files.  On success
    returns a message indicating the new model has been loaded.  On failure
    returns a message explaining what went wrong.
    """
    # Ensure the models directory exists
    models_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(models_dir, exist_ok=True)
    # Save the uploaded file
    filename = os.path.basename(file.filename)
    save_path = os.path.join(models_dir, filename)
    try:
        with open(save_path, "wb") as f:
            while contents := await file.read(1024 * 1024):
                f.write(contents)
    except Exception as e:
        return {"success": False, "message": f"Failed to save model: {e}"}

    # Attempt to reload the model.  Use a global to update the existing
    # instance so all incoming requests use the new model.
    global llm
    if llama_cpp is None:
        llm = _FallbackModel()  # type: ignore
        return {"success": False, "message": "llama_cpp is not installed; using fallback model."}
    try:
        llm = llama_cpp.Llama(
            model_path=save_path,
            n_ctx=2048,
            n_threads=4,
            verbose=False,
        )
        return {"success": True, "message": f"Model '{filename}' uploaded and loaded successfully."}
    except Exception as exc:
        # If loading fails, fall back to the placeholder model
        llm = _FallbackModel()  # type: ignore
        return {"success": False, "message": f"Failed to load model: {exc}"}


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
        }

        return ChatResponse(
            id=f"laquisha-{os.urandom(8).hex()}",
            object="chat.completion",
            created=int(time.time()),
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


# -----------------------------------------------------------------------------
# Frontend integration
# -----------------------------------------------------------------------------
# Serve the index.html file at the root URL ("/") so that visiting the root
# of the server returns the chat interface.  The HTML file should be located
# alongside this backend in the same directory.

@app.get("/", response_class=HTMLResponse)
async def get_chat_ui() -> HTMLResponse:
    """Serve the index.html file at the root URL ("/").

    This function reads the contents of ``index.html`` from the current
    working directory and returns it as an HTML response.  If the file
    cannot be found, an HTTP 500 error is raised.
    """
    try:
        with open("index.html", "r") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="index.html file not found. Please make sure the frontend file is present in the same directory.",
        )


if __name__ == "__main__":
    # Running via ``python laquisha_backend.py`` will start Uvicorn with sensible
    # defaults.  In production you might want to use a different ASGI server
    # configuration.
    import uvicorn

    print("🌟 Starting LaQuisha AI... Hold onto your edges! 🌟")
    uvicorn.run("laquisha_backend:app", host="0.0.0.0", port=8001, reload=False)
