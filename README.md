<img width="512" height="512" alt="LaQuisha" src="https://github.com/user-attachments/assets/b9b3b833-8e47-47c7-b707-5c597b5baa36" />

# LaQuisha - Fast API Backend for Llama.cpp

> *"When you need that real talk with some AI magic ✨"*

LaQuisha is a booty-licious fast API backend designed to run GGUF models with Llama.cpp. This FastAPI application provides an OpenAI-compatible chat API with LaQuisha's signature sass and personality.

## Features

- 🎯 **OpenAI-compatible API** - Drop-in replacement for OpenAI chat completions
- 🔥 **Sassy personality** - LaQuisha brings the attitude and real talk you need
- ⚡ **Fast performance** - Powered by llama.cpp for efficient inference
- 🛡️ **Resilient design** - Graceful fallback when models aren't available
- 🎚️ **Configurable sass** - Adjust LaQuisha's sass level from 1-10
- 🚀 **Easy deployment** - Simple setup with uvicorn

## Installation

### Prerequisites

- Python 3.8+
- A GGUF model file (e.g., LLaMA-3 7B)

### Install Dependencies

```bash
pip install fastapi uvicorn llama-cpp-python
```

### Download a Model

Place your GGUF model file in a `models/` directory:

```bash
mkdir models
# Download your preferred GGUF model to models/
```

## Quick Start

### Basic Usage

1. **Start the server:**

```bash
python laquisha_backend.py
```

2. **Test the health endpoint:**

```bash
curl http://localhost:8000/health
```

3. **Chat with LaQuisha:**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "laquisha-7b",
    "messages": [
      {"role": "user", "content": "Tell me about yourself"}
    ],
    "sass_level": 8
  }'
```

### Advanced Usage

**With custom model path:**

```bash
MODEL_PATH=/path/to/your/model.gguf python laquisha_backend.py
```

**Using uvicorn directly:**

```bash
MODEL_PATH=./models/llama-3-7b-chat.Q4_0.gguf uvicorn laquisha_backend:app --host 0.0.0.0 --port 8000
```

## Configuration

The application can be configured through environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_PATH` | `./models/llama-3-7b-chat.Q4_0.gguf` | Path to your GGUF model file |
| `MAX_TOKENS` | `512` | Maximum tokens to generate per request |
| `TEMPERATURE` | `0.8` | Model temperature (0.0-1.0) |

### Configuration in Code

You can also modify the configuration directly in `laquisha_backend.py`:

```python
# Maximum number of tokens to generate per request
MAX_TOKENS = 512

# Temperature controls randomness (higher = more varied responses)
TEMPERATURE = 0.8

# LaQuisha's signature phrases
LAQUISHA_QUOTES = [
    "Chile, let me tell you something...",
    "Honey, hold up—let me break this down for you.",
    # Add your own custom quotes here
]
```

## API Reference

### Chat Completions

**Endpoint:** `POST /v1/chat/completions`

**Request Body:**

```json
{
  "model": "laquisha-7b",
  "messages": [
    {"role": "user", "content": "Your message here"}
  ],
  "temperature": 0.8,
  "max_tokens": 512,
  "sass_level": 7
}
```

**Special Parameters:**

- `sass_level` (1-10): Controls how much attitude LaQuisha brings to responses
  - 1-5: Professional and helpful
  - 6-7: Some personality showing through
  - 8-10: Full LaQuisha experience with flavor and sass

### Health Check

**Endpoint:** `GET /health`

Returns server status and model availability.

### Home

**Endpoint:** `GET /`

Welcome message with available endpoints.

## Model Compatibility

LaQuisha works with any GGUF model compatible with llama.cpp, including:

- LLaMA 2/3 models
- Code Llama
- Mistral
- And many others

## Error Handling

LaQuisha handles errors gracefully:

- **No model loaded**: Returns helpful error messages in LaQuisha's voice
- **Model loading failures**: Falls back to informative responses
- **Runtime errors**: Provides sass-filled error messages

## Development

### Project Structure

```
LaQuisha-fast-API-backend-for-Llama.cpp/
├── README.md                 # This file
├── laquisha_backend.py      # Main FastAPI application
└── models/                  # Directory for GGUF model files
```

### Customization

**Adding Custom Quotes:**

Edit the `LAQUISHA_QUOTES` list in `laquisha_backend.py`:

```python
LAQUISHA_QUOTES = [
    "Chile, let me tell you something...",
    "Your custom quote here...",
]
```

**Adjusting Model Parameters:**

Modify the model initialization in `laquisha_backend.py`:

```python
llm = llama_cpp.Llama(
    model_path=MODEL_PATH,
    n_ctx=2048,        # Context length
    n_threads=4,       # Number of threads
    verbose=False,
)
```

## Contributing

Feel free to submit issues, feature requests, or pull requests. LaQuisha appreciates good code with proper attitude! 💅

## License

This project is open source. Check the repository for license details.

---

*"Baby, I'm 'bout to serve you some realness right now." - LaQuisha* 👑
