# NVIDIA MCP Server

The NVIDIA Resource Suite includes a Model Context Protocol (MCP) server that
exposes NVIDIA capabilities as tools inside Claude and Claude Code.

## Available Tools

| Tool | Description |
|------|-------------|
| `nvidia_chat` | Text generation via NIM LLMs |
| `nvidia_embed` | Text embeddings via NIM |
| `nvidia_vision` | Image analysis via NIM vision models |
| `nvidia_gpu_status` | GPU job queue status |
| `nvidia_submit_job` | Submit GPU compute job |
| `nvidia_list_models` | List available NIM models |
| `nvidia_education_tutor` | AI tutoring for World Interactive Origins |

## Installation (Claude Code)

Add to `~/.claude/settings.json`:

```json
{
  "mcpServers": {
    "nvidia-resource-suite": {
      "command": "python",
      "args": ["-m", "mcp_server.nvidia_mcp_server"],
      "cwd": "/path/to/nvidia-resource-suite",
      "env": {"NVIDIA_API_KEY": "nvapi-YOUR_KEY"}
    }
  }
}
```

Restart Claude Code. The NVIDIA tools will appear.

## Example Usage (in Claude)

**Generate educational content:**
```
Use nvidia_chat to write a dialogue between a student and an ancient Silk Road merchant
```

**Analyze an artifact image:**
```
Use nvidia_vision to analyze this image of a Tang Dynasty coin and explain what it tells us
```

**Submit a rendering job:**
```
Use nvidia_submit_job to queue a rendering job for the Harlem Renaissance world scene
```

## Models Reference

### Chat Models
```
meta/llama-3.1-70b-instruct        — Best overall quality
nvidia/llama-3.1-nemotron-70b      — NVIDIA-tuned, follows instructions precisely
microsoft/phi-3-medium-128k-instruct — Long context (128k tokens)
mistralai/mixtral-8x7b-instruct    — Fast, good for streaming
```

### Embedding Models
```
nvidia/nv-embedqa-e5-v5            — Best for Q&A and RAG
snowflake/arctic-embed-l           — High-quality general embeddings
```

### Vision Models
```
microsoft/phi-3-vision-128k-instruct — Best vision+language
nvidia/neva-22b                     — NVIDIA vision-language model
```
