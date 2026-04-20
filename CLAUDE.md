# spark-vllm-docker

## Workflow

### Launching a model

Use the `run-vllm` alias (defined in `~/.bashrc`):

```bash
run-vllm <recipe-name>          # start a model
run-vllm -l                     # list available personal recipes
```

`run-vllm` expands to `cd /home/alantkt/Documents/spark-vllm-docker && ./run-my-recipe.sh`.

`run-my-recipe.sh` mounts `/opt/llms/models` into the container and delegates to `run-recipe.sh <my-recipes/<name>.yaml> --solo`.

### Key folders

| Path | Purpose |
|------|---------|
| `my-recipes/` | Personal model recipes (YAML) |
| `recipes/` | Upstream/shared recipes |
| `mods/` | Container patches applied at build time |
| `/opt/llms/models/` | Local model weights (outside repo) |

---

## my-recipes conventions

- All personal recipes live in `my-recipes/` as `<model-name>.yaml`
- **Thinking mode must be ON by default** — every recipe must include `--reasoning-parser <parser>` and must NOT set `enable_thinking: false` in `--chat-template-kwargs`
- Models on DGX Spark: prefer MoE or small dense models — LPDDR5X bandwidth limits dense 40B+ to ~5 t/s

---

## Testing a model
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4-deckard",
    "messages": [{"role": "user", "content": "what is 2*3?"}],
    "max_tokens": 4096,
    "chat_template_kwargs": {"enable_thinking": false}  # remove this line to enable thinking
  }'
```

### Check available models
```bash
curl http://localhost:8000/v1/models
```
