# docwain-gptoss-finetune-pipeline

## Minimal run commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

```bash
./scripts/gen_dataset.sh
```

```bash
./scripts/train_lora.sh
```

```bash
./scripts/package_ollama.sh
```

```bash
./scripts/run_eval.sh
pytest -q
```

```bash
# Or run the full pipeline:
python main.py all
```
