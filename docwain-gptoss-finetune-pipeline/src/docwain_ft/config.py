import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass
class Config:
    base_model: str = os.getenv("BASE_MODEL", "gpt-oss")
    hf_base_model: str = os.getenv("HF_BASE_MODEL", "openai/gpt-oss-20b")
    output_dir: str = os.getenv("OUTPUT_DIR", "out/")
    dataset_size: int = int(os.getenv("DATASET_SIZE", "800"))
    evalset_size: int = int(os.getenv("EVALSET_SIZE", "200"))
    max_seq_len: int = int(os.getenv("MAX_SEQ_LEN", "4096"))
    num_epochs: int = int(os.getenv("NUM_EPOCHS", "2"))
    lr: float = float(os.getenv("LR", "2e-4"))
    batch_size: int = int(os.getenv("BATCH_SIZE", "1"))
    grad_accum: int = int(os.getenv("GRAD_ACCUM", "8"))
    use_qlora: bool = os.getenv("USE_QLORA", "1") == "1"
    seed: int = int(os.getenv("SEED", "42"))
    ollama_base: str = os.getenv("OLLAMA_BASE", "gpt-oss:20b")
    ollama_model_name: str = os.getenv("OLLAMA_MODEL_NAME", "DocWain-Agent")
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    top_p: float = float(os.getenv("TOP_P", "0.85"))
    repeat_penalty: float = float(os.getenv("REPEAT_PENALTY", "1.1"))
    num_ctx: int = int(os.getenv("NUM_CTX", "8192"))


CONFIG = Config()
