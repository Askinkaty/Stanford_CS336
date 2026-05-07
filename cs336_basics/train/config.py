from dataclasses import dataclass
from typing import Literal
from pathlib import Path

_DATA_DIR = Path(__file__).parent.parent / "data_tokenized"


@dataclass(frozen=True)
class DataConfig:
    train_path: str | Path = str(_DATA_DIR / "TinyStoriesV2-GPT4-train.npy")
    val_path: str | Path = str(_DATA_DIR / "TinyStoriesV2-GPT4-valid.npy")
    test_path: str | Path = ""
    batch_size: int = 4
    val_batch_size: int = 4
    context_length: int = 256


@dataclass(frozen=True)
class OptimizerConfig:
    optimizer_type: Literal["adam", "sgd", "adamw"] = "adamw"
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    betas: tuple = (0.9, 0.999)
    min_lr: float = 0.0
    scheduler_type: Literal["cosine", "linear", "constant"] = "cosine"
    num_warmup_steps: int = 1000
    cosine_steps: int = 10000


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 512
    n_head: int = 16
    num_layers: int = 4
    dim_feedforward: int = 1344
    vocab_size: int = 10000
    theta: float = 10000.0



@dataclass(frozen=True)
class TrainingConfig:
    gradient_accumulation_steps: int = 1
    run_name: str = "{date}"  # template
    output_dir: str | Path = "./output"
    load_from: str | Path | None = None
    num_epochs: int = 10
    seed: int = 42
    device : str = "cpu"
    dtype: Literal["float32", "float16", "bfloat16"] = "float32"
    max_steps: int = 100000
    log_interval: int = 100
    save_interval: int = 100
    val_interval: int = 500
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class Config:
    data: DataConfig = DataConfig()
    optimizer: OptimizerConfig = OptimizerConfig()
    model: ModelConfig = ModelConfig()
    trainer: TrainingConfig = TrainingConfig()
    project: str = "cs336_basics_project"


default_config = Config()




