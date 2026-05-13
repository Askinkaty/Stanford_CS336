from config import Config, ModelConfig, DataConfig, TrainingConfig, OptimizerConfig


cfg = Config(
    data=DataConfig(
        Config.data.train_path,
        Config.data.val_path,
        batch_size=96,
        val_batch_size=192,
        context_length=384,
    ),
    # layer params = 5.5M, total non-emb=24.5M + 5.5*8 = 68.5M
    model=ModelConfig(
        vocab_size=10000,
        d_model=768,
        dim_feedforward=2048,
        num_layers=8,
        n_head=12,
    ),
    optimizer=OptimizerConfig(learning_rate=7e-3, min_lr=1e-2, weight_decay=0.0, betas=(0.95, 0.99)),
    training=TrainingConfig(
        log_interval=50, save_interval=4000, val_interval=3000, max_steps=24000, gradient_accumulation_steps=2, dtype="bfloat16"
    ),
    project=Config.project,
)

default_cfg = Config(DataConfig(), ModelConfig(), OptimConfig(), TrainerConfig())