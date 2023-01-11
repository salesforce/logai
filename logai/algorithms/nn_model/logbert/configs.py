from attr import dataclass
from logai.config_interfaces import Config


@dataclass
class LogBERTConfig(Config):
    """Config for logBERT model

    Inherits:
        Config: config interface
    """

    pretrain_from_scratch: bool = True
    model_name: str = "bert-base-cased"
    model_dirname: str = None 
    mlm_probability: float = 0.15
    mask_ngram: int = 1
    max_token_len: int = 384
    evaluation_strategy: str = "steps"
    num_train_epochs: int = 20
    learning_rate: float = 1e-5
    logging_steps: int = 10
    per_device_train_batch_size: int = 50
    per_device_eval_batch_size: int = 256
    eval_accumulation_steps: int = 1000
    num_eval_shards: int = 10
    weight_decay: float = 0.0001
    save_steps: int = 50
    eval_steps: int = 50
    resume_from_checkpoint: bool = True
    output_dir: str = None
    tokenizer_dirpath: str = None
