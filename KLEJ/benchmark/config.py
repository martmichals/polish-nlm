import typing as t
from pydantic import BaseModel, conint, confloat

class Config(BaseModel):
    # Task
    task_name: t.Optional[str] = None
    """Name of the task model is being trained for"""
    run_id: t.Optional[str] = None
    """Unique ID for the run"""

    # Paths
    task_path: t.Optional[str] = None
    """Path to the task dataset"""
    predict_path: t.Optional[str] = None
    """Path to store predictions for the test set"""
    logger_path: t.Optional[str] = None
    """TODO: Define this well"""
    checkpoint_path: t.Optional[str] = None
    """Path to store saved model"""

    # Tokenizer
    tokenizer_name_or_path: t.Optional[str] = None
    """Name or path for the tokenizer"""
    max_seq_length: conint(gt=0) = 256
    """Maximum length of the input sequence to the model"""
    do_lower_case: bool = False
    """If using an uncased model, set this flag to true, tokenizer parameter"""

    # Model
    nlm_name_or_path: t.Optional[str] = None
    """Name or path to model data"""
    learning_rate: confloat(gt=0) = 2e-5
    """Initial learning rate"""
    adam_epsilon: confloat(gt=0) = 1e-8
    """Epsilon parameter for the ADAM optimizer"""
    warmup_steps: conint(gt=0) = 100
    """Number of lr warmup steps"""
    batch_size: conint(gt=0, multiple_of=2) = 16
    """Number of samples in a batch"""
    gradient_accumulation_steps: conint(gt=0) = 2
    """Number of batches for which to accumulate gradients"""
    num_train_epochs: conint(gt=0) = 4
    """Training epochs"""
    weight_decay: confloat(ge=0.0) = 0.0
    """Value of weight decay"""
    max_grad_norm: confloat(gt=0) = 1.0
    """Maximal value for the gradient norm"""

    # Reproducibility
    seed: int = 42
    """Seed for reproducibility"""

    # Compute
    num_workers: conint(ge=0) = 4
    """Number of processes to use for the DataLoader. 0 for main process."""
    num_gpu: conint(ge=0) = 1
    """Number of GPUs to use. Set to 0 for CPU-only computation."""
