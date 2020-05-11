from typing import List
from typing import Optional
from typing import NamedTuple
from typing import Tuple


class Config(NamedTuple):
    trn_videos: List[str]
    vld_videos: List[str]
    experiment_name: str
    experiments_dir: str
    dataset_dir: str
    frames_folder: str
    maps_folder: str
    fixations_filename: str
    height: int = 160
    width: int = 320
    nss_weight: float = -0.1
    cc_weight: float = -0.1
    mse_weight: float = 0.0
    kl_weight: float = 1.0
    cc_is_spherical: bool = False
    kernel_sizes: List[Tuple[int, int]] = [(3, 3)]
    weight_decay: float = 1e-5
    use_cuda: bool = True
    num_workers: int = 1
    batch_size: int = 32
    use_nesterov: bool = True
    lr: float = 0.001
    momentum: float = 0.9
    warmup_steps: int = 1000
    num_epochs: int = 10
    pin_gpu_memory: bool = True
    dump_interval: int = 10
    log_interval: int = 10
    ckpt_interval: int = 10
    best_model_count: int = 10
    vld_every_epoch: int = 2
    norm_type: str = 'batch'
    num_downsamples: int = 5
    num_filters: int = 64
    frac: float = 1.0
    random_seed: Optional[int] = None
