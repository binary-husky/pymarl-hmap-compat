REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
from .efficient_parallel_runner import EfficientParallelRunner
REGISTRY["parallel"] = ParallelRunner
REGISTRY["efficient_parallel_runner"] = EfficientParallelRunner
