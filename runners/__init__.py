REGISTRY = {}

from .episode_runner import EpisodeRunner
REGISTRY["episode"] = EpisodeRunner

from .parallel_runner import ParallelRunner
from .my_parallel_runner import MyParallelRunner
# REGISTRY["parallel"] = ParallelRunner
REGISTRY["parallel"] = MyParallelRunner
