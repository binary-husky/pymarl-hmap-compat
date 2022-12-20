from functools import partial


# from smac.env import MultiAgentEnv, StarCraft2Env
# from .one_step_matrix_game import OneStepMatrixGame
# from .particle import Particle
# from .stag_hunt import StagHunt
from .hmp_compat import HMP_compat   # hmp_compat

def env_fn(env, **kwargs):
    return env(**kwargs)

REGISTRY = {}
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
# REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)
# REGISTRY["particle"] = partial(env_fn, env=Particle)
# REGISTRY["stag_hunt"] = partial(env_fn, env=StagHunt)

REGISTRY["HMP_compat"] = partial(env_fn, env=HMP_compat)   # hmp_compat
