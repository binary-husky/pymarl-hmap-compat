REGISTRY = {}
from .basic_controller import BasicMAC
from .n_controller import NMAC
from .ppo_controller import PPOMAC
from .conv_controller import ConvMAC
from .basic_central_controller import CentralBasicMAC
from .lica_controller import LICAMAC
from .cqmix_controller import CQMixMAC

REGISTRY["basic_mac"] = BasicMAC
# REGISTRY["n_mac"] = NMAC
REGISTRY["ppo_mac"] = PPOMAC
REGISTRY["conv_mac"] = ConvMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["lica_mac"] = LICAMAC
REGISTRY["cqmix_mac"] = CQMixMAC

from .my_n_controller import my_NMAC
REGISTRY["n_mac"] = my_NMAC
