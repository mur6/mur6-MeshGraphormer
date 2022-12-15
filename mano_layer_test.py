import numpy as np
import torch

from src.modeling._mano import MANO
from manopth.manolayer import ManoLayer

# Mesh and SMPL utils
mano_model = MANO().to("cpu")
# mano_model.layer = mano_model.layer.cuda()

