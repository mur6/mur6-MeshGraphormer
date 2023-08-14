import pathlib

from src.modeling._mano import MANO, Mesh

print("make mano")

mano_dir = pathlib.Path("src/modeling/data")
from manopth.manolayer import ManoLayer
