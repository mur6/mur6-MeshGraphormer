import json
import pathlib

import numpy as np

from src.modeling._mano import MANO, Mesh


def save(d, filename):
    p = pathlib.Path(filename)
    with p.open(mode='w') as fh:
        json.dump(d, fh)

def main():
    mano_model = MANO()
    #mano_model.layer = mano_model.layer.cuda()
    #mesh_sampler = Mesh()
    faces = mano_model.face
    print(faces.shape, faces.dtype)
    data = faces.tolist()
    print(type(faces))
    np.save("faces", faces)
    #save(data, "faces.json")


main()
