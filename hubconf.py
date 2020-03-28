from PIL import Image
from torch import nn
import numpy as np
dependencies = ['torch']
import torch

def covid19(pretrained=True, **kwargs):
    from covid19 import create_model
    model=create_model()
    if pretrained:
        #checkpoint_uri="https://github.com/haruiz/COVID19-Xray/blob/master/covid19-1e76d47b.pt?raw=true"
        checkpoint_uri = "https://modelshub.blob.core.windows.net/models/covid19-1e76d47b.pt"
        checkpoint=torch.hub.load_state_dict_from_url(checkpoint_uri,progress=False,map_location='cpu')
        model.load_state_dict(checkpoint)
    return model








