import torch.nn as nn
import torch
import math
from reid.models.CLIP_ReID.model.make_model_clipreid import make_clipreid
import torch.nn.functional as F

import os.path as osp


class CLIP_Backbone(nn.Module):
    def __init__(self, num_class, camera_num, view_num=1):
        super(CLIP_Backbone, self).__init__()
        self.config_file = 'reid/models/CLIP_ReID/configs/person/vit_clipreid.yml' 
        from reid.models.CLIP_ReID.config import cfg
        cfg.merge_from_file(self.config_file)
        cfg.freeze()
        self.base = make_clipreid(cfg=cfg, num_class=num_class,camera_num=camera_num, view_num=view_num)
        print('Using ViT-based CLIP_ReID as the backbone...')
        self.num_classes = num_class

    def forward(self, x = None, label=None, get_image = False, get_text = False, requires_region_attn=False):
        got = self.base(x, label, get_image, get_text, requires_region_attn)
        return got

def make_model(num_class, camera_num, view_num=1):
    model = CLIP_Backbone(num_class=num_class, camera_num=camera_num, view_num=view_num)
    print('ViT based CLIP_ReID base model built...')
    return model

    



