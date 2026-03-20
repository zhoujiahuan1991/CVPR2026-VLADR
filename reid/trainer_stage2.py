from __future__ import print_function, absolute_import
import time
from torch.nn import functional as F
import torch
import torch.nn as nn
import os.path as osp
from .utils.meters import AverageMeter
from .utils.feature_tools import *
from reid.models.BLIP_gen import *
from reid.models.wrapper import CLIP_Backbone
from reid.utils.make_loss import make_loss
from tqdm import tqdm


class Stage2Trainer(object):
    def __init__(self,cfg,args, model:CLIP_Backbone, num_classes, text_feat_pool:Text_feat_Pool, writer=None):
        super(Stage2Trainer, self).__init__()
        self.cfg = cfg
        self.args = args
        self.model = model
        self.writer = writer
        self.num_classes = num_classes
        self.text_feat_pool = text_feat_pool

        self.part_names = getattr(self.text_feat_pool, 'part_names', ['head', 'upper', 'lower', 'foot']) 
        self.active_part_indices = self._part_ablation()

        self.AF_weight = 1
        self._last_stage = None  
        self.af_temp = 0.005
        self.af_eps = 1e-8

        self.loss_fn, self.center_criterion = make_loss(cfg, num_classes=num_classes)
        self.loss_ce = nn.CrossEntropyLoss(reduction='batchmean')
        self.KLDivLoss = nn.KLDivLoss(reduction="batchmean")

    def _part_ablation(self):
        pool_indices = getattr(self.text_feat_pool, 'active_part_indices', None)
        if pool_indices is not None:
            return list(pool_indices)

        args = self.args
        ablation_part = args.ablation_part

        mapping = {
            'all': list(range(len(self.part_names))),
            'head': [0],
            'upper': [1],
            'lower': [2],
            'foot': [3],
        }
        return mapping.get(ablation_part)

    def train(self, args, clip_stage2_loader, optimizer_stage2,train_iters=200, add_num=0, old_model=None,lr_scheduler_2=None):
        
        self.model.train()
        base = self.model.module.base 

        for m in base.modules():
            if isinstance(m, nn.BatchNorm2d):
                if not m.weight.requires_grad and not m.bias.requires_grad:
                    m.eval()

        data_time = AverageMeter()
        losses_AF = AverageMeter()
        
        end = time.time()
        device = next(self.model.parameters()).device

        active_indices = self._part_ablation()
        
        for iter in tqdm(range(train_iters), desc='Training'):
            start_time=time.time()
            data = clip_stage2_loader.next()


            img_transformed, _, fpaths, targets, camids, img_strong_augmented = data
            img_transformed = img_transformed.to('cuda')
            img_strong_augmented = img_strong_augmented.to('cuda')
            targets = targets.to('cuda')

            fnames = self.parser_fpaths(fpaths)
            data_time.update(time.time() - end)
            original_targets = targets
            shifted_targets = targets + add_num
            targets = shifted_targets

            got_cls, img_feats, img_feat_proj, region_attn_proj = self.model(img_transformed, targets) 
            cls_score = got_cls[0]

            got_cls_strong, img_feats_strong, img_feat_proj_strong, region_attn_proj_strong = self.model(img_strong_augmented, targets) # strong augmentation
            cls_score_strong = got_cls_strong[0]

            divergence = torch.tensor(0.0, device=device)

            if active_indices:
                region_attn_proj = region_attn_proj[:, active_indices, :]
                region_attn_proj_strong = region_attn_proj_strong[:, active_indices, :]

            if old_model is not None:
                    with torch.no_grad():
                        old_img_feat, old_region_attn_proj = old_model(img_transformed, requires_region_attn=True)
                    old_region_attn_proj = old_region_attn_proj[:, active_indices, :]
                    affinity_new = self.get_normal_affinity(region_attn_proj)
                    affinity_old = self.get_normal_affinity(old_region_attn_proj)

                    divergence_raw = self.cal_KL_old(affinity_new, affinity_old)
                    divergence = torch.clamp(divergence_raw, 0.0)  * args.AF_beta
                    losses_AF.update(divergence.detach().item())

            region_norm = region_attn_proj 

            if args.ablation_part != 'none':
                text_feats = self.text_feat_pool.grab(fnames, iter)

                text_norm = F.normalize(text_feats, dim=-1) if text_feats.numel() > 0 else text_feats
            #print(text_norm.shape)
            #exit() 

            batch_size, num_parts, _ = region_norm.shape 
            labels = torch.arange(batch_size, device=device) 

            losses = []

            for part in range(num_parts):
                    img_vec = region_norm[:, part, :] 
                    text_vec = text_norm[:, part, :] 

                    logits_i2t = img_vec @ text_vec.t()
                    logits_t2i = text_vec @ img_vec.t()

                    loss_part = 0.5 * (F.cross_entropy(logits_i2t, labels) + F.cross_entropy(logits_t2i, labels))
                    losses.append(loss_part)
            loss_img_txt_fine = torch.stack(losses).mean()

            with torch.no_grad():
                    unique_labels_global, inverse_idx = torch.unique(targets, sorted=True, return_inverse=True)
                    text_bank = base(label=unique_labels_global, get_text=True)
                    text_bank = F.normalize(text_bank, dim=1) 

            image_features_b2 = base(x=img_transformed, get_image=True)
            image_features_b2 = F.normalize(image_features_b2, dim=1) 

            logits = image_features_b2 @ text_bank.t()
            loss_image_text_coarse = F.cross_entropy(logits, inverse_idx)

            loss_ce, loss_tp = self.loss_fn(cls_score, img_feat_proj, targets, target_cam=None)
            loss_ce_strong, loss_tp_strong = self.loss_fn(cls_score_strong, img_feat_proj_strong, targets, target_cam=None)

            loss_stage2 = loss_ce + loss_tp + loss_ce_strong +\
                loss_tp_strong + loss_image_text_coarse + args.fine_weight * loss_img_txt_fine +divergence

            optimizer_stage2.zero_grad()
            loss_stage2.backward() 
            optimizer_stage2.step()

            lr_scheduler_2.step()

            if iter % 30 == 0 or iter == train_iters-1:
                print('\n[stage2] loss_ce: {}'.format(loss_ce+loss_ce_strong))
                print('[stage2] loss_tp: {}'.format(loss_tp+loss_tp_strong))
                print('[stage2] loss_image_text_fine: {}'.format(loss_img_txt_fine))
                print('[stage2] loss_image_text_coarse: {}'.format(loss_image_text_coarse))
                print('[stage2] query AF: {}, divergence: {}'.format(args.query_AF,float(divergence.detach())))


    def parser_fpaths(self, fpaths) -> list:
        fnames = []
        for path in fpaths:
            fname = osp.basename(path)
            fnames.append(fname)

        return fnames

    def get_normal_affinity(self, x, temperature=None):
        temperature = self.af_temp if temperature is None else temperature
        temperature = max(float(temperature), 1e-6)
        pre_matrix_origin = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=-1)
        pre_affinity_matrix = F.softmax(pre_matrix_origin / temperature, dim=1)
        return pre_affinity_matrix

    def cal_KL_old(self, affinity_new, affinity_old):
        target = affinity_old.detach()
        affinity_new_log = torch.log(affinity_new.clamp_min(self.af_eps))
        divergence = self.KLDivLoss(affinity_new_log, target)
        return divergence
        
