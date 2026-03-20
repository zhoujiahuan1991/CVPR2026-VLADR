
from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys
import numpy as np
from reid.models.BLIP_gen import Arbitrary_text_encoder,Text_feat_Pool
import random
from config import cfg
from reid.utils.logging import Logger
from reid.utils.serialization import load_checkpoint, save_checkpoint, copy_state_dict
from reid.utils.lr_scheduler import WarmupMultiStepLR
from reid.utils.feature_tools import *
from reid.models.layers import DataParallel
from tqdm import tqdm
from reid.models.wrapper import make_model
from reid.trainer_stage2 import Stage2Trainer
from lreid_dataset.datasets.get_data_loaders import build_data_loaders
from tools.Logger_results import Logger_res
from reid.evaluation.fast_test import fast_test_p_s
from tqdm import tqdm
import datetime
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
warnings.filterwarnings(action='ignore', category=FutureWarning)

from torch.utils.benchmark import Timer

def part_indices(args):
    part_names= ['head', 'upper', 'lower', 'foot']
    name_to_idx = {name: idx for idx, name in enumerate(part_names)}

    ablation = args.ablation_part
    if ablation == 'all':
        return list(range(len(part_names)))
    if ablation in name_to_idx:
        return [name_to_idx[ablation]] # [1] <=> upper
    if ablation == 'none':
        return None

def cur_timestamp_str():
    now = datetime.datetime.now()
    year = str(now.year)
    month = str(now.month).zfill(2)
    day = str(now.day).zfill(2)
    hour = str(now.hour).zfill(2)
    minute = str(now.minute).zfill(2)
    content = "{}-{}{}-{}{}".format(year, month, day, hour, minute)
    return content


def main():
    args = parser.parse_args()
    if args.seed is not None:
        print("setting the seed to",args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    cfg.merge_from_file(args.config_file)
    main_worker(args, cfg)


def main_worker(args, cfg):
    timestamp = cur_timestamp_str()
    if args.other_details is not None:
        log_name = f'log_{timestamp}_{args.other_details}.txt'
    else:
        log_name = f'log_{timestamp}.txt'

    if args.testing is not None:
        stage2_dir = osp.join(args.logs_dir, 'evaluation', timestamp)
    else:
        stage2_dir = osp.join(args.logs_dir, 'stage2', timestamp)
        
    if not args.evaluate:
        sys.stdout = Logger(osp.join(stage2_dir,log_name))
    else:
        log_dir = osp.dirname(args.test_folder)
        sys.stdout = Logger(osp.join(log_dir, 'stage2', log_name))

    if args.other_details is not None:
        log_res_name=f'log_res_{timestamp}_{args.other_details}.txt'
    else:
        log_res_name = f'log_res{timestamp}.txt'

    logger_res=Logger_res(osp.join(stage2_dir,log_res_name)) 
    
    if 1 == args.setting:
            training_set = ['market1501', 'cuhk_sysu', 'lpw', 'msmt17', 'cuhk03']
    elif 2 == args.setting:
            training_set = ['lpw', 'msmt17', 'market1501', 'cuhk_sysu', 'cuhk03']
    elif 51 == args.setting:
            training_set = ['msmt17', 'cuhk_sysu', 'lpw', 'market1501', 'cuhk03']
    elif 52 == args.setting:
            training_set = ['lpw', 'market1501', 'cuhk03', 'msmt17', 'cuhk_sysu']
    elif 53 == args.setting:
            training_set = ['cuhk_sysu', 'lpw', 'cuhk03', 'msmt17', 'market1501']
    elif 54 == args.setting:
            training_set = ['cuhk03', 'msmt17', 'lpw', 'market1501', 'cuhk_sysu']
    elif 55 == args.setting:
            training_set = ['market1501', 'msmt17', 'lpw', 'cuhk_sysu', 'cuhk03']
    all_set = ['market1501', 'lpw', 'msmt17', 'cuhk_sysu', 'cuhk03',
               'cuhk01', 'cuhk02', 'grid', 'sense', 'viper', 'ilids', 'prid']  # 'sense','prid'

    testing_only_set = [x for x in all_set if x not in training_set]
    all_train_sets, all_test_only_sets = build_data_loaders(args, training_set, testing_only_set)    

    dataset, num_classes, clip_stage1_loader, clip_stage2_loader, BLIP_loader, test_loader, init_loader, name = all_train_sets[0]
    model = make_model(num_class=num_classes, camera_num=0, view_num=0)
    
    model.to('cuda')
    model = DataParallel(model)

    prompt_path = osp.join(args.stage1_prompts_out_dir, f'{name}_clipreid_prompt.pth')

    prompt = load_prompt(prompt_path)

    prompt_initialize(model.module.base, prompt)
    print('--> model prompt initialized using {}...'.format(prompt_path))
    _set_stage2_mode(model=model, args=args)

    model_current = model
    writer = None

    if args.testing is not None:
        ckpt_names = [x + '_checkpoint.pth.tar' for x in training_set]
        ckpt = load_checkpoint(osp.join(args.testing, ckpt_names[0]))
        copy_state_dict(ckpt['state_dict'], model)

        for step in range(len(ckpt_names) - 1):
            model_old = copy.deepcopy(model)
            checkpoint = load_checkpoint(osp.join(args.testing, ckpt_names[step + 1]))
            copy_state_dict(checkpoint['state_dict'], model)

            best_alpha = 0.6
            model = linear_combination(args, model, model_old, best_alpha)
            print(f'checkpoint {ckpt_names[step]} and checkpoint {ckpt_names[step+1]} fused with alpha {best_alpha}...')
            print(f'begin testing...')
        fast_test_p_s(model, all_train_sets, all_test_only_sets, len(all_train_sets)-1, args, logger=logger_res, writer=writer)
        print('testing finished...')
        exit(0)
        
    for set_index, train_info in enumerate(all_train_sets):
        dataset, num_classes, _, clip_stage2_loader, _, _, init_loader, name = train_info

        print('=' * 80)
        print(f"[Stage2] Training domain {set_index + 1}/{len(all_train_sets)}: {name}")
        print('=' * 80)
        if set_index > 0:
            old_model = copy.deepcopy(model_current)
            old_model = old_model.cuda()
            old_model.eval()
        else:
            old_model = None

        add_num = sum([all_train_sets[i][1] for i in range(set_index)]) if set_index > 0 else 0
        total_classes = add_num + num_classes

        base = model_current.module.base 

        if set_index > 0:
            base.classifier = expand_linear_head(base.classifier, total_classes)
            model_current.module.classifier = base.classifier
            base.classifier_proj = expand_linear_head(base.classifier_proj, total_classes)
            model_current.module.classifier_proj = base.classifier_proj
            prompt_path = osp.join(args.stage1_prompts_out_dir, f"{name}_clipreid_prompt.pth")
            prompt = load_prompt(prompt_path)
            expand_prompt_with_stage1(base, prompt, add_num, total_classes)
            print('--> prompt successfully expanded using {}'.format(prompt_path))
            base.num_classes = total_classes

            model_current.module.num_classes = total_classes

            class_centers = initial_classifier(model_current, init_loader)
            base.classifier.weight.data[add_num:].copy_(class_centers)
        else:
            class_centers = initial_classifier(model_current, init_loader)
            base.classifier.weight.data.copy_(class_centers)

        _set_stage2_mode(model=model_current, args=args) 
        text_encoder = Arbitrary_text_encoder()
        text_desc_path = osp.join(args.BLIP_text_out_dir, f'{name}_description.txt')

        active_part_indices = part_indices(args)

        text_feature_pool = Text_feat_Pool(text_desc_path, text_encoder, active_part_indices=active_part_indices)
        optimizer_stage2 = make_optimizer_stage2(model_current.module.base, args=args)

        Stones = args.milestones
        lr_scheduler_2 = WarmupMultiStepLR(optimizer_stage2, Stones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

        trainer =Stage2Trainer(cfg=cfg, args=args, model=model_current, num_classes=total_classes, text_feat_pool=text_feature_pool)

        Epochs = args.epochs
        for epoch in tqdm(range(Epochs), desc='[Stage2-Training-{}]'.format(name)):
            clip_stage2_loader.new_epoch()

            trainer.train(args, clip_stage2_loader, optimizer_stage2, len(clip_stage2_loader), add_num, old_model=old_model, lr_scheduler_2=lr_scheduler_2)

            if ((epoch + 1) % args.eval_epoch == 0 or epoch+1==Epochs):
                logger_res.append('epoch: {}'.format(epoch + 1))
                timestamp = cur_timestamp_str()
                mAP=0.
                save_checkpoint({
                    'state_dict': model_current.state_dict(),
                    'epoch': epoch + 1,
                    'mAP': mAP,
                }, True, fpath=osp.join(stage2_dir, '_CKPTS', '{}_checkpoint.pth.tar'.format(name)))    


        if set_index>0:
            best_alpha=args.best_alpha
            model_current = linear_combination(args, model_current, old_model, best_alpha)          
            eval_stage = (args.eval_stage).split(',')    
            if str(set_index) in eval_stage: 
                fast_test_p_s(model_current, all_train_sets, all_test_only_sets, set_index=set_index, logger=logger_res,
                            args=args,writer=writer)
            

    
    print('='*80)
    print('stage2 over, everything done...')
    print('='*80)



def linear_combination(args, model, model_old, alpha, model_old_id=-1):
    print("*******combining the models with alpha: {}*******".format(alpha))
    '''old model '''
    model_old_state_dict = model_old.state_dict()
    '''latest trained model'''
    model_state_dict = model.state_dict()

    ''''create new model'''
    model_new = copy.deepcopy(model)
    model_new_state_dict = model_new.state_dict()

    def _fetch_old_param(sd, key):
        if key in sd:
            return sd[key]
        candidates = []
        if key.startswith('module.') and not key.startswith('module.base.'):
            candidates.append(key.replace('module.', 'module.base.', 1))
        if 'module.base.' in key:
            candidates.append(key.replace('module.base.', 'module.', 1))
        if key.startswith('module.'):
            candidates.append(key[len('module.'):])
        else:
            candidates.append('module.' + key)
        if key.startswith('base.'):
            candidates.append(key.replace('base.', '', 1))
        if '.base.' not in key:
            parts = key.split('.')
            if parts:
                base_insert = parts[:]
                base_insert.insert(len(base_insert) - 1, 'base')
                candidates.append('.'.join(base_insert))
        for cand in candidates:
            if cand in sd:
                return sd[cand]
        return None
    '''fuse the parameters'''
    for k, v in model_state_dict.items():
        old_param = _fetch_old_param(model_old_state_dict, k)
        if old_param is None:
            # key absent in old state dict -> keep new weights as-is
            continue
        if old_param.shape == v.shape:
            model_new_state_dict[k] = alpha * v + (1 - alpha) * old_param
        else:
            print(k, '...')
            num_class_old = old_param.shape[0]
            model_new_state_dict[k][:num_class_old] = alpha * v[:num_class_old] + (1 - alpha) * old_param
    model_new.load_state_dict(model_new_state_dict)
    return model_new


def make_optimizer_stage2(model, args): 
    params = []
    for key, value in model.named_parameters():
        if "text_encoder" in key or "prompt_learner" in key:
            value.requires_grad_(False)
            continue
        
        if not value.requires_grad:
            continue
        lr = 0.000005
        weight_decay = 0.0001
        if "region_attn" in key:
            lr=0.000005 * args.query_lr
            weight_decay = 0.0
        elif "bias" in key:
            lr = 0.000005 * 2
            weight_decay = 0.0001
        params.append({"params": [value], "lr": lr, "weight_decay": weight_decay})
    if not params:
        raise RuntimeError('No parameters found for stage2 optimizer')
    return torch.optim.Adam(params)    


def expand_prompt_with_stage1(base, prompt_state, start_idx, total_classes):
    prompt_learner = base.prompt_learner
    old_ctx = prompt_learner.cls_ctx.data
    n_ctx, dim = old_ctx.size(1), old_ctx.size(2) # (n_cls, n_ctx, dim)

    new_ctx = old_ctx.new_empty(total_classes, n_ctx, dim)
    new_ctx[:start_idx].copy_(old_ctx[:start_idx]) 
    domain_ctx = prompt_state['cls_ctx']

    domain_ctx = domain_ctx.to(new_ctx.device)
    new_ctx[start_idx:start_idx + domain_ctx.size(0)].copy_(domain_ctx)
    prompt_learner.cls_ctx = torch.nn.Parameter(new_ctx)  
    prompt_learner.num_class = total_classes

def refresh_prompt_with_stage1(base, prompt_state, num_classes):
    prompt_leaner = base.prompt_learner
    domain_ctx = prompt_state['cls_ctx']
    domain_ctx = domain_ctx.to(prompt_leaner.cls_ctx.device)
    
    prompt_leaner.cls_ctx = torch.nn.Parameter(domain_ctx)
    prompt_leaner.num_class = domain_ctx.size(0) 


def expand_linear_head(ln, out_dim):
    in_dim = ln.in_features
    old_weight = ln.weight.data.clone()
    new_linear = torch.nn.Linear(in_dim, out_dim, bias=False).to(ln.weight.device)
    torch.nn.init.normal_(new_linear.weight, std=0.01)
    copy_rows = min(old_weight.size(0), out_dim)

    new_linear.weight.data[:copy_rows].copy_(old_weight[:copy_rows])
    return new_linear


def prompt_initialize(base, prompt_state):
    prompt_learner = base.prompt_learner
    prompt_learner.load_state_dict(prompt_state, strict=False)
    prompt_learner.num_class = prompt_state['cls_ctx'].shape[0]


def load_prompt(path):
    state = torch.load(path, map_location='cpu')

    if isinstance(state, dict) and 'state_dict' in state:
        return state['state_dict']
    return state 

def _set_stage2_mode(model,args):
    base = model.module.base 

    if hasattr(base, 'prompt_learner'):
        for p in base.prompt_learner.parameters():
            p.requires_grad = False
    if hasattr(base, 'text_encoder'):
        for p in base.text_encoder.parameters():
            p.requires_grad = False

    if hasattr(base, 'region_attn'):
            for p in base.region_attn.parameters():
                p.requires_grad = True
    if hasattr(base, 'region_proj'):
            for p in base.region_attn.parameters():
                p.requires_grad = True


    for m in [getattr(base, 'image_encoder', None), getattr(base, 'classifier', None),getattr(base, 'classifier_proj', None),getattr(base, 'bottleneck', None),getattr(base, 'bottleneck_proj', None)]:
        if m is not None:
            for p in m.parameters():
                p.requires_grad = True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=8)
    parser.add_argument('--height', type=int, default=256, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")
    parser.add_argument('--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    parser.add_argument('--debug_eval', action='store_true', help="print feature stats during evaluation for debugging")
    # model    
    parser.add_argument('--MODEL', type=str, default='50x',
                        choices=['50x'])
    # optimizer
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'],
                        help="optimizer ")
    parser.add_argument('--lr', type=float, default=0.008,
                        help="learning rate of new parameters, for pretrained ")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-step', type=int, default=10)
    parser.add_argument('--milestones', nargs='+', type=int, default=[30],
                        help='milestones for the learning rate decay')
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--eval_epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--print-freq', type=int, default=200)
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/path/to/your/data')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join('./RESULTS/try'))
    parser.add_argument('--config_file', type=str, default='config/base.yml',
                        help="config_file")
    parser.add_argument('--test_folder', type=str, default=None, help="test the models in a file")
    parser.add_argument('--setting', type=int, default=1, choices=[1, 2,51,52,53,54,55], help="training order setting")
    parser.add_argument('--middle_test', action='store_true', help="test during middle step")
    parser.add_argument('--AF_weight', default=1.0, type=float, help="anti-forgetting weight")   
    parser.add_argument('--global_alpha',  type=float, default=100,  help="")   
    parser.add_argument('--absolute_feat',  action='store_true', help="")        
    parser.add_argument('--save_evaluation', action='store_true', help="save ranking results")
    parser.add_argument('--absolute_delta', action='store_true',default=True, help="only use dual teacher")
    parser.add_argument('--trans', action='store_true',default=True, help="only use dual teacher")
    parser.add_argument('--af_temp', type=float, default=0.05, help='temperature (smaller -> sharper) for AF affinity softmax')
    parser.add_argument('--af_feat_branch', type=str, default='proj', choices=['main','proj'], help='which feature space to compute AF (KL) on: main=768D, proj=512D')
    parser.add_argument('--use_center_loss', action='store_true', help='enable center loss on projection features (512D)')
    parser.add_argument('--center_loss_weight', type=float, default=1, help='weight for center loss term')
    parser.add_argument('--center_lr', type=float, default=0.5, help='learning rate for center loss parameters (centers)')
    parser.add_argument('--stage1_prompts_out_dir', type=str,default='./STAGE1_PROMPTS_WEIGHT')
    parser.add_argument('--BLIP_text_out_dir', type=str, default='./BLIP_TEXT_DESC')
    parser.add_argument('--ablation_part', type=str, default='all')
    parser.add_argument('--query_AF', action='store_true')
    parser.add_argument('--other_details', default=None, type=str)
    parser.add_argument('--testing', default=None, type=str)
    parser.add_argument('--fine_weight', default=1.0, type=float)
    parser.add_argument('--query_lr', default=1, type=float)
    parser.add_argument('--fix_EMA', default=0.5, type=float, help="model fusion weight") 
    parser.add_argument('--AF_beta', type=float, default=1.0)
    parser.add_argument('--best_alpha', default=0.6, type=float)
    parser.add_argument('--weak', action='store_true')
    parser.add_argument('--alpha', type=float,default=1.0)
    parser.add_argument('--eval_stage', type=str, default='0,1,2,3,4')

    main()
