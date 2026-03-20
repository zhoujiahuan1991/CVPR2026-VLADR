from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import sys
import os
from reid.models.BLIP_gen import BLIP_Text_Generator
import random
from config import cfg
from reid.utils.logging import Logger
from reid.utils.lr_scheduler import WarmupMultiStepLR
from reid.utils.feature_tools import *
from reid.models.layers import DataParallel
from tqdm import tqdm
from reid.models.wrapper import make_model
from lreid_dataset.datasets.get_data_loaders import build_data_loaders
from tools.Logger_results import Logger_res
from tqdm import tqdm
import datetime
from reid.models.BLIP_gen import BLIP_Text_Generator
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning)
from reid.models.CLIP_ReID.loss.supcontrast import SupConLoss

OUTPUT_PROMPT_DIR = "CLIP_REID_PROMPT_WEIGHT"
OUTPUT_TEXT_DIR = "DOMAIN_TEXT_DESC"

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

    os.makedirs(args.stage1_prompts_out_dir, exist_ok=True)
    os.makedirs(args.BLIP_text_out_dir, exist_ok=True)

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
    log_name = f'log_{timestamp}.txt'
    stage1_dir = osp.join(args.logs_dir, 'stage1', timestamp)

    if not args.evaluate:
        sys.stdout = Logger(osp.join(stage1_dir, log_name))
    else:
        log_dir = osp.dirname(args.test_folder)
        sys.stdout = Logger(osp.join(stage1_dir, log_name))

    log_res_name=f'log_res_{timestamp}.txt'
    logger_res=Logger_res(osp.join(stage1_dir, log_res_name))  

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

    supcon_loss = SupConLoss(device='cuda')

    if args.get_texts == 'yes':
        text_save_dir = args.BLIP_text_out_dir
        fpaths_get_tool = dataset_fpaths_get(root=args.data_dir)

        dataset_mapping = {
            'market1501':'Market-1501', # Market-1501/bounding_box_train/ *.jpg
            'cuhk_sysu': 'cuhksysu4reid', # cuhksysu4reid/train/ *.jpg
            'dukemtmc': 'DukeMTMC-reID', # DukeMTMC-reID/bounding_box_train/*.jpg
            'cuhk03': 'CUHK03_lifelong', # CUHK03_lifelong/image_labeled/*.png 
            'msmt17':'MSMT17_V2', # MSMT17_V2/mask_train_v2/{pid}/{pid}*.jpg
            'lpw':'LPW_s2' # LPW_s2/bounding_box_train/*.jpg
        }

        fpaths_get_mapping = {
            'market1501':fpaths_get_tool.market_use(dataset_dir=dataset_mapping['market1501']),
            'cuhk_sysu': fpaths_get_tool.cuhksysu_use(dataset_dir=dataset_mapping['cuhk_sysu']),
            'dukemtmc': fpaths_get_tool.duke_use(dataset_dir=dataset_mapping['dukemtmc']),
            'cuhk03': fpaths_get_tool.cuhk03_use(dataset_dir=dataset_mapping['cuhk03']),
            'msmt17':fpaths_get_tool.msmt17_use(dataset_dir=dataset_mapping['msmt17']),
            'lpw':fpaths_get_tool.lpw_use(dataset_dir=dataset_mapping['lpw'])
        }
        for set_index, train_info in enumerate(all_train_sets):
            dataset, num_classes, clip_stage1_loader, clip_stage2_loader, BLIP_loader, test_loader, init_loader, name = train_info

            fpaths = fpaths_get_mapping[name]

            if fpaths is None:
                raise RuntimeError('No fpaths found for dataset {} !!'.format(name))
            
            text_save_path = osp.join(text_save_dir, f'{name}_description.txt')
            print(f'[Stage1] BLIP begins to get the texts for images in domain {set_index + 1}/{len(all_train_sets)}: {name}')   

            batch_size=16
            for i in range(0, len(fpaths), batch_size):
                batch_paths=  fpaths[i : i + batch_size]

                records_for_this_batch = BLIP_Text_Generator(fpaths=batch_paths,device='cuda', dataset_name=name,text_save_dir=text_save_dir)
                append_record_to_txt(text_save_path, records_for_this_batch)        

        print('='*80)
        print('Stage1 BLIP is over...Coming up for clipreid stage1 prompt training...')
        print('='*80)

    if args.get_prompts == 'yes':
        for set_index, train_info in enumerate(all_train_sets):
            dataset, num_classes, clip_stage1_loader, clip_stage2_loader, BLIP_loader, test_loader, init_loader, name = train_info

            print('='*80)
            print(f'[Stage1] Training the prompt for domain {set_index + 1}/{len(all_train_sets)}: {name}')
            print('='*80)

            model = make_model(num_class=num_classes, camera_num=0, view_num=0)
            model.to('cuda')
            model = DataParallel(model)

            Stones=args.milestones
            base = _set_stage1_mode(model=model)
            optimizer_stage1 = _make_optimizer_for_stage1(base)
            lr_scheduler_1 = WarmupMultiStepLR(optimizer_stage1, Stones, gamma=0.1, warmup_factor=0.01, warmup_iters=args.warmup_step)

            stage1_epochs = 120 #120 
            for epoch in range(stage1_epochs):
                train_prompt_epoch(model, base, clip_stage1_loader, optimizer_stage1, supcon_loss, lr_scheduler_1, dataset_name=name,epoch=epoch)

            prompt_path = osp.join(args.stage1_prompts_out_dir, f'{name}_clipreid_prompt.pth')

            prompt_learner = getattr(base, 'prompt_learner', None)
            if prompt_learner is None:
                raise RuntimeError('prompt learner is None!!!!!!!!!')
            
            state = {
                "state_dict": prompt_learner.state_dict(),
                'num_class': prompt_learner.num_class,
                'n_ctx': getattr(prompt_learner, 'n_cls_ctx', None),
                'dataset_name': name
            }

            torch.save(state, prompt_path)
            print(f'Prompt Weight for {name} saved at: {prompt_path}')

        print('='*80)
        print('Stage1 prompt training ends...')
        print('='*80)

    print('Finished...')

import glob
class dataset_fpaths_get:
    def __init__(self, root):
        self.root = root

    def market_use(self, dataset_dir):
        imgs_dir = osp.join(self.root, dataset_dir, 'bounding_box_train')
        fpaths = sorted(glob.glob(osp.join(imgs_dir, '*.jpg')))
        print(f'market fpaths collected, got{len(fpaths)}...')
        return fpaths
    
    def cuhksysu_use(self, dataset_dir):
        imgs_dir = osp.join(self.root, dataset_dir, 'train')
        fpaths = sorted(glob.glob(osp.join(imgs_dir, '*.jpg')))
        print(f'cuhksysu fpaths collected, got{len(fpaths)}...')
        return fpaths
    
    def cuhk03_use(self, dataset_dir):
        imgs_dir = osp.join(self.root, dataset_dir, 'images_labeled')
        fpaths = sorted(glob.glob(osp.join(imgs_dir, '*.png')))
        print(f'cuhk03 fpaths collected, got{len(fpaths)}...')
        return fpaths

    def msmt17_use(self, dataset_dir):
        fpaths = []
        pid_dirs = sorted(glob.glob(osp.join(self.root, dataset_dir, 'mask_train_v2', '*')))
        for pid_dir in pid_dirs:
            img_paths = sorted(glob.glob(osp.join(pid_dir, '*.jpg')))
            for img_path in img_paths:
                fpaths.append(img_path)
        print(f'msmt17 fpaths collected, got{len(fpaths)}...')
        return fpaths

    def duke_use(self, dataset_dir):
        imgs_dir = osp.join(self.root, dataset_dir, 'bounding_box_train')
        fpaths = sorted(glob.glob(osp.join(imgs_dir, '*.jpg')))
        print(f'duke fpaths collected, got{len(fpaths)}...')
        return fpaths

    def lpw_use(self, dataset_dir):
        imgs_dir = osp.join(self.root, dataset_dir, 'bounding_box_train')
        fpaths = sorted(glob.glob(osp.join(imgs_dir, '*.jpg')))
        print(f'lpw fpaths collected, got {len(fpaths)}...')
        return fpaths

def append_record_to_txt(path, records_for_this_batch):
    def _check(s):
        s = '' if s is None else str(s)
        return s.replace('\t', ' ').replace('\n', ' ')

    sep = '\t'*3
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a') as f:
        for r in records_for_this_batch:

            # r['fname'] -> file name
            # r['description'] -> a list of 4 strings, namely "Head...", "upper...", ...
            filename = r['fname']
            description = r['description']

            description = [_check(des) for des in description[:4]]
            line = filename + sep + sep.join(description) + '\n'
            f.write(line)

def train_prompt_epoch(model, base, loader, optimizer, supcon_loss, lr_scheduler=None, \
                       log_interval=None, dataset_name=None,epoch=1):
    model.train()
    loader.new_epoch() 
    losses = []

    num_iters = len(loader) 

    for iter in tqdm(range(num_iters), desc=f'[Epoch:{epoch}, Dataset:{dataset_name}]'):

        inputs = loader.next() 

        img_transformed, fpaths, targets, camids, domains = inputs
        img_transformed = img_transformed.to('cuda')
        targets = targets.to('cuda')

        optimizer.zero_grad()
        with torch.no_grad():
            img_features = base(x=img_transformed, get_image=True) 
        txt_features = base(label=targets, get_text=True)

        i2t = supcon_loss(img_features, txt_features, targets, targets)
        t2i = supcon_loss(txt_features, img_features, targets, targets)

        loss_stage1 = i2t + t2i

        loss_stage1.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

    print(f'\n[stage1]: i2t loss: {i2t}\n[stage1]: t2i loss: {t2i}')
    print(f'[stage1]: total loss: {loss_stage1}')

def _set_stage1_mode(model):
    if hasattr(model, 'module'):
        base = model.module.base
    else:
        base = model.base

    for m in [getattr(base, 'image_encoder', None),
              getattr(base, 'classifier', None),
              getattr(base, 'classifier_proj', None),
              getattr(base, 'bottleneck', None),
              getattr(base, 'bottleneck_proj', None)]:
        if m is not None:
            for p in m.parameters():
                p.requires_grad = False

    if hasattr(model.module if hasattr(model, 'module') else model, 'classifier'):
        classifier = getattr(model.module if hasattr(model, 'module') else model, 'classifier')
        if isinstance(classifier, torch.nn.Module):
            for p in classifier.parameters():
                p.requires_grad = False

    if hasattr(base, 'text_encoder'):
        for p in base.text_encoder.parameters():
            p.requires_grad = False

    if hasattr(base, 'prompt_learner'):
        for p in base.prompt_learner.parameters():
            p.requires_grad = True

    if hasattr(base, 'region_attn'):
        for p in base.region_attn.parameters():
            p.requires_grad = False

    return base

def _make_optimizer_for_stage1(model):
    params = []
    keys = []

    for key, value in model.named_parameters():
        if 'prompt_learner' in key:
            params += [         
                {
                    "params": [value],
                    "lr": 0.0035,
                    "weight_decay": 1e-4
                }
            ]
            keys += [keys]
    optimizer = torch.optim.Adam(params=params)
    return optimizer



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Continual training for lifelong person re-identification")
    # data
    parser.add_argument('-b', '--batch-size', type=int, default=16)
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
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--evaluate', action='store_true',
                        help="evaluation only")
    parser.add_argument('--epochs0', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--eval_epoch', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--print-freq', type=int, default=200)
    
    # path   
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default='/data/dataset/xukunlun/PRID')
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join('./RESULTS/try'))

    parser.add_argument('--config_file', type=str, default='config/base.yml',
                        help="config_file")
  
    parser.add_argument('--test_folder', type=str, default=None, help="test the models in a file")
   
    parser.add_argument('--setting', type=int, default=1, choices=[1, 2,51,52,53,54,55], help="training order setting")
    parser.add_argument('--middle_test', action='store_true', help="test during middle step")
    parser.add_argument('--AF_weight', default=1.0, type=float, help="anti-forgetting weight")   

    parser.add_argument('--fix_EMA', default=0.5, type=float, help="model fusion weight") 
   
    parser.add_argument('--global_alpha',  type=float, default=100,  help="")   
    parser.add_argument('--absolute_feat',  action='store_true', help="")        
    parser.add_argument('--save_evaluation', action='store_true', help="save ranking results")
    parser.add_argument('--absolute_delta', action='store_true',default=True, help="only use dual teacher")
    parser.add_argument('--trans', action='store_true',default=True, help="only use dual teacher")

    # parser.add_argument('--color_style', type=str,default='rgb', help="select the color", choices=['lab','rgb'])
    # parser.add_argument('--learn_kernel', action='store_true', help="learnable style transfer kernel")
    parser.add_argument('--random_rehearser', action='store_true', help="select a random rehearser for data augmentation")
    parser.add_argument('--blur', action='store_true', help="adopt blur augmentation")
    parser.add_argument('--n_kernel', default=1, type=int, help="number of Distribution Transfer kernel")   
    parser.add_argument('--groups', default=1, type=int, help="convolution group number of each Distribution Transfer kernel")  
    parser.add_argument('--joint_test', action='store_true', help="use the AKPNet model during testing")   
    parser.add_argument('--mobile', action='store_true', help="use the mobilenet-v3 as the backbone of synthetic models") 
    parser.add_argument('--aux_weight', default=4.5, type=float, help="the loss weight of rehearsed data, e.g. β in the paper") 
    # ---- Anti-Forgetting & Center Loss tuning additions ----
    parser.add_argument('--af_temp', type=float, default=0.05, help='temperature (smaller -> sharper) for AF affinity softmax')
    parser.add_argument('--af_feat_branch', type=str, default='proj', choices=['main','proj'], help='which feature space to compute AF (KL) on: main=768D, proj=512D')
    parser.add_argument('--use_center_loss', action='store_true', help='enable center loss on projection features (512D)')
    parser.add_argument('--center_loss_weight', type=float, default=1, help='weight for center loss term')
    parser.add_argument('--center_lr', type=float, default=0.5, help='learning rate for center loss parameters (centers)')

    # CLIP_ReID prompts and BLIP text generation
    parser.add_argument('--stage1_prompts_out_dir', type=str,default='./STAGE1_PROMPTS_WEIGHT')
    parser.add_argument('--BLIP_text_out_dir', type=str, default='./BLIP_TEXT_DESC')
    parser.add_argument('--get_prompts', type=str, default='yes')
    parser.add_argument('--get_texts', type=str, default='yes')


    main()
