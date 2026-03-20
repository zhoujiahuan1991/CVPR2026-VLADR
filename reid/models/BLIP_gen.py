# for single instance in each ID, to generate a text description, and process it into the standard template
# provides the text description of each body part (head, upper, lower, foot)

# head: xxx
# upper_body: xxx
# lower_body: xxx
# foot: xxx

import os
import torch.nn as nn                                                                                                               
import json
import jsonlines
import torch

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
_HF_CACHE = os.path.expanduser('~/.cache/huggingface')  
os.environ['TRANSFORMERS_CACHE'] = _HF_CACHE
os.environ['HF_HOME']            = _HF_CACHE
os.environ['HF_HUB_OFFLINE'] = '0' 

from tqdm import tqdm
import random
import os.path as osp

from lavis.models import load_model_and_preprocess
import torch
from PIL import Image

os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'
torch.set_num_threads(4)

_BLIP_CACHE = {}

def _ensure_device(device):
    if device is None:
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if isinstance(device, torch.device):
        return device
    return torch.device(device)

def _get_blip_components(device):
    device = _ensure_device(device)
    key = str(device)
    if key not in _BLIP_CACHE:
        print('Loading BLIP...')
        model, vis_processors, _ = load_model_and_preprocess(
            name="blip2_opt",
            model_type='caption_coco_opt2.7b',
            is_eval=True,
            device=device
        )
        model = model.to(device)
        model.eval()
        _BLIP_CACHE[key] = (model, vis_processors)
        print('BLIP2 Loaded...')
    return _BLIP_CACHE[key]


def BLIP_Text_Generator(fpaths, device, dataset_name, text_save_dir):
    '''
    ARGS:
    - fpaths: obtained by train loader
    - device: device
    - ...

    RETURN:
    - descriptions of the imgs in the batch -> [head, upper, lower, foot] by default
    '''

    device = _ensure_device(device)
    model, vis_processors = _get_blip_components(device)

    # BLIP usages (official guidance):
    # caption generate (direct description)
    # >>> answer1 = model.generate({'image': image})

    # most suitable prompt type:
    # single-round:
    # "Question: {your_question} Answer:"
    # multi-round:
    # "Question: {q1} Answer: {a1}. Question: {q2} Answer: {a2}. Question: {q3} Answer:"
    #prompts = {
    #    'Head':"Question: What color is the person's hair? Response in one word. Answer:",
    #    'Upper_body': "Question: What color is the upper body of the person in the image wearing? Answer:",
    #    'Lower_body': "Question: Describe the person's lower body. Answer:",
    #    'Foot': "Question: Describe the footwear. Answer:",
    #}


    # text templates:
    # Head: A {gender} with {color} {length} hair, wearing {accessories}
    # Upper_body: wearing {color} {clothing}, {movements}
    # Lower_body: wearing {color} {clothing}
    # Foot: wearing {color} {type} 

    prompt_dict = {
        'Head': {
            'gender': "Question: What is the gender of the person? Answer:",
            'color': "Question: What color is the person's hair? Answer:",
            'length': "Question: How long is the person's hair? Answer:",
            'accessories': "Question: What accessory is the person wearing? If none, answer none. Answer:",
        },

        'Upper_body': {
            'color': "Question: What color is the upper body clothing? Answer:",
            'clothing': "Question: What is the person wearing on the upper body?. Answer:",
            'movements': "Question: What is the person doing? If none, answer none. Answer:",
        },

        'Lower_body': {
            'color': "Question: What color is the lower body clothing? Answer:",
            'clothing': "Question: What is the person wearing on the lower body?. Answer:",
        },

        'Foot': {
            'color': "Question: What color are the shoes? Answer:",
            'type': "Question: What type of shoes is the person wearing? Answer:",
        }
    }

    print('Generating answer...')

    descriptions = []
    records_for_this_batch = []

    for idx, path in enumerate(tqdm(fpaths,desc='BLIP Generating Discriptions')):
            results = {} 
            fname = osp.basename(path)
            raw_image = Image.open(path).convert('RGB')
            image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)
            for part, prompts in prompt_dict.items():
                results[part] = {}

                for question_type, prompt in prompts.items():
                    answer = model.generate(
                        {'image': image, 'prompt': prompt},
                        use_nucleus_sampling=False, 
                        num_beams=2,
                        max_length=8,
                        min_length=1,
                        repetition_penalty=1.5
                        )
                
                    result = answer[0] if isinstance(answer, list) else answer
                    results[part][question_type] = result

            head_description = 'A {} with {}, {} hair, wearing {}.'.format(results['Head']['gender'].lower(), results['Head']['color'].lower(), results['Head']['length'].lower(), results['Head']['accessories'].lower())
            uppery_body_description = 'Wearing {} {}, {}'.format(results['Upper_body']['color'].lower(), results['Upper_body']['clothing'].lower(), results['Upper_body']['movements'].lower())
            lower_body_description = 'Wearing {} {}'.format(results['Lower_body']['color'].lower(), results['Lower_body']['clothing'].lower())
            foot_description = 'Wearing {} {}'.format(results['Foot']['color'].lower(), results['Foot']['type'].lower())

            description = [head_description, uppery_body_description, lower_body_description, foot_description]
            meta = {
                'dataset':dataset_name,
                'fname':fname,
                'description':description
            }

            records_for_this_batch.append(meta)
            descriptions.append(description)

            if idx % 10 == 0:
                print(descriptions[-1])
    return records_for_this_batch

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

from reid.models.CLIP_ReID.model.clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)
    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, text, tokenized_text):
        x = text + self.positional_embedding.type(self.dtype) 
        x = x.permute(1, 0, 2)  # NLD -> LND 
        x = self.transformer(x) 
        x = x.permute(1, 0, 2)  # LND -> NLD 
        x = self.ln_final(x).type(self.dtype) 
        x = x[torch.arange(x.shape[0]), tokenized_text.argmax(dim=-1)] @ self.text_projection # [B] @ [B, dim]
        return x
    
import random
import torch.nn.functional as F

class Arbitrary_text_encoder:
    def __init__(self):
        '''
        METHOD: encoder(texts):
        ARGS: texts: str list
        RETURN: text feature tensor
        '''
        self.device = 'cuda'
        self.model_name = 'ViT-B-16'
        self.stride = 16
        self.h_resolution = int((256-16)//16 + 1)
        self.w_resolution = int((128-16)//16 + 1)

        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.stride)
        clip_model = clip_model.to(self.device)
        clip_model.eval()

        self.token_embedding = clip_model.token_embedding
        self.dtype = clip_model.dtype
        self.text_encoder = TextEncoder(clip_model)

    def encode(self, texts) -> torch.Tensor:
        tokenized = clip.tokenize(texts).to(self.device)
        with torch.no_grad():
            embedding = self.token_embedding(tokenized).type(self.dtype)
            features = self.text_encoder(embedding, tokenized)
        return F.normalize(features, dim=-1)
    


class Text_feat_Pool:
    PART_NAMES = ['head', 'upper', 'lower', 'foot']
    def __init__(self, text_desc_path, text_encoder, active_part_indices=None):
        self.fname_texts  = {} # {filename: [head, upper, lower, foot]}
        self.filenames = self.fname_texts.keys() 
        self.encoder = text_encoder

        self.part_names = self.PART_NAMES
        self.active_part_indices = active_part_indices
        if self.active_part_indices is not None:
            self.active_part_indices = self._preprocess(active_part_indices)  

        self._load_txt(text_desc_path)
        
    def _preprocess(self, indices):
        mid = [int(idx) for idx in indices]

        seen = set()
        unique = []
        for idx in mid:
            if idx not in seen:
                unique.append(idx)
                seen.add(idx)
                
        return unique

    def _load_txt(self, path):
        with open(path, 'r') as f:
            for _, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                sep = '\t'*3
                tokens = [token.strip() for token in raw_line.split(sep)]
                if not tokens:
                    continue

                filename = tokens[0]
                desc_texts = [tokens[1], tokens[2], tokens[3], tokens[4]]
                self.fname_texts[filename] = desc_texts


    def grab(self, file_names, iter) -> torch.Tensor: 
        '''
        ARGS:
        - file_names: list
        - iter: obtained during the train loop
        '''
        batch_size = len(file_names)

        features = []

        if self.active_part_indices is not None:
            if iter % 30 == 0 and file_names:
                example_key = str(random.choice(file_names))
                example_texts = self.fname_texts.get(example_key)
                if example_texts is not None:
                    print(f'\nGot key: {example_key}')
                    for idx in self.active_part_indices:
                        print(f'  {self.part_names[idx]}: {example_texts[idx]}')

            for name in file_names:
                key = str(name)
                texts = self.fname_texts.get(key)
                if texts is None:
                    raise KeyError(f"Filename {key} not found in text description pool.")
                selected_texts = [texts[idx] for idx in self.active_part_indices]
                text_feat_tensor = self.encoder.encode(selected_texts).cpu()
                features.append(text_feat_tensor)

            batch_feature = torch.stack(features, dim=0) 

            return batch_feature.to(self.encoder.device)
        else:
            return None 




    

