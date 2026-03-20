
CUDA_VISIBLE_DEVICES=0 python train_stage1.py \
--get_prompts yes \
--get_texts no \
--stage1_prompts_out_dir ./_my_prompts \
--BLIP_text_out_dir ./_my_text_desc \
--data-dir /data/dataset/xukunlun/PRID \
--config_file config/base.yml \
--logs-dir ./_RESULTS \
--setting 1 \
--seed 0
