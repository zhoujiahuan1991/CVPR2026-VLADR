
# order 1
CUDA_VISIBLE_DEVICES=0 python train_stage2.py \
--ablation_part 'all' \
--testing ./_PRETRAINED/order1 \
--data-dir '/data/dataset/xukunlun/PRID' \
--stage1_prompts_out_dir './_STAGE1_PROMPTS_WEIGHT' \
--BLIP_text_out_dir './_BLIP_TEXT_DESC' \
--logs-dir '_RESULTS' \
--query_AF \
-b 64 \
--other_details 'VLADR_Eval_order_1'

# order 2
CUDA_VISIBLE_DEVICES=0 python train_stage2.py \
--ablation_part 'all' \
--testing ./_PRETRAINED/order2 \
--data-dir '/data/dataset/xukunlun/PRID' \
--stage1_prompts_out_dir './_STAGE1_PROMPTS_WEIGHT' \
--BLIP_text_out_dir './_BLIP_TEXT_DESC' \
--logs-dir '_RESULTS' \
--query_AF \
-b 64 \
--other_details 'VLADR_Eval_order_2'
