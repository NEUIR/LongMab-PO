cd ../src/syn_po_data
export MODEL_PATH="/data1/duanshaohua/hf_hub/models--meta-llama--Llama-3.1-8B-Instruct"
export CUDA_VISIBLE_DEVICES=0
export MAX_PROMPT_LENGTH=8192

python construct_po_data.py \
 --model_path $MODEL_PATH \
 --input_dir "../../data/training_data/1500/step2_rollout_top4_rounds4_0_50.jsonl" \
 --output_dir "../../data/training_data/1500/step3_dpodata_prompt${MAX_PROMPT_LENGTH}.jsonl" \
 --max_prompt_length $MAX_PROMPT_LENGTH
