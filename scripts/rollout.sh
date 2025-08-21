cd ../src/syn_po_data
export MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
export CHUNK_SIZE=1500
export CUDA_VISIBLE_DEVICES=0
export ROUNDS=30
export TOP_K=4
export INIT_METHOD="probe_similarity"
export START=0
export END=5000

echo "Model Path $MODEL_PATH"
echo "Running rollout from $START to $END"
python rollout.py \
 --model_path $MODEL_PATH \
 --input_dir "../../data/train_data/${CHUNK_SIZE}/step1_passage2probe_${START}_${END}.jsonl" \
 --output_dir "../../data/train_data/${CHUNK_SIZE}/step2_rollout_top${TOP_K}_rounds${ROUNDS}_${START}_${END}.jsonl" \
 --rounds $ROUNDS \
 --top_k $TOP_K \
 --init_method $INIT_METHOD
