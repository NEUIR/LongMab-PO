cd ../src/syn_po_data
export MODEL_PATH="/data1/duanshaohua/hf_hub/models--meta-llama--Llama-3.1-8B-Instruct"
export EMBEDDING_MODEL_PATH="/data1/duanshaohua/hf_hub/MiniCPM-embedding"
export RAW_DARASET="/data1/duanshaohua/longmab/data/raw_data/raw_data_42_8k_16k.jsonl"
export CHUNK_SIZE=1500
export START=0
export END=2000

export CUDA_VISIBLE_DEVICES=0

echo "Model Path $MODEL_PATH"
echo "Running gen_probe from $START to $END"
python gen_probe.py \
 --model_path $MODEL_PATH \
 --embedding_model_path $EMBEDDING_MODEL_PATH \
 --input_dir $RAW_DARASET \
 --output_dir "../../data/training_data/${CHUNK_SIZE}/step1_passage2probe_${START}_${END}.jsonl" \
 --chunk_size $CHUNK_SIZE \
 --start $START \
 --end $END
