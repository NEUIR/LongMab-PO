cd ../src/evaluation
export CUDA_VISIBLE_DEVICES=0
export INPUT_DIR="../../data/test_data"
export TEMPERATURE=0.0
export BATCH_SIZE=10
export MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
export OUTPUT_DIR="../../result"

echo "Running evaluation for model: $MODEL_PATH"
echo "Output will be saved to: $OUTPUT_DIR"

python eval.py \
  --model_path $MODEL_PATH \
  --input_dir $INPUT_DIR \
  --output_dir $OUTPUT_DIR \
  --temperature $TEMPERATURE \
  --batch_size $BATCH_SIZE
