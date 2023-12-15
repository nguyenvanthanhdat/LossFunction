MODEL_NAME_OR_PATH=vinai/phobert-large
OUTPUT_DIR='output/finetune'
BZ=12


python -m loss_nli.training \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR \
    --do_train \
    --train_dir 'data/finetune/train' \
    --do_eval \
    --valid_dir 'data/finetune/validation' \
    --per_device_train_batch_size $BZ \
    --per_device_train_batch_size $BZ \
    --num_train_epochs 4 \
    --max_len 256 \
    --save_steps 1000 \
    --eval_steps 1000 \
    --logging_steps 100 \
    --evaluation_strategy 'steps' \
    --overwrite_output_dir \
    --load_best_model_at_end

$BASH