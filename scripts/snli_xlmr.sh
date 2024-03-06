MODEL_NAME_OR_PATH='xlm-roberta-large'
DATASET_NAME='presencesw/snli'
DATASET = DATASET_NAME.split("/")[1]
OUTPUT_DIR=f'output/xlm-roberta-large/snli'
RUN_NAME=f"xlm-roberta-large--snli_TEST"
HF_TOKEN = userdata.get('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')
BZ=14
GRA_ACC=8

!accelerate launch --num_processes 1 --gpu_ids 0 --config_file ds.yaml -m loss_nli.training \
    --hf_token $HF_TOKEN \
    --num_labels 4 \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name $DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --do_train \
    --train_dir 'data/finetune/train' \
    --do_eval \
    --valid_dir 'data/finetune/validation' \
    --per_device_train_batch_size $BZ \
    --per_device_train_batch_size $BZ \
    --gradient_accumulation_steps $GRA_ACC \
    --num_train_epochs 15 \
    --save_steps 100 \
    --eval_steps 100 \
    --logging_steps 20 \
    --evaluation_strategy 'steps' \
    --overwrite_output_dir \
    --loss_func_name 'cross' \
    --load_best_model_at_end \
    --save_total_limit 1 \
    --overwrite_output_dir \
    --run_name $RUN_NAME \
    --save_only_model