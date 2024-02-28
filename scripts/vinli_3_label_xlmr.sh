MODEL_NAME_OR_PATH='xlm-roberta-large'
DATASET_NAME='presencesw/vinli_3_label'
DATASET='vinli_3_label'
OUTPUT_DIR='output/'xlm-roberta-large'/vinli_3_label'
RUN_NAME="xlm-roberta-large--vinli_3_label"
BZ=14
GRA_ACC=8

!accelerate launch --num_processes 2 --gpu_ids 0,1 -m loss_nli.training \
    --num_labels 3 \
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
    --evaluation_strategy 'epoch' \
    --save_strategy 'epoch' \
    --logging_strategy 'epoch' \
    --overwrite_output_dir \
    --loss_func_name 'cross' \
    --load_best_model_at_end \
    --save_total_limit 1 \
    --overwrite_output_dir \
    --run_name $RUN_NAME \
    --save_only_model \
    --lr_scheduler_type "linear" \
    --learning_rate "5e-5"