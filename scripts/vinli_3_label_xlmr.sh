WANDB_PROJECT='Loss-Function-cross'
MODEL_NAME_OR_PATH='xlm-roberta-large'
DATASET_NAME='presencesw/snli'
DATASET='snli'
OUTPUT_DIR='output/xlm-roberta-large/snli'
RUN_NAME="xlm-roberta-large--snli_5e-06"
HF_TOKEN='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'
BZ=15
GRA_ACC=5

accelerate launch --num_processes 2 --gpu_ids 0,1 --config_file ds.yaml -m loss_nli.training \
    --hf_token $HF_TOKEN \
    --wandb_project $WANDB_PROJECT \
    --num_labels 3 \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --dataset_name $DATASET_NAME \
    --output_dir $OUTPUT_DIR \
    --do_train \
    --train_dir 'data/finetune/train' \
    --do_eval \
    --valid_dir 'data/finetune/validation' \
    --per_device_train_batch_size $BZ \
    --gradient_accumulation_steps $GRA_ACC \
    --num_train_epochs 15 \
    --save_steps 100 \
    --eval_steps 100 \
    --logging_steps 100 \
    --learning_rate 5e-06 \
    --evaluation_strategy 'steps' \
    --overwrite_output_dir \
    --loss_func_name 'cross' \
    --save_total_limit 1 \
    --overwrite_output_dir \
    --run_name $RUN_NAME \
    --save_only_model
