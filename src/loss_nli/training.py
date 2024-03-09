from .arguments import ModelArguments, DataTrainingArguments
import logging, sys, os
import torch
import transformers
from .data import data
from transformers import (
    HfArgumentParser,
    set_seed,
    AutoTokenizer,
    AutoConfig,
    CONFIG_MAPPING,
    BitsAndBytesConfig,
    Seq2SeqTrainingArguments,
    AutoModelForSeq2SeqLM,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification
)
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from .trainer import (
    CrossEntropyLossTrainer, 
    TripletLossTrainer, 
    ContrastiveLossTrainer, 
    CosineSimilarityLossTrainer,
    compute_metrics
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

label_3_dict = {
    "contradiction": 0,
    "neutral": 1,
    "entailment": 2,
}

label_4_dict = {
    "contradiction": 0,
    "neutral": 1,
    "entailment": 2,
    "other": 3,
}

number_3_dict = {
    0: "contradiction",
    1: "neutral",
    2: "entailment",
}

number_4_dict = {
    0: "contradiction",
    1: "neutral",
    2: "entailment",
    3: "other"
}

def main():

    # init wandb
    os.system("wandb login 138c38699b36fb0223ca0f94cde30c6d531895ca")
    

    # TODO: Load config 
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # if we pass only one arguments to the scripts and it's the path to a json file,
        # let's parse it to get our arguments.

        model_args, data_args, training_args= parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args= parser.parse_args_into_dataclasses()

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level =  training_args.get_process_log_level()
    logger.setLevel(log_level)

    #Init/enter wandb project
    os.environ["WANDB_PROJECT"] = data_args.wandb_project
    # Detecting Last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logging.info(
                f"Checkpoint detected resuming training at {training_args.output_dir}. To avoid this behavior, change "
                " the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    # TODO: Load tokenizer
    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name)
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    # fix for fp16
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    label_dict = label_3_dict if data_args.num_labels == 3 else label_4_dict
    number_dict = number_3_dict if data_args.num_labels == 3 else number_4_dict 
    if model_args.model_name_or_path:
        if model_args.use_seq2seq:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.model_name_or_path,
                # quantization_config=quant_config if model_args.quantize else None,
                # device_map={"": 0},
                num_labels=data_args.num_labels,
                label2id=label_dict,
                id2label=number_dict,
            )
        else:
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                # quantization_config=quant_config if model_args.quantize else None,
                # device_map={"": 0},
                num_labels=data_args.num_labels,
                label2id=label_dict,
                id2label=number_dict,
            )
        base_model.config.use_cache = False
        base_model.config.pretraining_tp = 1
    else:
        raise NotImplemented
    
    loss_dict = {
        "cross": CrossEntropyLossTrainer,
        "triplet": TripletLossTrainer,
        "contras": CosineSimilarityLossTrainer,
        "cosine": CosineSimilarityLossTrainer
    }
    
    loss_trainer = loss_dict[data_args.loss_func_name]
    
    # dataset = data.ViNLI(tokenizer_name='xlmr', load_all_labels=data_args.load_all_labels).get_dataset()
    dataset = load_dataset(data_args.dataset_name, token=data_args.hf_token)

    # TODO: clean dataset
    dataset = dataset.filter(lambda example: example["gold_label"] in list(label_dict.keys()))
    
    
    # TODO: calculate max length in train split
    max_length = data.Find_max_length(dataset=dataset, split_dict=None, tokenize_name=model_args.model_name_or_path)

    # TODO: convert labels to classify label
    dataset = dataset.map(lambda example: {"labels": label_dict[example["gold_label"]]}, remove_columns=["gold_label"])
    # dataset = dataset.rename_column("gold_label", "labels")

    # TODO: tokenizer dataset
    dataset = dataset.map(
        lambda examples: tokenizer(
            examples["sentence2"], 
            examples["sentence1"],
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        ), 
        batched=True
    )

    
    # TODO: model trainer
    trainer = loss_trainer(
        model=base_model,
        args=training_args,
        # data_collator=data_collator,
        # compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"]
    )
    
    base_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    
    trainer.train()

    trainer.model.save_pretrained()

if __name__ == '__main__':
    main()