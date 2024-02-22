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
from .trainer import CrossEntropyLossTrainer, TripletLossTrainer, ContrastiveLossTrainer, CosineSimilarityLossTrainer

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

label_dict = {
    "contradiction": 0,
    "neutral": 1,
    "entailment": 2,
    "other": 3,
    "-": -1
}

def main():

    # init wandb
    os.system("wandb login 138c38699b36fb0223ca0f94cde30c6d531895ca")
    os.environ["WANDB_PROJECT"] = "Loss-Function"

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

    # Quantize config
    # if model_args.quantize:
    #     quant_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type='nf4',
    #         bnb_4bit_compute_dtype=torch.float16,
    #         bnb_4bit_use_double_quant=False
    #     )
    
    # model_type_dict = {
    #     "bert": AutoModelForMaskedLM,
    #     "xlm-r": AutoModelForMaskedLM,
    #     "t5": AutoModelForSeq2SeqLM
    # }

    # for key in model_type_dict:
    #     if key in model_args.model_name_or_path:
    #         auto_model = model_type_dict[key]

    if model_args.model_name_or_path:
        if model_args.use_seq2seq:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(
                model_args.model_name_or_path,
                # quantization_config=quant_config if model_args.quantize else None,
                # device_map={"": 0},
                num_labels=data_args.num_labels
            )
        else:
            base_model = AutoModelForSequenceClassification.from_pretrained(
                model_args.model_name_or_path,
                # quantization_config=quant_config if model_args.quantize else None,
                # device_map={"": 0},
                num_labels=data_args.num_labels
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
    dataset = load_dataset(data_args.dataset_name)
    
    # TODO: calculate max length in train split
    max_length = data.Find_max_length(dataset=dataset, split_dict=None, tokenize_name=model_args.model_name_or_path)

    # TODO: convert labels to classify label
    dataset = dataset.map(lambda example: {"labels": label_dict[example["gold_label"]]}, remove_columns=["gold_label"])

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
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"]
    )
    
    base_model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    
    trainer.train()

    trainer.model.save_pretrained()

if __name__ == '__main__':
    main()