from typing import Optional
from dataclasses import dataclass, field
from transformers import MODEL_FOR_MASKED_LM_MAPPING
from transformers.utils.versions import require_version


MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

################################################################################
#                                HF Model                                      #
################################################################################

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    # low_cpu_mem_usage: bool = field(
    #     default=False,
    #     metadata={
    #         "help": (
    #             "It is an option to create the model as an empty shell, then only materialize its parameters when the "
    #             "pretrained weights are loaded. Set True will benefit LLM loading time and RAM consumption."
    #         )
    #     },
    # )
    # quantize:bool = field(
    #     default=False,
    #     metadata={
    #         "help" : (
    #             "Quantize 4bit"
    #         )
    #     }
    # )
    
    

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


################################################################################
#                           HF Data Training / Eval                            #
################################################################################

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_dir: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    valid_dir: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_len: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated."
            )
        },
    )
    dataset_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_valid_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    train_path_list: Optional[str] = field(default=None,
                                           metadata={"help": "The input training data file (a text file)."})
    validation_path_list: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    loss_func_name: str = field(
        default=None,
        metadata={"help": "Choose loss function in ['cross', 'triplet', 'contras', 'cosine']"}
    )
    num_labels: int = field(default=3, metadata={"help": "choose number of label in dataset"})
    load_all_labels: bool=field(default=False, metadata={"help": "if true load 4 label, vice versa"})
    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")



################################################################################
#                                     LoRA                                     #
################################################################################

# @dataclass
# class LoraArguments:
#     """
#     Arguments pertaining to what data we are going to input our model for training and eval.
#     """

#     use_lora: bool = field(default=False, metadata={"help": "Using LoRA finetune model"})
#     lora_bias: Optional[str] = field(
#         default='none', metadata={"help": "Bias type for Lora. Can be [none], [all] or [lora_only]"}
#     )
#     lora_r: Optional[int] = field(default=8, metadata={"help": "Lora attention dimension."})
#     lora_alpha: Optional[int] = field(default=16, metadata={"help": "The alpha parameter for Lora scaling."})
#     lora_dropout: Optional[float] = field(default=0.0, metadata={"help": "The dropout probability for Lora layers."})
#     use_int8_training: bool = field(default=False, metadata={"help": "Using int8 training"})
#     lora_name_id: Optional[str] = field(default='gradients-vienBERT', metadata={"help":"The name of save LoRA model"})
#     save_base: Optional[bool] = field(default=False, metadata={"help" : "Save base model"})