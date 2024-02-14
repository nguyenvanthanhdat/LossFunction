from datasets import load_dataset, DatasetDict, Dataset
from tqdm import tqdm
import argparse

hf_token = "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"

def print_line():
    print("="*80)

def convert_entailment(exmaples, target):
    labels = exmaples['gold_label']
    labels = [f"not_{target}" if i != target else target for i in labels]
    exmaples['gold_label'] = labels
    return exmaples

def convert_dataset(example):
    # print(example['negative'])
    number = len(example['negative'][0])
    # print(gold_label)
    gold_label = example['gold_label'] * number
    anchor = example['anchor'] * number
    positive = example['positive'] * number
    return {'gold_label': gold_label, 'anchor': anchor, 
            'positive': positive, 'negative': example['negative'][0]}

def add_columns(examples):
    # print(len(examples))
    length = len(examples['gold_label'])
    examples['negative'] = [''] * length
    return examples


def merge_data(dataset_not):
    # duplicate dataset not entailment
    new_dataset_not_entail1 = Dataset.to_pandas(dataset_not)
    new_dataset_not_entail2 = Dataset.to_pandas(dataset_not)
    j = 0
    for i in tqdm(range(len(new_dataset_not_entail1))):
        temp = set()
        # check = False
        # for j in range(len(new_dataset_not_entail2)):
        for index in range(100):
            new_j = j + index
            if new_j == len(new_dataset_not_entail1):
                j = new_j
                break
            if new_dataset_not_entail1.iloc[i]['sentence1'] != new_dataset_not_entail2.iloc[new_j]['sentence1']:
                j = new_j
                break
            temp.add(new_dataset_not_entail2.iloc[new_j]['sentence2'])
        new_dataset_not_entail1.at[i, 'sentence2'] = temp
    dataset_not_entail = new_dataset_not_entail1[new_dataset_not_entail1['sentence2'] != set()]
    return dataset_not_entail

def main(dataset_name, target):
    dataset = load_dataset(dataset_name, token=hf_token)
    print_line()
    print(dataset)
    print_line()

    converted_dataset = dataset.map(convert_entailment, batched=True, fn_kwargs={"target": target})
    new_dataset_entail = converted_dataset.filter(lambda example: example['gold_label'] == target)['train']
    new_dataset_not_entail = converted_dataset.filter(lambda example: example['gold_label'] == f'not_{target}')['train']

    # convert dataset to pandas for next step
    new_dataset_entail = Dataset.to_pandas(new_dataset_entail)
    print(f"MERGE DATA NOT {target}")
    dataset_not_entail = merge_data(new_dataset_not_entail)

    dataset_not_entail = dataset_not_entail.reset_index(drop=True)

    dataset_not_entail = dataset_not_entail.rename(columns={"sentence2": "negative"})
    dataset_not_entail = dataset_not_entail.drop(columns="gold_label")

    new_dataset_entail = new_dataset_entail.merge(dataset_not_entail, how="left", left_on="sentence1", right_on="sentence1")
    new_dataset_entail = new_dataset_entail[new_dataset_entail['negative'].notna()]


    new_dataset = Dataset.from_pandas(new_dataset_entail)
    new_dataset = new_dataset.rename_column("sentence1", "anchor")
    new_dataset = new_dataset.rename_column("sentence2", "positive")
    try:
        new_dataset = new_dataset.remove_columns(["__index_level_0__"])
    except:
        pass
    print(new_dataset)
    print_line()

    print("SEPERATE DATASET")
    new_dataset = new_dataset.map(
        lambda example: convert_dataset(example),                   
        remove_columns=['gold_label', 'anchor', 'positive', 'negative'],
        batched=True,
        batch_size=1
    )
    print(new_dataset)
    print_line()

    converted_dataset = converted_dataset.rename_column("sentence1", "anchor")
    converted_dataset = converted_dataset.rename_column("sentence2", "positive")
    converted_dataset = converted_dataset.map(add_columns, batched=True)

    all_data = DatasetDict()
    if dataset_name == "presencesw/multinli":
        all_data['train'] = new_dataset
        all_data['dev_matched'] = converted_dataset['dev_matched']
        all_data['dev_mismatched'] = converted_dataset['dev_mismatched']
    else:
        all_data['train'] = new_dataset
        all_data['dev3'] = converted_dataset['dev']
        all_data['test'] = converted_dataset['test']
    all_data.push_to_hub(f"{dataset_name}_{target}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        help="The dataset repo or folder in huggingface"
    )
    parser.add_argument(
        "--label",
        help="The target to convert dataset"
    )
    args = parser.parse_args()
    main(args.dataset_name, args.label)