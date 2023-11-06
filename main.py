from loss_nli.data import data

data_path = '../data/vinli/UIT_ViNLI_1.0_test.jsonl'
vinli = data.ViNLI(data_path=[data_path],features=['sentence1', 'sentence2', 'gold_label'], bs=8)
dataset = vinli.transform_data()