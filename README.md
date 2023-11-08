First install python package

```python
pip insatll -r requirements.txt
```

Then install setup package the code

```python
python setup.py develop
```

To download dataset for this empirical, run code in root folder

```bash
bash download.sh
```

To load the dataset, please run:

```python
dataset = data.dataset_name(tokenizer_name="<model_name>").get_dataset()
Ex:
dataset = data.ViNLI(tokenizer_name="xlmr").get_dataset()
```
