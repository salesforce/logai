#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from transformers import BertConfig, BertForMaskedLM
import math 
import os 
import pandas as pd 
from datasets import Dataset as HFDataset
from datasets import DatasetDict


def load_dataset(data_file, data_field, special_tokens):
    #dataset = load_dataset("csv", delimiter=",", data_files={"test": test_txt_file}, column_names=['loglines'])
    data_df = pd.read_csv(data_file)
    data_df[data_field+'_removed_specialtokens'] = data_df[data_field].apply(lambda x: " ".join(set(x.split(' '))-set(special_tokens)))
    data_df = data_df[data_df[data_field+'_removed_specialtokens'] != ""]
    d = pd.DataFrame({data_field: list(data_df[data_field])})
    dataset = HFDataset.from_pandas(d)
    labels = {k:v for k,v in enumerate(list(data_df['labels']))}
    if 'count' in data_df:
        counts = {k:v for k,v in enumerate(list(data_df['count']))}
        print ("Total positive instances ", sum([counts[k] for k in labels if labels[k]==1]))
        print ("Total negative instances ", sum([counts[k] for k in labels if labels[k]==0]))
    else:
        counts = None
    return dataset, labels, counts


dataset_name = "HBT"#"Thunderbird" #"BGL" "HDFS"

train_file = "../../datasets/public/"+dataset_name+"/output/nonparsed/"+dataset_name+"_train.csv"
valid_file = "../../datasets/public/"+dataset_name+"/output/nonparsed/"+dataset_name+"_dev.csv"

if dataset_name == "HDFS":
    block_size = 384
    lr = 1e-5
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[INT]", "[IP]", "[BLOCK]", "[FILE]",  ".", "*", ":", "$", "_", "-", "/"]    

elif dataset_name == "BGL":
    block_size = 120
    lr = 1e-5
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[HEX]", "[INT]", "[IP]", ".", "*", ":", "$", "_", "-", "/"]

elif dataset_name == "Thunderbird":
    block_size = 120
    lr = 1e-4
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[FILE]", "[HEX]", "[INT]", "[IP]", "[WARNING]", "[ALPHANUM]", ".", "*", ":", "$", "_", "-", "/"]


elif dataset_name == "HBT":
    block_size = 384
    lr = 1e-5
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[HEX]", "[IP]", "[INT]", "[WARNING]", "[ALPHANUM]", "[FILE]", "[BLOCK]", ".", "*", ":", "$", "_", "-", "/"]

def tokenize_function(examples):
    return tokenizer(examples["loglines"], truncation=True, padding='max_length', max_length=block_size)


train_dataset, _, _ = load_dataset(train_file, "loglines", special_tokens)
valid_dataset, _, _ = load_dataset(valid_file, "loglines", special_tokens)

datasets = DatasetDict({"train": train_dataset, "validation": valid_dataset})

train_dir = os.path.dirname(train_file)


model_checkpoint = "bert-base-cased"
if model_checkpoint ==  "bert-base-uncased":
    custom_tokenizer = os.path.abspath(os.path.join(train_dir, "log-bert-uncased-tokenizer"))
elif model_checkpoint == "bert-base-cased":
    custom_tokenizer = os.path.abspath(os.path.join(train_dir,"log-bert-cased-tokenizer"))
print ('Taking custom_tokenizer from ',custom_tokenizer)
tokenizer = AutoTokenizer.from_pretrained(custom_tokenizer, use_fast=True)

model_dir = os.path.join(train_dir, 'models')
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_dir = os.path.join(model_dir, model_checkpoint+'_finetuned_custom-tokenizer_lr'+str(lr)+'_maxlen'+str(block_size))
if os.path.exists(model_dir) and len(os.listdir(model_dir))>0:
    model_checkpoint = model_dir
    checkpoint_dir = 'checkpoint-'+str(max([int(x.split('-')[1]) for x in os.listdir(model_dir)]))
    model_checkpoint = os.path.abspath(os.path.join(model_checkpoint, checkpoint_dir))

tokenized_datasets = datasets.map(tokenize_function, batched=True, num_proc=4, remove_columns=["loglines"])

pretrain_from_scratch = True 

if pretrain_from_scratch is False:
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
else:
    vocab_size = len(open(os.path.join(custom_tokenizer, 'vocab.txt')).readlines())
    config = BertConfig(vocab_size=vocab_size)
    model = BertForMaskedLM(config)
    model.tokenizer = tokenizer


training_args = TrainingArguments(
    model_dir,
    evaluation_strategy = "steps",
    num_train_epochs=20,
    learning_rate=lr,
    logging_steps=10,
    per_device_train_batch_size=50,
    per_device_eval_batch_size=100,
    weight_decay=0.0001,
    save_steps=500,
    eval_steps=500,
    resume_from_checkpoint=True,
    push_to_hub=False
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, pad_to_multiple_of=block_size)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator
)

trainer.train()

eval_results = trainer.evaluate()
print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")






