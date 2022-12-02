#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from datasets import load_dataset
from tokenizers import decoders, models, normalizers, pre_tokenizers, processors, trainers, Tokenizer
from transformers import BertTokenizerFast
import os
import pandas as pd 
from datasets import Dataset as HFDataset
from datasets import DatasetDict


def load_dataset(data_file, data_field, special_tokens):
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

dataset_name = "HBT" #"Thunderbird" #"BGL" "HDFS"

train_file = "../../datasets/public/"+dataset_name+"/output/nonparsed/"+dataset_name+"_train.csv"

if dataset_name == "HDFS":
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[BLOCK]", "[IP]", "[FILE]", "[INT]"]
elif dataset_name == "BGL":
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[HEX]", "[IP]", "[INT]"]
elif dataset_name == "Thunderbird":
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[HEX]", "[IP]", "[INT]", "[WARNING]", "[ALPHANUM]", "[FILE]"]
elif dataset_name == "HBT":
    special_tokens = ["[UNK]", "[PAD]", "[CLS]", "[SEP]", "[MASK]", "[HEX]", "[IP]", "[INT]", "[WARNING]", "[ALPHANUM]", "[FILE]", "[BLOCK]"]

data_field = "loglines"
train_dir = os.path.dirname(train_file)
dataset, _, _ = load_dataset(train_file, data_field, special_tokens)

model = "bert-base-cased"

batch_size = 1000

def batch_iterator():
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size][data_field]

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
if "-uncased" in model:
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.Lowercase(), normalizers.StripAccents()])
elif "-cased" in model:    
    tokenizer.normalizer = normalizers.Sequence(
        [normalizers.NFD(), normalizers.StripAccents()])
tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

#print (tokenizer.pre_tokenizer.pre_tokenize_str("This is an example!"))


trainer = trainers.WordPieceTrainer(vocab_size=5000, special_tokens=special_tokens)

tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

cls_token_id = tokenizer.token_to_id("[CLS]")
sep_token_id = tokenizer.token_to_id("[SEP]")
print(cls_token_id, sep_token_id)

tokenizer.post_processor = processors.TemplateProcessing(
    single=f"[CLS]:0 $A:0 [SEP]:0",
    pair=f"[CLS]:0 $A:0 [SEP]:0 $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", cls_token_id),
        ("[SEP]", sep_token_id),
    ],
)

tokenizer.decoder = decoders.WordPiece(prefix="##")

new_tokenizer = BertTokenizerFast(tokenizer_object=tokenizer)

if model == "bert-base-uncased":
    tokenizer_dirname = os.path.abspath(os.path.join(train_dir, "log-bert-uncased-tokenizer"))
elif model == "bert-base-cased":
    tokenizer_dirname = os.path.abspath(os.path.join(train_dir, "log-bert-cased-tokenizer"))
new_tokenizer.save_pretrained(tokenizer_dirname)
