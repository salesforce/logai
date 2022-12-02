#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd 
from datasets import Dataset as HFDataset

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