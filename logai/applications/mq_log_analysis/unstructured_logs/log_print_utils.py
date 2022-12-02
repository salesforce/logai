#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={'figure.figsize':(11.7,8.27)})

def trim_logline_to_print(logline, max_len):
    if len(logline)>max_len:
        return logline[:max_len].strip()+' ... '
    else:
        return logline 
        
def print_log_cluster(parsed_group_data, data_type, cluster_label):
    topk = 10
    parsed_group_unique_patterns = parsed_group_data['pattern'].head(topk)
    parsed_group_unique_patterns_counts = parsed_group_data['coverage'].head(topk)
    if topk < len(parsed_group_data):
        print ('top-'+str(topk)+' of unique '+data_type+' (out of '+str(len(parsed_group_data))+') in Cluster : ',cluster_label)
    else:
        print ('All unique '+data_type+' in Cluster : ',cluster_label)
    print ('\t'+'\n\t'.join([trim_logline_to_print(x.replace('sfdc.common.messaging.','...'), 500).strip()+'\t\t'+str(round(y*100, 3))+"%" for x,y in zip(list(parsed_group_unique_patterns), list(parsed_group_unique_patterns_counts))]))
    print ('\n')

def print_func_distribution(parsed_group_data):
    parsed_group_data['funcname'] = parsed_group_data['pattern'].apply(lambda x: x.split(' ')[0])
    funcname_distribution = {}
    for data in parsed_group_data.groupby('funcname'):
        funcname = data[0]
        coverage = sum(list(data[1]['coverage']))
        funcname_distribution[funcname] = coverage
    funcname_distribution = {k: v for k, v in sorted(funcname_distribution.items(), key=lambda item: item[1], reverse=True)}
    for k,v in funcname_distribution.items():
        print ('\t',k.replace('sfdc.common.messaging.','...'),'\t', str(round(v*100,3))+"%")
        

def cluster_tsne(feature_vector, clusters):
    tsne_em = TSNE(n_components=2, perplexity=30.0, n_iter=1000, verbose=1)
    X_embedded = tsne_em.fit_transform(feature_vector)
    Y = clusters
    palette = sns.color_palette("bright", len(set(Y.tolist())))
    sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=Y, legend=None, palette=palette)
    plt.show()
