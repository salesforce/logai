#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os
from attr import attributes
import pandas as pd
from logai.algorithms.clustering_algo.kmeans import KMeansParams
from logai.utils import constants
from logai.dataloader.data_loader import DataLoaderConfig, FileDataLoader
from logai.preprocess.preprocess import Preprocessor
from logai.information_extraction.log_parser import LogParser
from logai.utils.functions import get_parameter_list
from logai.information_extraction.log_parser import LogParserConfig
from logai.preprocess.preprocess import PreprocessorConfig
import itertools
import numpy as np
from logai.analysis.clustering import ClusteringConfig, Clustering
import re 
from logai.algorithms.clustering_algo.dbscan import DbScanParams

from logai.information_extraction.log_vectorizer import VectorizerConfig, LogVectorizer
from logai.algorithms.vectorization_algo.word2vec import Word2VecParams
from logai.information_extraction.feature_extractor import (
    FeatureExtractor,
    FeatureExtractorConfig,
)


def collate_all_files(data_dir):
    data = None
    for file in os.listdir(data_dir):
        data_i = pd.read_csv(os.path.join(data_dir, file))
        if data is None:
            data = data_i
        else:
            data = pd.concat([data, data_i])
    print("Collated all data")
    return data

def clean_and_truncate(preprocessed_loglines):
    preprocessed_loglines = preprocessed_loglines.apply(
        lambda x: x.replace("XX", "").replace("*", "").replace("\[[0-9., ]*\]", "")
    )
    preprocessed_loglines = preprocessed_loglines.apply(
        lambda x: " ".join(x[:500].split(" ")[:100]) if len(x) > 1000 else x
    )
    return preprocessed_loglines


def tokenize_logline(logline):
    logline_orig = logline
    logline = logline.replace('sfdc.common.messaging.','').replace('common.messaging.','')
    funcname = logline.split(' ')[0]
    logline = ' '.join(logline.split(' ')[1:]).replace(funcname, '').replace('*','')
    funcname_tokenized = re.sub(r'(?<!^)(?=[A-Z])', ' ', funcname).replace('.',' ').lower()
    logline =  re.sub(' +', ' ', funcname_tokenized+' '+logline)[:100].strip()
    #print (logline_orig)
    #print ('----->', logline,'\n')
    return logline 

def cluster_templates(vectorizer, feature_extractor, clustering, parsed_loglines):
    parsed_loglines = parsed_loglines.apply(lambda x: tokenize_logline(x))
    #parsed_loglines = parsed_loglines.apply(lambda x: ' '.join(x.split(' ')[1:]))
    vectorizer.fit(parsed_loglines)
    log_vectors_w2v = vectorizer.transform(parsed_loglines)
    _, feature_vector = feature_extractor.convert_to_feature_vector(
        log_vectors_w2v, attributes=None, timestamps=None
    )
    clustering.fit(feature_vector)
    dbscan_clusters = clustering.predict(feature_vector)
    return dbscan_clusters

def print_group(parsed_group_data, data_type, cluster_label):
    topk = 10
    parsed_group_unique_patterns = parsed_group_data['pattern'].head(topk)
    parsed_group_unique_patterns_counts = parsed_group_data['coverage'].head(topk)
    if topk < len(parsed_group_data):
        print ('top-'+str(topk)+' of unique '+data_type+' (out of '+str(len(parsed_group_data))+') in Cluster : ',cluster_label)
    else:
        print ('All unique '+data_type+' in Cluster : ',cluster_label)
    print ('\t'+'\n\t'.join([(x.replace('sfdc.common.messaging.','...')[:500]+' ...').strip()+'\t\t'+str(round(y*100, 3))+"%" for x,y in zip(list(parsed_group_unique_patterns), list(parsed_group_unique_patterns_counts))]))
    print ('\n')    
if __name__=="__main__":
    data_dir = '../../datasets/mq/'
    #incident_dir = 'na163_2022-03-04'
    #incident_dir = 'cs142_2022-03-09'
    #incident_dir = 'cs191_2022-03-17'
    #incident_dir = 'cs218_2022-03-20'
    incident_dir = 'na94_2022-03-23'
    #incident_dir = 'cs5_2022-03-26'
    #incident_dir = 'na128_2022-03-29'
    #incident_dir = 'na100_2022-03-29'
    log_record_type = 'mqdbg'

    data_dir = os.path.join(data_dir, log_record_type, incident_dir)

    incident_filepath = os.path.join(data_dir, "incident_data.csv")
    if not os.path.exists(incident_filepath):
        collated_data = collate_all_files(data_dir)
        collated_data.to_csv(incident_filepath)

    log_type = "csv"

    LOG_ANCHOR_FIELD_DBG = "logName"

    LOG_BODY_FIELD_DBG = "message"

    LOG_TIME_FIELD_DBG = "_time"

    dimensions = {
        "timestamp": [LOG_TIME_FIELD_DBG],
        "attributes": [LOG_ANCHOR_FIELD_DBG],  # messagetypename, wallclocktime
        "body": [LOG_BODY_FIELD_DBG],
    }

    config = DataLoaderConfig(
        filepath=incident_filepath, log_type="csv", dimensions=dimensions, header=0
    )

    preprocessor_config = PreprocessorConfig(custom_delimiters_regex=[r"`+|\s+"])
    parsing_algo_params = {"sim_th": 0.1, "extra_delimiters": []}
    parser_config = LogParserConfig()
    incident_dataloader = FileDataLoader(config)

    incident_logrecord = incident_dataloader.load_data()

    parsing_result = []
    non_qualified_clusters = []

    preprocessor = Preprocessor(preprocessor_config)
    preprocessed_loglines, _ = preprocessor.clean_log(
        incident_logrecord.body[constants.LOGLINE_NAME]
    )
    preprocessed_loglines = clean_and_truncate(preprocessed_loglines)
    index_groups = preprocessor.group_log_index(
        incident_logrecord.attributes, by=[LOG_ANCHOR_FIELD_DBG]
    )

    #algo_params = DbScanParams(min_samples=2)
    #clustering_config = ClusteringConfig('DBScan', algo_params, None)


    w2v_params = Word2VecParams()
    vectorizer_config = VectorizerConfig("word2vec", w2v_params, None)
    vectorizer = LogVectorizer(vectorizer_config)

    fe_config = FeatureExtractorConfig()
    feature_extractor = FeatureExtractor(fe_config)

    parser = LogParser(parser_config)

    for i in index_groups.index:
        funcname = index_groups[LOG_ANCHOR_FIELD_DBG].iloc[i]
        indices = index_groups["group_index"][i]
        if index_groups[LOG_ANCHOR_FIELD_DBG].iloc[i] == -1:
            continue
        if len(indices) == 1:
            continue
        loglines_in_group = preprocessed_loglines.iloc[indices]
        print ('\n-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
        print ('\n', funcname)

        parsed_result = parser.parse(loglines_in_group)
        uniq_patterns_counts = parsed_result[
            constants.PARSED_LOGLINE_NAME
        ].value_counts(normalize=True, sort=True, ascending=False)
        uniq_patterns = list(uniq_patterns_counts.index)
        uniq_patterns_counts = list(uniq_patterns_counts)
        num_patterns = len(uniq_patterns)

        algo_params = KMeansParams(n_clusters=min(num_patterns, 3))
        clustering_config = ClusteringConfig('kmeans', algo_params, None)

        clustering = Clustering(clustering_config)
        
        if num_patterns > 1:
            cluster_labels = cluster_templates(
                vectorizer, feature_extractor, clustering, pd.Series(uniq_patterns)
            )
        else:
            cluster_labels = [0] * num_patterns
        unique_parsed_result = pd.DataFrame(
            zip(cluster_labels, uniq_patterns, uniq_patterns_counts),
            columns=["cluster_labels", "pattern", "coverage"],
        )
        print("number of clusters:", len(set(cluster_labels)))
        for parsed_group in unique_parsed_result.groupby("cluster_labels"):
            parsed_group_label = parsed_group[0]
            cluster_label = parsed_group_label
            print ('\ncluster label: ', cluster_label)
            parsed_group_data = parsed_group[1]
            print_group(parsed_group_data, 'Templates',cluster_label)
            parsed_group_data = parsed_result[parsed_result[constants.PARSED_LOGLINE_NAME].isin(parsed_group_data['pattern'])]
            parsed_group_data[constants.PARAMETER_LIST_NAME] = parsed_group_data.apply(get_parameter_list, axis=1)
            parsed_group_data = parsed_group_data[constants.PARAMETER_LIST_NAME].value_counts(normalize=True, sort=True, ascending=False)
            parameters_list = pd.DataFrame(zip(list(parsed_group_data.index), list(parsed_group_data)), columns=['pattern', 'coverage'])
            print_group(parameters_list, 'Parameters',cluster_label)

