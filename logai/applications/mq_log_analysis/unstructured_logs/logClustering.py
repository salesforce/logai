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
from logai.information_extraction.feature_extractor import FeatureExtractor, FeatureExtractorConfig

from logai.applications.mq_log_analysis.unstructured_logs.log_print_utils import print_log_cluster, cluster_tsne, print_func_distribution
from logai.applications.mq_log_analysis.unstructured_logs.log_utils import clean_and_truncate, tokenize_logline
    



class LogClustering:

    def __init__(self, num_clusters, visualize_clusters=False):
        self.LOG_ANCHOR_FIELD_DBG = 'logName'
        self.LOG_BODY_FIELD_DBG = 'message' 
        self.LOG_TIME_FIELD_DBG = '_time'
        
        algo_params = KMeansParams(n_clusters=num_clusters)
        clustering_config = ClusteringConfig('kmeans', algo_params, None)

        self.clustering = Clustering(clustering_config)

        w2v_params = Word2VecParams()
        vectorizer_config = VectorizerConfig("word2vec", w2v_params, None)
        self.vectorizer = LogVectorizer(vectorizer_config)
        
        fe_config = FeatureExtractorConfig()
        self.feature_extractor = FeatureExtractor(fe_config)

        self.visualize_clusters = visualize_clusters


    def load_data(self, filepath):
        dimensions = {'timestamp': [self.LOG_TIME_FIELD_DBG],
                    'attributes': [self.LOG_ANCHOR_FIELD_DBG],  # messagetypename, wallclocktime
                    'body': [self.LOG_BODY_FIELD_DBG]}

        config = DataLoaderConfig(
            filepath=filepath,
            log_type='csv',
            dimensions=dimensions,
            header=0
        )

        dataloader = FileDataLoader(config)
        logrecord = dataloader.load_data()
        return logrecord 



    def parse_logs(self, logrecord, parsed_output_filepath=None):
        if parsed_output_filepath is None or not os.path.exists(parsed_output_filepath):
            preprocessor_config = PreprocessorConfig(custom_delimiters_regex=[r"`+|\s+"])
            preprocessor = Preprocessor(preprocessor_config)
            preprocessed_loglines, _ = preprocessor.clean_log(logrecord.body[constants.LOGLINE_NAME])
            preprocessed_loglines = clean_and_truncate(preprocessed_loglines)

            parsing_algo_params = {'sim_th': 0.1, 'extra_delimiters': []}
            parser_config = LogParserConfig()
            parser = LogParser(parser_config)
            parsed_result = parser.parse(preprocessed_loglines)
            parsed_result[constants.PARSED_LOGLINE_NAME] = logrecord.attributes[self.LOG_ANCHOR_FIELD_DBG].astype(str) +" "+ parsed_result[constants.PARSED_LOGLINE_NAME].astype(str)
            if parsed_output_filepath is not None:
                parsed_result.to_csv(parsed_output_filepath)
        else:
            parsed_result = pd.read_csv(parsed_output_filepath)
        return parsed_result

    def cluster_templates(self, parsed_loglines):
        parsed_loglines = parsed_loglines.apply(lambda x: tokenize_logline(x))
        #parsed_loglines = parsed_loglines.apply(lambda x: ' '.join(x.split(' ')[1:]))
        self.vectorizer.fit(parsed_loglines)
        log_vectors_w2v = self.vectorizer.transform(parsed_loglines)
        _, feature_vector = self.feature_extractor.convert_to_feature_vector(log_vectors_w2v, attributes=None, timestamps=None)
        self.clustering.fit(feature_vector)
        clusters = self.clustering.predict(feature_vector)
        return clusters, feature_vector

    def get_log_clusters(self, parsed_result):
        uniq_patterns_counts = parsed_result[constants.PARSED_LOGLINE_NAME].value_counts(normalize=True, sort=True, ascending=False)
        uniq_patterns = list(uniq_patterns_counts.index)
        uniq_patterns_counts = list(uniq_patterns_counts)

        cluster_labels, feature_vector = self.cluster_templates(pd.Series(uniq_patterns))
        unique_parsed_result = pd.DataFrame(zip(cluster_labels, uniq_patterns, uniq_patterns_counts), columns=['cluster_labels', 'pattern', 'coverage'])
        print ('number of clusters:', len(set(cluster_labels)))

        clusterwise_template_param_data = {}
        for parsed_group in unique_parsed_result.groupby('cluster_labels'):
            parsed_group_label = parsed_group[0]
            cluster_label = parsed_group_label + 1
            print ('\ncluster label: ', cluster_label)
            parsed_group_data = parsed_group[1]
            clusterwise_template_param_data[cluster_label] = {'Templates': parsed_group_data}
            print_func_distribution(parsed_group_data)
            #print_log_cluster(parsed_group_data, 'Templates')
            parsed_group_data = parsed_result[parsed_result[constants.PARSED_LOGLINE_NAME].isin(parsed_group_data['pattern'])]
            parsed_group_data = parsed_group_data[constants.PARAMETER_LIST_NAME].value_counts(normalize=True, sort=True, ascending=False)
            parameters_list = pd.DataFrame(zip(list(parsed_group_data.index), list(parsed_group_data)), columns=['pattern', 'coverage'])
            #print_log_cluster(parameters_list, 'Parameters')
            clusterwise_template_param_data[cluster_label].update({'Parameters': parameters_list})
        return clusterwise_template_param_data, cluster_labels, feature_vector


    def log_clustering(self, filepath):
        parsed_filepath = filepath.replace('.csv', '_parsed.csv')
        logrecord = self.load_data(filepath)
        parsed_result = self.parse_logs(logrecord, parsed_output_filepath=parsed_filepath)
        clusterwise_template_param_data, cluster_labels, feature_vector = self.get_log_clusters(parsed_result)
        if self.visualize_clusters:
            cluster_tsne(feature_vector=feature_vector, clusters=cluster_labels)
        return clusterwise_template_param_data

    def print_cluster_interactive(self, clusterwise_template_param_data):
        while True:
            cluster_label = input('\nEnter Cluster Label you want to see Template & Parameter distribution of: ')
            print ('\n------------------------------------------------------------------------------------------------------------------\n')
            if len(cluster_label)==0:
                return None
            try:
                cluster_label = int(cluster_label)
            except:
                break 
            for k,v in clusterwise_template_param_data[cluster_label].items():
                print_log_cluster(v, k, cluster_label)
            print ('\n------------------------------------------------------------------------------------------------------------------\n')
    




class LogClusteringPerFunction():

    def __init__(self):
        self.LOG_ANCHOR_FIELD_DBG = 'logName'
        self.LOG_BODY_FIELD_DBG = 'message' 
        self.LOG_TIME_FIELD_DBG = '_time'

        preprocessor_config = PreprocessorConfig(custom_delimiters_regex=[r"`+|\s+"])
        self.preprocessor = Preprocessor(preprocessor_config)
        
        parsing_algo_params = {'sim_th': 0.1, 'extra_delimiters': []}
        parser_config = LogParserConfig()
        self.parser = LogParser(parser_config)

        w2v_params = Word2VecParams()
        vectorizer_config = VectorizerConfig("word2vec", w2v_params, None)
        self.vectorizer = LogVectorizer(vectorizer_config)
        
        fe_config = FeatureExtractorConfig()
        self.feature_extractor = FeatureExtractor(fe_config)


    def load_data(self, filepath):
        dimensions = {'timestamp': [self.LOG_TIME_FIELD_DBG],
                    'attributes': [self.LOG_ANCHOR_FIELD_DBG],  # messagetypename, wallclocktime
                    'body': [self.LOG_BODY_FIELD_DBG]}

        config = DataLoaderConfig(
            filepath=filepath,
            log_type='csv',
            dimensions=dimensions,
            header=0
        )
        dataloader = FileDataLoader(config)
        logrecord = dataloader.load_data()

        return logrecord

    def process_logs(self, logrecord):
        preprocessed_loglines, _ = self.preprocessor.clean_log(logrecord.body[constants.LOGLINE_NAME])
        preprocessed_loglines = clean_and_truncate(preprocessed_loglines)
        index_groups = self.preprocessor.group_log_index(logrecord.attributes, by=[self.LOG_ANCHOR_FIELD_DBG])
        return preprocessed_loglines, index_groups

    def cluster_templates(self, num_patterns, parsed_loglines):

        algo_params = KMeansParams(n_clusters=min(num_patterns, 3))
        clustering_config = ClusteringConfig('kmeans', algo_params, None)
        clustering = Clustering(clustering_config)

        parsed_loglines = parsed_loglines.apply(lambda x: tokenize_logline(x))
        #parsed_loglines = parsed_loglines.apply(lambda x: ' '.join(x.split(' ')[1:]))
        self.vectorizer.fit(parsed_loglines)
        log_vectors_w2v = self.vectorizer.transform(parsed_loglines)
        _, feature_vector = self.feature_extractor.convert_to_feature_vector(log_vectors_w2v, attributes=None, timestamps=None)
        clustering.fit(feature_vector)
        clusters = clustering.predict(feature_vector)
        return clusters

    def get_log_clusters(self, parsed_result):
        uniq_patterns_counts = parsed_result[constants.PARSED_LOGLINE_NAME].value_counts(normalize=True, sort=True, ascending=False)
        uniq_patterns = list(uniq_patterns_counts.index)
        uniq_patterns_counts = list(uniq_patterns_counts)
        num_patterns = len(uniq_patterns)

        if num_patterns > 1:
            cluster_labels = self.cluster_templates(num_patterns, pd.Series(uniq_patterns))
        else:
            cluster_labels = [0]*num_patterns
        unique_parsed_result = pd.DataFrame(zip(cluster_labels, uniq_patterns, uniq_patterns_counts), columns=['cluster_labels', 'pattern', 'coverage'])
        print ('number of clusters:', len(set(cluster_labels)))
        for parsed_group in unique_parsed_result.groupby('cluster_labels'):
            parsed_group_label = parsed_group[0]
            cluster_label = parsed_group_label
            print ('\ncluster label: ', cluster_label)
            parsed_group_data = parsed_group[1]
            print_log_cluster(parsed_group_data, 'Templates',cluster_label)
            parsed_group_data = parsed_result[parsed_result[constants.PARSED_LOGLINE_NAME].isin(parsed_group_data['pattern'])]
            parsed_group_data[constants.PARAMETER_LIST_NAME] = parsed_group_data.apply(get_parameter_list, axis=1)
            parsed_group_data = parsed_group_data[constants.PARAMETER_LIST_NAME].value_counts(normalize=True, sort=True, ascending=False)
            parameters_list = pd.DataFrame(zip(list(parsed_group_data.index), list(parsed_group_data)), columns=['pattern', 'coverage'])
            print_log_cluster(parameters_list, 'Parameters',cluster_label)

    def log_clustering(self, filepath):
        logrecord = self.load_data(filepath)
        preprocessed_loglines, index_groups = self.process_logs(logrecord)
        for i in index_groups.index:
            funcname = index_groups[self.LOG_ANCHOR_FIELD_DBG].iloc[i]
            indices = index_groups['group_index'][i]
            if index_groups[self.LOG_ANCHOR_FIELD_DBG].iloc[i] == -1:
                continue
            if len(indices) == 1:
                continue
            loglines_in_group = preprocessed_loglines.iloc[indices]
            print ('\n-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')
            print ('\n', funcname)
            parsed_result = self.parser.parse(loglines_in_group)
            self.get_log_clusters(parsed_result)