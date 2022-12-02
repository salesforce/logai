#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#

import pandas as pd
from logai.utils import constants
from logai.dataloader.data_loader import DataLoaderConfig, FileDataLoader
from logai.preprocess.preprocess import Preprocessor
from logai.information_extraction.log_parser import LogParser, LogParserConfig
from logai.preprocess.preprocess import PreprocessorConfig
import numpy as np 
import re
import os 
from logai.information_extraction.categorical_encoder import CategoricalEncoderConfig, CategoricalEncoder
from logai.information_extraction.feature_extractor import FeatureExtractor, FeatureExtractorConfig
from logai.algorithms.anomaly_detection_algo.local_outlier_factor import LOFParams
from logai.algorithms.anomaly_detection_algo.isolation_forest import IsolationForestParams
from logai.algorithms.anomaly_detection_algo.distribution_divergence import DistributionDivergenceParams
from logai.analysis.anomaly_detector import AnomalyDetectionConfig, AnomalyDetector


#from logai.applications.mq_log_analysis.log_print_utils import print_topk_most_anomalous, get_topk_most_anomalous
    


def clean_and_truncate(preprocessed_loglines):
    preprocessed_loglines = preprocessed_loglines.apply(lambda x: x.replace('XX','').replace('*','').replace('\[[0-9., ]*\]',''))
    preprocessed_loglines = preprocessed_loglines.apply(lambda x: ' '.join(x[:500].split(' ')[:100]) if len(x)>1000 else x)
    return preprocessed_loglines


def clean_and_truncate(preprocessed_loglines):
    preprocessed_loglines = preprocessed_loglines.apply(lambda x: x.replace('XX','').replace('*','').replace('\[[0-9., ]*\]',''))
    preprocessed_loglines = preprocessed_loglines.apply(lambda x: ' '.join(x[:500].split(' ')[:100]) if len(x)>1000 else x)
    return preprocessed_loglines
'''
def cluster_templates(vectorizer, feature_extractor, clustering, parsed_loglines):
    parsed_loglines = parsed_loglines.apply(lambda x: ' '.join(x.split(' ')[1:]))
    vectorizer.fit(parsed_loglines)
    log_vectors_w2v = vectorizer.transform(parsed_loglines)
    _, feature_vector = feature_extractor.convert_to_feature_vector(log_vectors_w2v, attributes=None, timestamps=None)
    clustering.fit(feature_vector)
    dbscan_clusters = clustering.predict(feature_vector)
    return dbscan_clusters


def print_group(parsed_group_data, data_type):
    topk = 30
    parsed_group_unique_patterns = parsed_group_data['pattern'].head(topk)
    parsed_group_unique_patterns_counts = parsed_group_data['coverage'].head(topk)
    print ('top-'+str(topk)+' of unique '+data_type+' (out of '+str(len(parsed_group_data))+') in Cluster : ')
    print ('\t'+'\n\t'.join([(x[:200]+' ...').strip()+'\t\t'+str(y) for x,y in zip(list(parsed_group_unique_patterns), list(parsed_group_unique_patterns_counts))]))
''' 


class UnstructuredLogAnomalyDetection:

    def __init__(self):

        self.LOG_ANCHOR_FIELD_DBG = 'logName'
        self.LOG_BODY_FIELD_DBG = 'message' 
        self.LOG_TIME_FIELD_DBG = '_time'

        self.group_by_time = "1s"

        self.num_bins_freq_count = 100 
        
        preprocessor_config = PreprocessorConfig(custom_delimiters_regex=[r"`+|\s+"])
        self.preprocessor = Preprocessor(preprocessor_config)

        parsing_algo_params = {'sim_th': 0.1, 'extra_delimiters': []}
        parser_config = LogParserConfig()#parsing_algo_params)
        self.parser = LogParser(parser_config)

        count_config = FeatureExtractorConfig(
            group_by_time=self.group_by_time,
            group_by_category=[constants.PARSED_LOGLINE_NAME]
        )

        self.feature_extractor = FeatureExtractor(count_config)

        self.num_lof_field = 'num_local_outlier'
        self.num_isf_field = 'num_isf_outlier'
        self.kl_div_field = 'kl_div'
        self.js_div_field = 'js_div'
        self.output_fields = [self.num_lof_field, self.num_isf_field, self.kl_div_field, self.js_div_field]
        
    def load_data(self, filepath):
        dimensions = {'timestamp': [self.LOG_TIME_FIELD_DBG],
                    'attributes': [self.LOG_ANCHOR_FIELD_DBG],  
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
            preprocessed_loglines, _ = self.preprocessor.clean_log(logrecord.body[constants.LOGLINE_NAME])
            preprocessed_loglines = clean_and_truncate(preprocessed_loglines)
            parsed_result = self.parser.parse(preprocessed_loglines)
            parsed_result[constants.PARSED_LOGLINE_NAME] = logrecord.attributes[self.LOG_ANCHOR_FIELD_DBG].astype(str) +" "+ parsed_result[constants.PARSED_LOGLINE_NAME].astype(str)
            parsed_result[self.LOG_ANCHOR_FIELD_DBG] = logrecord.attributes
            if parsed_output_filepath is not None:
                parsed_result.to_csv(parsed_output_filepath)
        else:
            parsed_result = pd.read_csv(parsed_output_filepath)
        return parsed_result


    def process_logs(self, logrecord, parsed_result):
        
        timestamps = pd.to_datetime(logrecord.timestamp[self.LOG_TIME_FIELD_DBG])
        counter_vector = self.feature_extractor.convert_to_counter_vector(
            log_pattern=parsed_result[constants.PARSED_LOGLINE_NAME],
            attributes=logrecord.attributes,
            timestamps=timestamps
        )

        counter_per_logname = {x[0]:list(x[1]['Counts']) for x in counter_vector.groupby(constants.PARSED_LOGLINE_NAME)}
        return counter_vector, counter_per_logname


    def anomaly_detection_with_freq_count_per_msgtype(self, incident_counts, incident_counts_msgtype, reference_counts, reference_counts_msgtype, stats_per_msgtype):

        num_incident_timechunks = max([len(x) for x in incident_counts_msgtype.values()])
        num_reference_timechunks = max([len(x) for x in reference_counts_msgtype.values()])

        incident_counts_msgtype = {k:v+[0]*(num_incident_timechunks - len(v)) for k,v in incident_counts_msgtype.items()}
        reference_counts_msgtype = {k:v+[0]*(num_reference_timechunks - len(v)) for k,v in reference_counts_msgtype.items()}

        lof_params = LOFParams()
        lof_params.contamination = 0.001
        lof_params.n_neighbors = 20
        lof_params.novelty = True 
        all_msgtypes = set(incident_counts_msgtype).union(set(reference_counts_msgtype))
        ad_config = AnomalyDetectionConfig(algo_name='lof', algo_params=lof_params)
        anomaly_detector = AnomalyDetector(ad_config)

        all_counts = []
        all_counts.extend(list(incident_counts['Counts']))
        all_counts.extend(list(reference_counts['Counts']))
        max_freq_counts_value = max(all_counts)
        range_of_count_values = range(0, max_freq_counts_value+1)
        _, freq_count_bins = np.histogram(np.array(range_of_count_values), bins= self.num_bins_freq_count)

        distdiv_params = DistributionDivergenceParams(n_bins=freq_count_bins, type=["KL", "JS"])
        ad_config = AnomalyDetectionConfig(algo_name='distribution_divergence', algo_params=distdiv_params)
        anomaly_ranker = AnomalyDetector(ad_config)

        for msgtype in all_msgtypes:
            if msgtype not in reference_counts_msgtype or len(reference_counts_msgtype[msgtype]) < 2:
                continue

            if msgtype not in incident_counts_msgtype:
                incident_counts_msgtype[msgtype] = [0]*num_incident_timechunks

            anomaly_detector.fit(pd.DataFrame(reference_counts_msgtype[msgtype]))
            lof_prediction = anomaly_detector.predict(pd.DataFrame(incident_counts_msgtype[msgtype]))
            num_anomalies = len(lof_prediction[lof_prediction==-1])

            anomaly_ranker.fit(pd.DataFrame(reference_counts_msgtype[msgtype]))
            kl_div, js_div = anomaly_ranker.predict(pd.DataFrame(incident_counts_msgtype[msgtype]))

            if msgtype not in stats_per_msgtype:
                stats_per_msgtype[msgtype] = {k:{} for k in self.output_fields}

            stats_per_msgtype[msgtype][self.num_lof_field]['freq_count'] = num_anomalies
            stats_per_msgtype[msgtype][self.kl_div_field]['freq_count'] = kl_div
            stats_per_msgtype[msgtype][self.js_div_field]['freq_count'] = js_div

        return stats_per_msgtype

    def joint_anomaly_detection_with_freq_count_all_msgtypes(self, incident_counts, reference_counts, stats_per_msgtype):

        encoder_config = CategoricalEncoderConfig()
        encoder = CategoricalEncoder(encoder_config)

        print (incident_counts[constants.PARSED_LOGLINE_NAME][0])

        incident_counts = incident_counts[[constants.PARSED_LOGLINE_NAME, 'Counts']]
        reference_counts = reference_counts[[constants.PARSED_LOGLINE_NAME, 'Counts']]

        incident_lognames = incident_counts[constants.PARSED_LOGLINE_NAME].apply(lambda x: x.split(' ')[0])
        reference_lognames = reference_counts[constants.PARSED_LOGLINE_NAME].apply(lambda x: x.split(' ')[0])
        incident_counts[constants.PARSED_LOGLINE_NAME] = encoder.fit_transform(incident_counts)[constants.PARSED_LOGLINE_NAME+'_categorical']
        reference_counts[constants.PARSED_LOGLINE_NAME] = encoder.fit_transform(reference_counts)[constants.PARSED_LOGLINE_NAME+'_categorical']


        isf_params = IsolationForestParams()
        ad_config = AnomalyDetectionConfig(algo_name='isolation_forest', algo_params=isf_params)
        anomaly_detector = AnomalyDetector(ad_config)

        anomaly_detector.fit(reference_counts)
        anomaly_scores_counts_isf = anomaly_detector.predict(incident_counts)

        incident_counts['isf_anomaly_scores'] = (anomaly_scores_counts_isf==-1).astype('float')
        incident_counts[self.LOG_ANCHOR_FIELD_DBG] = incident_lognames
        reference_counts[self.LOG_ANCHOR_FIELD_DBG] = reference_lognames
        incident_isf_scores_per_msgtype = {x[0]:sum(list(x[1]['isf_anomaly_scores'])) for x in incident_counts.groupby(self.LOG_ANCHOR_FIELD_DBG)}
        
        for msgtype, num_anomalies in incident_isf_scores_per_msgtype.items():
            if msgtype not in stats_per_msgtype:
                stats_per_msgtype[msgtype] = {k:{} for k in self.output_fields}
            stats_per_msgtype[msgtype][self.num_isf_field]['freq_count'] = num_anomalies
        return stats_per_msgtype


    def unstructured_log_anomaly_detection(self, incident_filepath, reference_filepath, stats_per_msgtype={}):

        parsed_incident_filepath = incident_filepath.replace('.csv', '_parsed.csv')
        parsed_reference_filepath = reference_filepath.replace('.csv', '_parsed.csv')
        incident_logrecord = self.load_data(incident_filepath)
        reference_logrecord = self.load_data(reference_filepath)

        incident_parsed_result = self.parse_logs(incident_logrecord, parsed_incident_filepath)
        reference_parsed_result = self.parse_logs(reference_logrecord, parsed_reference_filepath)
        
        incident_counts, incident_counter_per_logname = self.process_logs(incident_logrecord, incident_parsed_result)
        reference_counts, reference_counter_per_logname = self.process_logs(reference_logrecord, reference_parsed_result)
            
        stats_per_msgtype = self.anomaly_detection_with_freq_count_per_msgtype(incident_counts, incident_counter_per_logname, reference_counts, reference_counter_per_logname, stats_per_msgtype)
        self.joint_anomaly_detection_with_freq_count_all_msgtypes(incident_counts, reference_counts, stats_per_msgtype)

        return stats_per_msgtype

