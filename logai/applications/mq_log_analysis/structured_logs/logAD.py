#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from logai.dataloader.data_loader import FileDataLoader
from logai.dataloader.data_loader import DataLoaderConfig
from logai.information_extraction.feature_extractor import FeatureExtractor, FeatureExtractorConfig
from logai.algorithms.anomaly_detection_algo.local_outlier_factor import LOFParams
from logai.algorithms.anomaly_detection_algo.isolation_forest import IsolationForestParams
from logai.algorithms.anomaly_detection_algo.distribution_divergence import DistributionDivergenceParams
from logai.information_extraction.categorical_encoder import CategoricalEncoderConfig, CategoricalEncoder
from logai.analysis.anomaly_detector import AnomalyDetectionConfig, AnomalyDetector
import pandas as pd 
import numpy as np
import os 



class StructuredLogAnomalyDetection:
    def __init__(self, anomaly_detection_by_metric_or_freq=None, log_metric=None):
        if anomaly_detection_by_metric_or_freq == 'metric':
            if not log_metric:
                raise Exception('Log metric argument should be mentioned')

        self.anomaly_detection_by_metric_or_freq = anomaly_detection_by_metric_or_freq
        self.log_msgtype = 'MessageTypeName'
        self.log_metric = log_metric
        self.log_datetime_field = '_time'
        if self.log_metric is not None:
            self.log_attribute_fields = [self.log_msgtype, self.log_metric]
        else:
            self.log_attribute_fields = [self.log_msgtype]
        self.num_lof_field = 'num_local_outlier'
        self.num_isf_field = 'num_isf_outlier'
        self.kl_div_field = 'kl_div'
        self.js_div_field = 'js_div'
        self.output_fields = [self.num_lof_field, self.num_isf_field, self.kl_div_field, self.js_div_field]

        if self.anomaly_detection_by_metric_or_freq=='freq_count':
            self.count_partitioning_config = FeatureExtractorConfig(
                group_by_time='1min',
                group_by_category=[self.log_msgtype]  #messagetype
            )
            self.count_feature_extractor = FeatureExtractor(self.count_partitioning_config)

        if self.anomaly_detection_by_metric_or_freq=='metric':
            self.metric_partitioning_config = FeatureExtractorConfig(
                group_by_category= self.log_attribute_fields #messagetype
            )
            self.metric_feature_extractor = FeatureExtractor(self.metric_partitioning_config)


    def structured_log_anomaly_detection(self, incident_filepath, reference_filepath, stats_per_msgtype={}):
        incident_logrecord = self.load_data(incident_filepath)
        reference_logrecord = self.load_data(reference_filepath)
        if self.anomaly_detection_by_metric_or_freq=='freq_count':
            stats_per_msgtype = self.anomaly_detection_with_freq_count_per_msgtype(incident_logrecord, reference_logrecord, stats_per_msgtype)
            stats_per_msgtype = self.joint_anomaly_detection_with_freq_count_all_msgtypes(incident_logrecord, reference_logrecord, stats_per_msgtype)
        elif self.anomaly_detection_by_metric_or_freq=='metric':
            stats_per_msgtype = self.anomaly_detection_with_metric_per_msgtype(incident_logrecord, reference_logrecord, stats_per_msgtype)
            stats_per_msgtype = self.joint_anomaly_detection_with_metric_all_msgtype(incident_logrecord, reference_logrecord, stats_per_msgtype)
        return stats_per_msgtype


    def load_data(self, filepath):
        dimensions = {'timestamp': [self.log_datetime_field],
                    'attributes': self.log_attribute_fields,  # messagetypename, wallclocktime
                    'body': []}

        file_config = DataLoaderConfig(
            filepath=filepath,
            log_type='csv',
            dimensions=dimensions,
            header=0
        )

        dataloader = FileDataLoader(file_config)
        logrecord = dataloader.load_data()
        return logrecord 


    def anomaly_detection_with_freq_count_per_msgtype(self, incident_logrecord, reference_logrecord, stats_per_msgtype):
        
        incident_attributes = incident_logrecord.attributes 
        incident_timestamps = pd.to_datetime(incident_logrecord.timestamp[self.log_datetime_field])

        reference_attributes = reference_logrecord.attributes 
        reference_timestamps = pd.to_datetime(reference_logrecord.timestamp[self.log_datetime_field])

        incident_counts = self.count_feature_extractor.convert_to_counter_vector(log_pattern=None, attributes=incident_attributes, timestamps=incident_timestamps)
        reference_counts = self.count_feature_extractor.convert_to_counter_vector(log_pattern=None, attributes=reference_attributes, timestamps=reference_timestamps)

        incident_counts_msgtype = {x[0]:list(x[1]['Counts']) for x in incident_counts.groupby(self.log_msgtype)}
        reference_counts_msgtype = {x[0]:list(x[1]['Counts']) for x in reference_counts.groupby(self.log_msgtype)}

        num_incident_timechunks = max([len(x) for x in incident_counts_msgtype.values()])
        num_reference_timechunks = max([len(x) for x in reference_counts_msgtype.values()])

        incident_counts_msgtype = {k:v+[0]*(num_incident_timechunks - len(v)) for k,v in incident_counts_msgtype.items()}
        reference_counts_msgtype = {k:v+[0]*(num_reference_timechunks - len(v)) for k,v in reference_counts_msgtype.items()}

        lof_params = LOFParams()
        lof_params.contamination = 0.001
        lof_params.n_neighbors = 20
        lof_params.novelty = True 
        ad_config = AnomalyDetectionConfig(algo_name='lof', algo_params=lof_params)
        anomaly_detector = AnomalyDetector(ad_config)


        num_bins_freq_count = 100

        all_counts = []
        all_counts.extend(list(incident_counts['Counts']))
        all_counts.extend(list(reference_counts['Counts']))
        max_freq_counts_value = max(all_counts)
        range_of_count_values = range(0, max_freq_counts_value+1)
        _, freq_count_bins = np.histogram(np.array(range_of_count_values), bins= num_bins_freq_count)

        distdiv_params = DistributionDivergenceParams(n_bins=freq_count_bins, type=["KL", "JS"])
        ad_config = AnomalyDetectionConfig(algo_name='distribution_divergence', algo_params=distdiv_params)
        anomaly_ranker = AnomalyDetector(ad_config)


        all_msgtypes = set(incident_counts_msgtype).union(set(reference_counts_msgtype))

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

    def anomaly_detection_with_metric_per_msgtype(self, incident_logrecord, reference_logrecord, stats_per_msgtype):

        incident_attributes = incident_logrecord.attributes 
        incident_timestamps = pd.to_datetime(incident_logrecord.timestamp[self.log_datetime_field])

        reference_attributes = reference_logrecord.attributes 
        reference_timestamps = pd.to_datetime(reference_logrecord.timestamp[self.log_datetime_field])

        _, incident_metrics = self.metric_feature_extractor.convert_to_feature_vector(None, attributes=incident_attributes, timestamps=incident_timestamps)
        _, reference_metrics = self.metric_feature_extractor.convert_to_feature_vector(None, attributes=reference_attributes, timestamps=reference_timestamps)

        incident_metrics_count = self.metric_feature_extractor.convert_to_counter_vector(None, attributes=incident_attributes, timestamps=incident_timestamps)
        reference_metrics_count = self.metric_feature_extractor.convert_to_counter_vector(None, attributes=reference_attributes, timestamps=reference_timestamps)


        incident_metrics_count_msgtype = {x[0]:list(x[1]['Counts']) for x in incident_metrics_count.groupby(self.log_msgtype)}
        reference_metrics_count_msgtype = {x[0]:list(x[1]['Counts']) for x in reference_metrics_count.groupby(self.log_msgtype)}

        incident_metrics_msgtype = {x[0]:list(x[1][self.log_metric]) for x in incident_metrics.groupby(self.log_msgtype)}
        reference_metrics_msgtype = {x[0]:list(x[1][self.log_metric]) for x in reference_metrics.groupby(self.log_msgtype)}


        all_msgtypes = set(incident_metrics_count_msgtype).union(set(reference_metrics_count_msgtype))

        lof_params = LOFParams()
        lof_params.contamination = 0.001
        lof_params.n_neighbors = 20
        lof_params.novelty = True 
        ad_config = AnomalyDetectionConfig(algo_name='lof', algo_params=lof_params)
        anomaly_detector = AnomalyDetector(ad_config)

        num_bins_metric_value = 1000 
        all_metrics = []
        all_metrics.extend(list(reference_metrics[self.log_metric]))
        all_metrics.extend(list(incident_metrics[self.log_metric]))
        all_metrics = np.array(all_metrics)
        _, data_bins = np.histogram(all_metrics, bins=num_bins_metric_value)

        distdiv_params = DistributionDivergenceParams(n_bins=data_bins, type=["KL", "JS"])
        ad_config = AnomalyDetectionConfig(algo_name='distribution_divergence', algo_params=distdiv_params)
        anomaly_ranker = AnomalyDetector(ad_config)

        for msgtype in all_msgtypes:
            if msgtype not in incident_metrics_msgtype or msgtype not in reference_metrics_msgtype:
                continue

            incident_metric_vals = incident_metrics_msgtype[msgtype]
            incident_metric_freqs = incident_metrics_count_msgtype[msgtype]
            incident_metric_data = pd.DataFrame(np.repeat(incident_metric_vals, incident_metric_freqs), columns=['data'])

            reference_metric_vals = reference_metrics_msgtype[msgtype]
            reference_metric_freqs = reference_metrics_count_msgtype[msgtype]
            reference_metric_data = pd.DataFrame(np.repeat(reference_metric_vals, reference_metric_freqs), columns=['data'])

            if len(reference_metric_data) < 2:
                continue
            anomaly_detector.fit(reference_metric_data)

            lof_prediction = anomaly_detector.predict(incident_metric_data)
            num_anomalies = len(lof_prediction[lof_prediction==-1])
            anomaly_ranker.fit(reference_metric_data)
            kl_div, js_div = anomaly_ranker.predict(incident_metric_data)

            if msgtype not in stats_per_msgtype:
                stats_per_msgtype[msgtype] = {k:{} for k in self.output_fields}
                
            stats_per_msgtype[msgtype][self.num_lof_field][self.log_metric] = num_anomalies
            stats_per_msgtype[msgtype][self.kl_div_field][self.log_metric] = kl_div
            stats_per_msgtype[msgtype][self.js_div_field][self.log_metric] = js_div

        return stats_per_msgtype


    def joint_anomaly_detection_with_freq_count_all_msgtypes(self, incident_logrecord, reference_logrecord, stats_per_msgtype):

        encoder_config = CategoricalEncoderConfig()
        encoder = CategoricalEncoder(encoder_config)

        incident_attributes = incident_logrecord.attributes 
        incident_timestamps = pd.to_datetime(incident_logrecord.timestamp[self.log_datetime_field])

        reference_attributes = reference_logrecord.attributes 
        reference_timestamps = pd.to_datetime(reference_logrecord.timestamp[self.log_datetime_field])

        incident_counts = self.count_feature_extractor.convert_to_counter_vector(log_pattern=None, attributes=incident_attributes, timestamps=incident_timestamps)
        reference_counts = self.count_feature_extractor.convert_to_counter_vector(log_pattern=None, attributes=reference_attributes, timestamps=reference_timestamps)

        incident_counts = incident_counts[[self.log_msgtype, 'Counts']]
        reference_counts = reference_counts[[self.log_msgtype, 'Counts']]

        print ('incident_counts: ', incident_counts)

        incident_counts_msgtype = incident_counts[self.log_msgtype]
        incident_counts[self.log_msgtype] = encoder.fit_transform(incident_counts)[self.log_msgtype+'_categorical']
        reference_counts[self.log_msgtype] = encoder.fit_transform(reference_counts)[self.log_msgtype+'_categorical']


        isf_params = IsolationForestParams()
        ad_config = AnomalyDetectionConfig(algo_name='isolation_forest', algo_params=isf_params)
        anomaly_detector = AnomalyDetector(ad_config)

        anomaly_detector.fit(reference_counts)
        anomaly_scores_counts_isf = anomaly_detector.predict(incident_counts)

        incident_counts['isf_anomaly_scores'] = (anomaly_scores_counts_isf==-1).astype('float')
        incident_counts[self.log_msgtype] = incident_counts_msgtype
        incident_isf_scores_per_msgtype = {x[0]:sum(list(x[1]['isf_anomaly_scores'])) for x in incident_counts.groupby(self.log_msgtype)}
        
        for msgtype, num_anomalies in incident_isf_scores_per_msgtype.items():
            if msgtype not in stats_per_msgtype:
                stats_per_msgtype[msgtype] = {k:{} for k in self.output_fields}
            stats_per_msgtype[msgtype][self.num_isf_field]['freq_count'] = num_anomalies
        return stats_per_msgtype


    def joint_anomaly_detection_with_metric_all_msgtype(self, incident_logrecord, reference_logrecord, stats_per_msgtype):

        incident_attributes = incident_logrecord.attributes 
        incident_timestamps = pd.to_datetime(incident_logrecord.timestamp[self.log_datetime_field])

        reference_attributes = reference_logrecord.attributes 
        reference_timestamps = pd.to_datetime(reference_logrecord.timestamp[self.log_datetime_field])

        incident_metrics_count = self.metric_feature_extractor.convert_to_counter_vector(None, attributes=incident_attributes, timestamps=incident_timestamps)
        reference_metrics_count = self.metric_feature_extractor.convert_to_counter_vector(None, attributes=reference_attributes, timestamps=reference_timestamps)

        encoder_config = CategoricalEncoderConfig()
        encoder = CategoricalEncoder(encoder_config)

        incident_counts_msgtype = incident_metrics_count[self.log_msgtype]

        incident_metrics_count[self.log_msgtype] = encoder.fit_transform(incident_metrics_count)[self.log_msgtype+'_categorical']
        reference_metrics_count[self.log_msgtype] = encoder.fit_transform(reference_metrics_count)[self.log_msgtype+'_categorical']

       

        isf_params = IsolationForestParams()
        ad_config = AnomalyDetectionConfig(algo_name='isolation_forest', algo_params=isf_params)
        anomaly_detector = AnomalyDetector(ad_config)

        anomaly_detector.fit(reference_metrics_count)
        anomaly_scores_isf = anomaly_detector.predict(incident_metrics_count)

        incident_metrics_count['isf_anomaly_scores'] = (anomaly_scores_isf==-1).astype('float')
        incident_metrics_count[self.log_msgtype] = incident_counts_msgtype
        incident_isf_scores_per_msgtype = {x[0]:sum(list(x[1]['isf_anomaly_scores'])) for x in incident_metrics_count.groupby(self.log_msgtype)}
        
        for msgtype, num_anomalies in incident_isf_scores_per_msgtype.items():
            if msgtype not in stats_per_msgtype:
                stats_per_msgtype[msgtype] = {k:{} for k in self.output_fields}
            stats_per_msgtype[msgtype][self.num_isf_field][self.log_metric] = num_anomalies
        return stats_per_msgtype


        
