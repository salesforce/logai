<!--
Copyright (c) 2023 Salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

-->
# Build LogAI Application in Jupyter Notebook

Here we provide examples to use Jupyter notebook to do interactive log analysis using LogAI.

### Create Log Record Object

Log-AI defines `LogRecordObject` as the data model for log process. `LogRecordObject` data model follows [OpenTelemetry Log and Event Record Definition](https://github.com/open-telemetry/opentelemetry-specification/blob/main/specification/logs/data-model.md#log-and-event-record-definition). 
In addition, an optional field `labels` is used to host labels for log analytics applications, such as log anomaly detection or log clustering. 

LogAI provides data loaders to consume logs from different sources. 
* `FileDataLoader` implements methods to consume local log files, including character separated values file formats (`.csv` or `.tsv`) 
and plain text file formats (`.txt` and `.log`). 
* [TBA] `DefaultDataLoader` implements connectors to consum logs from common log platforms (Splunk, DataDog, Prometheus, etc.).

Rather than using data loaders, you can also create `LogRecordObject` directly by assigning data object to each field.

### Preprocess

Log-AI implements `preprocess` module to conduct log preprocessing tasks, such as identify and replace special 
delimeter characters, identify and remove special string patterns to clean the logs, identify timestamps in the logs if missing, etc.
preprocessing is optional and with no ML process.

```python
# Preprocess
# Do customer rules to initially parse the loglines. Add custom delimeters in a regex
# Group log records by any attributes. Return grouped log index so follow up process can handle them separately.

custom_delimeter_regex = r"`+|\s+"
preprocessed_loglines = Preprocess.clean_log(logrecord.body, custom_delimeter_regex)

```

### Information Extraction

Information extraction layer transforms loglines to log features that can be consumed by machine learning models.
Inforamtion extraction layer contains three modules: log parsing, log vectorizaztion and log featurization.

#### Log Parsing

Log-AI integrates common log parsing algorithms (currently only DRAIN). You can create a log parser and call specific algorithms
in parser config.

```python
# Log Parsing
# Parse logs using DRAIN

# Parser parameters. This is algorithm specific. Default values will be used if not specific.
drain_params = DrainParams()

# Parser configuration. Specify which parsing algorithm to use.
log_parser_config = LogParserConfig(
    parsing_algorithm='drain',
    parsing_algo_params=drain_params
)

parser = LogParser(log_parser_config)
parsed_result = parser.parse(preprocessed_loglines)
```

#### Log Vectorization

Log vectorization transforms log tokens to a numeric vector that conveys the information to represent
this logline. Below is a very simple way to use Word2Vec for log vectorization. 

```python
## Vectorization
# Vectorization using Word2Vec

params = Word2VecParams()
vectorizer_config = VectorizerConfig("word2vec", params, None)
vectorizor = LogVectorizer(vectorizer_config)
vectorizor.fit(parsed_loglines)

#Log vector is a pandas.Series
log_vectors_w2v = vectorizor.transform(parsed_loglines)

```

#### Log Featurization

The main difference between log featurization and vectorization is log vectorization is performed on individual loglines
while featurization is performed on a log group. A log group here is a group of log records, and each log records
contains the unstructured portion (logline) and the structured portion (log attributes). Log featurization
transforms all information in this log group into a single feature vector. Here we define three different types of feature vectors:
* **Log event value vector**. Summarize feature values (numerical or categorical) of all log records in the same log group. 
Log event value vector is a tabular representation of a log group and can be used to used in general machine learning with tagbular data.
* **Log event sequence**. Concatenating all log record events linearly to form an event sequence. Log even sequence can be used to 
perform sequence modeling, such as LSTM, DeepAR, etc.
* **Log event counter**. Count occurrence of each log group in given time interval and form time series. Log event counter is suitable for
time-series based machine learning techniques.

Below is a simple example of creating log event value vector for a log group.

```python
#Feature extraction
# implement log vector to feature
# this will convert the vector metrics to a n dimensional feature.
# implement simple 0 padding method.

config = FeatureExtractorConfig(
    group_by_time="15min",
    group_by_category=['cluster_label', 'logRecordType']
)

max_len = 300
feature_extractor = FeatureExtractor(config)
attributes = logrecord.attributes
timestamps = pd.to_datetime(logrecord.timestamp['timestamp'])
# convert to feature vector
feature_vector = feature_extractor.convert_to_feature_vector(log_vectors_w2v, attributes, timestamps, max_len)

```

### Analytics

The analytics layer contains modules to perform different types of log analytics tasks. Here we started with two
common types:
* Log clustering
* Log-based anomaly detection

#### Log Clustering

Log clustering is to group log records by their similarity. Here with the log feature vectors, calculating similarity
is fairly straightforward. Below is an simple example to cluster logs using k-means clustering algorithm.

```python
#Clustering using K-Means
from logai.algorithms.clustering_algo.kmeans import KMeansParams

algo_params = KMeansParams()
clustering_config = ClusteringConfig('KMeans', algo_params, None)

clustering = Clustering(clustering_config)

feature_for_clustering = feature_vector.loc[:, ~feature_vector.columns.isin(['timestamp', 'cluster_label', 'logRecordType'])]

clustering.fit(feature_for_clustering)
kmeans_res = clustering.predict(feature_for_clustering)

```

#### Log-based Anomaly Detection

Log-based anomaly detection contains a variety of problems, and the definition of log anomalies may be different in
different context. Here we define log-based anomaly detection is to detect the anomalous log event groups we've retrieved
in log featurization. Thus, based on different types of featurization outcomes, the anomaly detection can also be in three types:
* AD for tabular log data.
* AD for log sequences.
* AD for time-series log counters.

Here we show a simple example of AD for tabular log data using single class SVM.

```python
# Anomaly Detection
# One class SVM for outlier detection

feature_for_anomaly_detection = feature_vector.loc[:, ~feature_vector.columns.isin(['timestamp', 'cluster_label', 'logRecordType'])]
train, test = train_test_split(feature_for_anomaly_detection, train_size=0.7, test_size=0.3)

from logai.analysis.anomaly_detector import AnomalyDetectionConfig, AnomalyDetector
from logai.algorithms.anomaly_detection_algo.one_class_svm import OneClassSVMParams

algo_params = OneClassSVMParams()
config = AnomalyDetectionConfig('OneClassSVM', algo_params, None)

anomaly_detector = AnomalyDetector(config)
anomaly_detector.fit(train)
res = anomaly_detector.predict(test)

```
