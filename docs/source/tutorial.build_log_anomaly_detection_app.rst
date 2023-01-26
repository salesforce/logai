
.. role:: file (code)
  :language: shell
  :class: highlight

.. image:: _static/logai_logo.jpg
   :width: 650
   :align: center

Tutorial: Log Anomaly Detection Using LogAI
=========================================================

This is an example to show how to use LogAI to conduct log anomaly detection analysis.

Load Data
----------------------------------------------

You can use :file:`OpensetDataLoader` to load a sample open log dataset. Here we use HealthApp dataset from
`LogHub <https://zenodo.org/record/3227177#.Y1M3LezML0o>`_ as an example.

.. code-block:: python

    import os

    from logai.dataloader.openset_data_loader import OpenSetDataLoader, OpenSetDataLoaderConfig

    #File Configuration
    filepath = os.path.join("..", "datasets", "HealthApp_2000.log") # Point to the target HealthApp.log dataset

    dataset_name = "HealthApp"
    data_loader = OpenSetDataLoader(
        OpenSetDataLoaderConfig(
            dataset_name=dataset_name,
            filepath=filepath)
    )

    logrecord = data_loader.load_data()

    logrecord.to_dataframe().head(5)


Preprocess
---------------------------------------------------------

In preprocessing step user can retrieve and replace any regex strings and clean the raw loglines. This
can be very useful to improve information extraction of the unstructured part of logs,
as well as generate more structured attributes with domain knowledge.

Here in the example, we use the below regex to retrieve IP addresses.

.. code-block:: python

    from logai.preprocess.preprocessor import PreprocessorConfig, Preprocessor
    from logai.utils import constants

    loglines = logrecord.body[constants.LOGLINE_NAME]
    attributes = logrecord.attributes

    preprocessor_config = PreprocessorConfig(
        custom_replace_list=[
            [r"\d+\.\d+\.\d+\.\d+", "<IP>"],   # retrieve all IP addresses and replace with <IP> tag in the original string.
        ]
    )

    preprocessor = Preprocessor(preprocessor_config)

    clean_logs, custom_patterns = preprocessor.clean_log(
        loglines
    )

Parsing
---------------------------------------------------------------

After preprocessing, we call auto-parsing algorithms to automatically parse the cleaned logs.

.. code-block:: python

    from logai.information_extraction.log_parser import LogParser, LogParserConfig
    from logai.algorithms.parsing_algo.drain import DrainParams

    # parsing
    parsing_algo_params = DrainParams(
        sim_th=0.5, depth=5
    )

    log_parser_config = LogParserConfig(
        parsing_algorithm="drain",
        parsing_algo_params=parsing_algo_params
    )

    parser = LogParser(log_parser_config)
    parsed_result = parser.parse(clean_logs)

    parsed_loglines = parsed_result['parsed_logline']


Time-series Anomaly Detection
---------------------------------------------------------------

Here we show an example to conduct time-series anomaly detection with parsed logs.

Feature Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

After parsing the logs and get log templates, we can extract time-series features by converting
these parsed loglines into counter vectors.

.. code-block:: python

    from logai.information_extraction.feature_extractor import FeatureExtractorConfig, FeatureExtractor

    config = FeatureExtractorConfig(
        group_by_time="15min",
        group_by_category=['parsed_logline', 'Action', 'ID'],
    )

    feature_extractor = FeatureExtractor(config)

    timestamps = logrecord.timestamp['timestamp']
    parsed_loglines = parsed_result['parsed_logline']
    counter_vector = feature_extractor.convert_to_counter_vector(
        log_pattern=parsed_loglines,
        attributes=attributes,
        timestamps=timestamps
    )

    counter_vector.head(5)


Anomaly Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the generated :file:`counter_vcetor`, you can use :file:`AnomalyDetector` to detect timeseries anomalies.
Here we use :file:`ETS` algorithm integrated in Merlion.

.. code-block:: python

    from logai.analysis.anomaly_detector import AnomalyDetector, AnomalyDetectionConfig
    from sklearn.model_selection import train_test_split
    import pandas as pd

    counter_vector["attribute"] = counter_vector.drop(
                    [
                        constants.LOG_COUNTS,
                        constants.LOG_TIMESTAMPS,
                        constants.EVENT_INDEX
                    ],
                    axis=1
                ).apply(
                    lambda x: "-".join(x.astype(str)), axis=1
                )

    attr_list = counter_vector["attribute"].unique()

    anomaly_detection_config = AnomalyDetectionConfig(
        algo_name='dbl'
    )

    res = pd.DataFrame()
    for attr in attr_list:
        temp_df = counter_vector[counter_vector["attribute"] == attr]
        if temp_df.shape[0] >= constants.MIN_TS_LENGTH:
            train, test = train_test_split(
                temp_df[[constants.LOG_TIMESTAMPS, constants.LOG_COUNTS]],
                shuffle=False,
                train_size=0.3
            )
            anomaly_detector = AnomalyDetector(anomaly_detection_config)
            anomaly_detector.fit(train)
            anom_score = anomaly_detector.predict(test)
            res = res.append(anom_score)

Then you chan check detected anomalou datapoints:

.. code-block:: python

    # Get anomalous datapoints
    anomalies = counter_vector.iloc[res[res>0].index]
    anomalies.head(5)



Semantic Anomaly Detection
---------------------------------------------------------------

We can also use the log template for semantic based anomaly detection. In this approach, we retrieve
the semantic features from the logs. This includes two parts: vectorizing the unstructured log templates
and encoding the structured log attributes.

Vectorization for unstructured loglines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Here we use `word2vec` to vectorize unstructured part of the logs. The output will be a list of
numeric vectors that representing the semantic features of these log templates.

.. code-block:: python

    from logai.information_extraction.log_vectorizer import VectorizerConfig, LogVectorizer

    vectorizer_config = VectorizerConfig(
        algo_name = "word2vec"
    )

    vectorizer = LogVectorizer(
        vectorizer_config
    )

    # Train vectorizer
    vectorizer.fit(parsed_loglines)

    # Transform the loglines into features
    log_vectors = vectorizer.transform(parsed_loglines)

Categorical Encoding for log attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We also do categorical encoding for log attributes to convert the strings into numerical representations.

.. code-block:: python

    from logai.information_extraction.categorical_encoder import CategoricalEncoderConfig, CategoricalEncoder

    encoder_config = CategoricalEncoderConfig(name="label_encoder")

    encoder = CategoricalEncoder(encoder_config)

    attributes_encoded = encoder.fit_transform(attributes)


Feature Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Then we extract and concate the semantic features for both the unstructured and structured part of logs.


.. code-block:: python

    from logai.information_extraction.feature_extractor import FeatureExtractorConfig, FeatureExtractor

    timestamps = logrecord.timestamp['timestamp']

    config = FeatureExtractorConfig(
        max_feature_len=100
    )

    feature_extractor = FeatureExtractor(config)

    _, feature_vector = feature_extractor.convert_to_feature_vector(log_vectors, attributes_encoded, timestamps)


Anomaly Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With the extracted log semantic feature set, we can perform anomaly detection to find the abnormal
logs. Here we use `isolation_forest` as an example.

.. code-block:: python

    from sklearn.model_selection import train_test_split

    train, test = train_test_split(feature_vector, train_size=0.7, test_size=0.3)

    from logai.algorithms.anomaly_detection_algo.isolation_forest import IsolationForestParams
    from logai.analysis.anomaly_detector import AnomalyDetectionConfig, AnomalyDetector

    algo_params = IsolationForestParams(
        n_estimators=10,
        max_features=100
    )
    config = AnomalyDetectionConfig(
        algo_name='isolation_forest',
        algo_params=algo_params
    )

    anomaly_detector = AnomalyDetector(config)
    anomaly_detector.fit(train)
    res = anomaly_detector.predict(test)
    # obtain the anomalous datapoints
    anomalies = res[res==1]

Check the corresponding loglines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    loglines.iloc[anomalies.index].head(5)


Check the corresponding attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    attributes.iloc[anomalies.index].head(5)


To run this example, you can check the
`jupyter notebook <https://github.com/salesforce/logai/blob/main/examples/jupyter_notebook/tutorial_log_anomaly_detection.ipynb>`_
example on Github.