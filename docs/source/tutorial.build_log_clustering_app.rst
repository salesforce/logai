
.. role:: file (code)
  :language: shell
  :class: highlight

.. image:: _static/logai_logo.jpg
   :width: 650
   :align: center

Tutorial: Log Clustering using LogAI
=======================================================

This is an example to show how to use LogAI to conduct log clustering analysis.

Load Data
-------------------------------------------------------

You can use :file:`OpensetDataLoader` to load a sample open log dataset. Here we use HDFS dataset from
`LogHub <https://zenodo.org/record/3227177#.Y1M3LezML0o>`_ as an example.

.. code-block:: python

    import os
    from logai.dataloader.openset_data_loader import OpenSetDataLoader, OpenSetDataLoaderConfig

    #File Configuration
    filepath = "../datasets/HDFS_2000.log"
    filepath = os.path.join("..", "datasets", "HDFS_2000.log")

    dataset_name = "HDFS"
    data_loader = OpenSetDataLoader(
        OpenSetDataLoaderConfig(
            dataset_name=dataset_name,
            filepath=filepath)
    )

    logrecord = data_loader.load_data()

    logrecord.to_dataframe().head(5)


Preprocess
------------------------------------------------------------------------------

In preprocessing step user can retrieve and replace any regex strings and clean the raw loglines. This
can be very useful to improve information extraction of the unstructured part of logs,
as well as generate more structured attributes with domain knowledge.

Here in the example, we use the below regex to retrieve Block IDs, IP addresses and filepaths.

.. code-block:: python

    from logai.preprocess.preprocessor import PreprocessorConfig, Preprocessor
    from logai.utils import constants

    loglines = logrecord.body[constants.LOGLINE_NAME]
    attributes = logrecord.attributes

    preprocessor_config = PreprocessorConfig(
        custom_replace_list=[
            [r"(?<=blk_)[-\d]+", "<block_id>"],
            [r"\d+\.\d+\.\d+\.\d+", "<IP>"],
            [r"(/[-\w]+)+", "<file_path>"],
        ]
    )

    preprocessor = Preprocessor(preprocessor_config)

    clean_logs, custom_patterns = preprocessor.clean_log(
        loglines
    )


Parsing
------------------------------------------------------------------------------

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


Information Extraction
------------------------------------------------------------------------------------

Vectorization for unstructured loglines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We also do categorical encoding for log attributes to convert the strings into numerical representations.

.. code-block:: python

    from logai.information_extraction.categorical_encoder import CategoricalEncoderConfig, CategoricalEncoder

    encoder_config = CategoricalEncoderConfig(name="label_encoder")

    encoder = CategoricalEncoder(encoder_config)

    attributes_encoded = encoder.fit_transform(attributes)


Feature Extraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Then we extract and concate the semantic features for both the unstructured and structured part of logs.

.. code-block:: python

    from logai.information_extraction.feature_extractor import FeatureExtractorConfig, FeatureExtractor

    timestamps = logrecord.timestamp['timestamp']

    config = FeatureExtractorConfig(
        max_feature_len=100
    )

    feature_extractor = FeatureExtractor(config)

    _, feature_vector = feature_extractor.convert_to_feature_vector(log_vectors, attributes_encoded, timestamps)


Clustering
----------------------------------------------------------------------------------------

Here we use K-Means clustering algorithm as an example. We set the number of clusters to 7 in
K-Means algorithm parameter configuration.

.. code-block:: python

    from logai.algorithms.clustering_algo.kmeans import KMeansParams
    from logai.analysis.clustering import ClusteringConfig, Clustering

    clustering_config = ClusteringConfig(
        algo_name='kmeans',
        algo_params=KMeansParams(
            n_clusters=7
        )
    )

    log_clustering = Clustering(clustering_config)

    log_clustering.fit(feature_vector)

    cluster_id = log_clustering.predict(feature_vector).astype(str).rename('cluster_id')


Then you can check the clustering results

.. code-block:: python

    # Check clustering results.
    logrecord.to_dataframe().join(cluster_id).head(5)

To run this example, you can check the
`jupyter notebook <https://github.com/salesforce/logai/blob/main/examples/jupyter_notebook/tutorial_log_clustering.ipynb>`_
example on Github.

