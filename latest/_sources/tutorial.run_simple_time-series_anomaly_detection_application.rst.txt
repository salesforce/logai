
.. role:: file (code)
  :language: shell
  :class: highlight

.. image:: _static/logai_logo.jpg
   :width: 650
   :align: center

Run Simple Time-series Anomaly Detection Application
==================================================================

You can also use LogAI in more programtic ways. LogAI supports configuration files in :file:`.json` or :file:`.yaml`.
Below is a sample :file:`log_anomaly_detection_config.json` configuration for anomaly detection application. Make sure
to set :file:`filepath` to the target log dataset file path.


.. code-block:: json

   {
      "open_set_data_loader_config": {
        "dataset_name": "HDFS",
        "filepath": ""
      },
      "preprocessor_config": {
          "custom_delimiters_regex":[]
      },
      "log_parser_config": {
        "parsing_algorithm": "drain",
        "parsing_algo_params": {
          "sim_th": 0.5,
          "depth": 5
        }
      },
      "feature_extractor_config": {
          "group_by_category": ["Level"],
          "group_by_time": "1s"
      },
      "log_vectorizer_config": {
          "algo_name": "word2vec"
      },
      "categorical_encoder_config": {
          "name": "label_encoder"
      },
      "anomaly_detection_config": {
          "algo_name": "one_class_svm"
      }
    }

Then to run log anomaly detection. You can simple create below python script:

.. code-block:: python

    import json

    from logai.applications.application_interfaces import WorkFlowConfig
    from logai.applications.log_anomaly_detection import LogAnomalyDetection

    # path to json configuration file
    json_config = "./log_anomaly_detection_config.json"

    # Create log anomaly detection application workflow configuration
    config = json.loads(json_config)
    workflow_config = WorkFlowConfig.from_dict(config)

    # Create LogAnomalyDetection Application for given workflow_config
    app = LogAnomalyDetection(workflow_config)

    # Execute App
    app.execute()

Then you can check anomaly detection results by calling :file:`app.anomaly_results`.
To run this example, you can check the
`jupyter notebook <https://github.com/salesforce/logai/blob/main/examples/jupyter_notebook/log_anomaly_detection_example.ipynb>`_
example on Github.


