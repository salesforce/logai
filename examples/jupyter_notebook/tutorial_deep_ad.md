
# Deep-learning Anomaly Detection Benchmarking

## Benchmarking Result Table

Below table is the comparison between 
different supervised and unsupervised deep learning anomaly detection
models in LogAI and [Deep-Loglizer](https://github.com/logpai/deep-loglizer) library, using F1-Score as the 
performance metric. The dashed (-) cells indicate that there are no reported numbers in the Deep-Loglizer 
paper corresponding to those configurations.
<table>
    <thead>
        <tr>
            <th rowspan=2>Model</th>
            <th rowspan=2>Details</th>
            <th rowspan=2>Supervision</th>
            <th rowspan=2>Log Parsing</th>
            <th rowspan=2>Log Representation</th>
            <th colspan=2>HDFS</th>
            <th colspan=2>BGL</th>
        </tr>
        <tr>
            <th>LogAI</th>
            <th>Deep Loglizer</th>
            <th>LogAI</th>
            <th>Deep Loglizer</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=4>LSTM</td>
            <td rowspan=2>Undirectional, No Attention</td>
            <td rowspan=2>Unsupervised</td>
            <td>:heavy_check_mark:</td>
            <td>sequential</td>
            <td>0.981</td>
            <td>0.944</td>
            <td>0.938</td>
            <td>0.961</td>
        </tr>
        <tr>
            <td>:heavy_multiplication_x:</td>
            <td>semantic</td>
            <td>0.981</td>
            <td>0.945</td>
            <td>0.924</td>
            <td>0.967</td>
        </tr>
        <tr>
            <td rowspan=2>Bidirectional, with Attention</td>
            <td rowspan=2>Supervisedn</td>
            <td>:heavy_check_mark:</td>
            <td>sequntial</td>
            <td>0.979</td>
            <td>-</td>
            <td>0.925</td>
            <td>-</td>
        </tr>
        <tr>
            <td>:heavy_multiplication_x:</td>
            <td>semantic</td>
            <td>0.981</td>
            <td>-</td>
            <td>0.924</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan=4>CNN</td>
            <td rowspan=4>2-D Convolution with 1-D Max Pooling</td>
            <td rowspan=2>Unsupervised</td>
            <td>:heavy_check_mark:</td>
            <td>sequential</td>
            <td>0.981</td>
            <td>-</td>
            <td>0.929</td>
            <td>-</td>
        </tr>
        <tr>
            <td>:heavy_multiplication_x:</td>
            <td>sequential</td>
            <td>0.981</td>
            <td>-</td>
            <td>0.922</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan=2>Supervised</td>
            <td>:heavy_check_mark:</td>
            <td>sequential</td>
            <td>0.943</td>
            <td>0.97</td>
            <td>0.983</td>
            <td>-</td>
        </tr>
        <tr>
            <td>:heavy_multiplication_x:</td>
            <td>sequential</td>
            <td>0.946</td>
            <td>-</td>
            <td>0.99</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan=8>Transformer</td>
            <td rowspan=8>Multihead single-layer self-attention model, trained from scratch</td>
            <td rowspan=4>Unsupervised</td>
            <td rowspan=2>:heavy_check_mark:</td>
            <td>sequential</td>
            <td>0.971</td>
            <td>0.905</td>
            <td>0.933</td>
            <td>0.956</td>
        </tr>
        <tr>
            <td>semantic</td>
            <td>0.978</td>
            <td>925</td>
            <td>0.921</td>
            <td>0.957</td>
        </tr>
        <tr>
            <td rowspan=2>:heavy_multiplication_x:</td>
            <td>sequential</td>
            <td>0.98</td>
            <td>-</td>
            <td>0.92</td>
            <td>-</td>
        </tr>
        <tr>
            <td>semantic</td>
            <td>0.975</td>
            <td>-</td>
            <td>0.917</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan=4>Supervised</td>
            <td rowspan=2>:heavy_check_mark:</td>
            <td>sequential</td>
            <td>0.934</td>
            <td>-</td>
            <td>0.986</td>
            <td>-</td>
        </tr>
        <tr>
            <td>semantic</td>
            <td>0.784</td>
            <td>-</td>
            <td>0.963</td>
            <td>-</td>
        </tr>
        <tr>
            <td rowspan=2>:heavy_multiplication_x:</td>
            <td>sequential</td>
            <td>0.945</td>
            <td>-</td>
            <td>0.994</td>
            <td>-</td>
        </tr>
        <tr>
            <td>semantic</td>
            <td>0.915</td>
            <td>-</td>
            <td>0.977</td>
            <td>-</td>
        </tr>
    </tbody>
</table>


## Create Benchmarking Application Workflow 

Below is a sample `hdfs_log_anomaly_detection_unsupervised_lstm.yaml` yaml config file which provides the configs for each component of the log anomaly detection workflow on the public dataset HDFS using an unsupervised Deep-Learning based Anomaly Detector. 

```yaml
{
workflow_config:  
  label_filepath: "tests/logai/test_data/HDFS_AD/anomaly_label.csv"
  output_dir: "temp_output"
  training_type: "unsupervised"
  parse_logline: True
  dataset_name: hdfs

  data_loader_config:
    filepath: "tests/logai/test_data/HDFS_AD/HDFS_5k.log"
    reader_args: 
      log_format: "<Date> <Time> <Pid> <Level> <Component> <Content>"
    log_type: "log"
    dimensions:
      body: ['Content']
      timestamp: ['Date', 'Time']
    datetime_format: '%y%m%d %H%M%S'
    infer_datetime: True
    
  preprocessor_config:
    custom_delimiters_regex:
                [':', ',', '=', '\t']
    custom_replace_list: [
                ['(blk_-?\d+)', ' BLOCK '],
                ['/?/*\d+\.\d+\.\d+\.\d+',  ' IP '],
                ['(0x)[0-9a-zA-Z]+', ' HEX '],
                ['\d+', ' INT ']
            ]
    
  log_parser_config:
    parsing_algorithm: "drain"

  open_set_partitioner_config:
    session_window: False
    sliding_window: 10


  log_vectorizer_config:
    algo_name: "forecast_nn"
    algo_param:
      feature_type: "sequential"
      max_token_len: 10
      embedding_dim: 100
      output_dir: "temp_output"


  nn_anomaly_detection_config:
    algo_name: "lstm"
    algo_params:
        model_name: "lstm"
        metadata_filepath: "temp_output/embedding_model/metadata.pkl"
        feature_type: "sequential"
        label_type: "next_log"
        num_train_epochs: 10
        batch_size: 4
        output_dir: "temp_output"

}
```

Then to run the end to end log anomaly detection on the HDFS dataset using LSTM Anomaly Detector (a sequence-based deep-learning model), you can use a python script like below: 

```python
from logai.applications.openset.anomaly_detection.openset_anomaly_detection_workflow import OpenSetADWorkflowConfig
from logai.utils.file_utils import read_file
from logai.utils.dataset_utils import split_train_dev_test_for_anomaly_detection
from logai.dataloader.data_loader import FileDataLoader
from logai.preprocess.hdfs_preprocessor import HDFSPreprocessor
from logai.information_extraction.log_parser import LogParser
from logai.preprocess.openset_partitioner import OpenSetPartitioner
from logai.analysis.nn_anomaly_detector import NNAnomalyDetector
from logai.information_extraction.log_vectorizer import LogVectorizer
from logai.utils import constants

# Loading workflow config from yaml file 
config_path = "hdfs_log_anomaly_detection_unsupervised_lstm.yaml" # above config yaml file
config_parsed = read_file(config_path)
config_dict = config_parsed["workflow_config"]
validate_config_dict(config_dict)
config = OpenSetADWorkflowConfig.from_dict(config_dict)

# Loading raw log data as LogRecordObject 
dataloader = FileDataLoader(config.data_loader_config)
logrecord = dataloader.load_data()

# Preprocessing raw log data using dataset(HDFS) specific Preprocessor
preprocessor = HDFSPreprocessor(config.preprocessor_config, config.label_filepath)           
logrecord = preprocessor.clean_log(logrecord)

# Parsing the preprocessed log data using Log Parser
parser = LogParser(config.log_parser_config)
parsed_result = parser.parse(logrecord.body[constants.LOGLINE_NAME])
logrecord.body[constants.LOGLINE_NAME] = parsed_result[constants.PARSED_LOGLINE_NAME]

# Partitioning the log data into sliding window partitions, to get log sequences
partitioner = OpenSetPartitioner(config.open_set_partitioner_config)
logrecord = partitioner.partition(logrecord)

# Splitting the log data (LogRecordObject) into train, dev and test data (LogRecordObjects)
(train_data, dev_data, test_data) = split_train_dev_test_for_anomaly_detection(
                logrecord,training_type=config.training_type,
                test_data_frac_neg_class=config.test_data_frac_neg,
                test_data_frac_pos_class=config.test_data_frac_pos,
                shuffle=config.train_test_shuffle
            )

# Vectorizing the log data i.e. transforming the raw log data into vectors 
vectorizer = LogVectorizer(config.log_vectorizer_config)
vectorizer.fit(train_data)
train_features = vectorizer.transform(train_data)
dev_features = vectorizer.transform(dev_data)
test_features = vectorizer.transform(test_data)


# Training the neural anomaly detector model on the training log data 
anomaly_detector = NNAnomalyDetector(config=config.nn_anomaly_detection_config)
anomaly_detector.fit(train_features, dev_features)

# Running inference on the test log data to predict whether a log sequence is anomalous or not 
predict_results = anomaly_detector.predict(test_features)
print (predict_results)            
```
This kind of Anomaly Detection workflow for various Deep-Learning models and various experimental settings have also been automated in `logai.applications.openset.anomaly_detection.openset_anomaly_detection_workflow.OpenSetADWorkflow` class which can be easily invoked like the below example

```python
import os 
from logai.applications.openset.anomaly_detection.openset_anomaly_detection_workflow import OpenSetADWorkflow, get_openset_ad_config

TEST_DATA_PATH = "tests/logai/test_data/HDFS_AD/HDFS_5k.log"
TEST_LABEL_PATH = "tests/logai/test_data/HDFS_AD/anomaly_label.csv"
TEST_OUTPUT_PATH = "tests/logai/test_data/HDFS_AD/output"

kwargs = {
      "config_filename": "hdfs",
      "anomaly_detection_type": "lstm_sequential_unsupervised_parsed_AD",
      "vectorizer_type": "forecast_nn_sequential" ,
      "parse_logline": True ,
      "training_type": "unsupervised"
}

config = get_openset_ad_config(**kwargs)   

config.data_loader_config.filepath = TEST_DATA_PATH
config.label_filepath = TEST_LABEL_PATH
config.output_dir = TEST_OUTPUT_PATH
if not os.path.exists(config.output_dir):
    os.makedirs(config.output_dir)

workflow = OpenSetADWorkflow(config)
workflow.execute()
```

