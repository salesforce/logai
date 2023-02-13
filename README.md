<!--
Copyright (c) 2023 Salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

-->

<p align="center">
    <br>
    <img src="./img/logai_logo.jpg" width="400"/>
    </br>
</p>

<div align="center">
  <a href="https://github.com/salesforce/logai/actions/workflows/tests.yml">
    <img alt="Latest Release" src="https://github.com/salesforce/logai/actions/workflows/tests.yml/badge.svg?branch=main" />
  </a>
  <a href="https://github.com/salesforce/logai/actions/workflows/pages/pages-build-deployment">
    <img alt="pages-build-deployment" src="https://github.com/salesforce/logai/actions/workflows/pages/pages-build-deployment/badge.svg" />
  </a>
  <a href="https://github.com/salesforce/logai/releases">
    <img alt="Latest Release" src="https://img.shields.io/github/release/salesforce/logai.svg" />
  </a>
  <a href="https://badge.fury.io/py/logai">
    <img src="https://badge.fury.io/py/logai.svg" alt="PyPI version">
  </a>
  <a href="https://opensource.org/licenses/BSD-3-Clause">
    <img alt="license" src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg"/>
  </a>
</div>

# LogAI: A Library for Log Analytics and Intelligence

## Table of Contents
* [Introduction](#introduction)
* [Installation](#installation)
* [Getting Started](#getting-started)
  * [Explore LogAI GUI Portal](#explore-logai-gui-portal)
  * [Run Simple Time-series Anomaly Detection Application](#run-simple-time-series-anomaly-detection-application)
  * [Build Customized LogAI Applications](#build-customized-logai-applications)
  * [Deep-learning Anomaly Detection Benchmarking](#deep-learning-anomaly-detection-benchmarking)
* [Documentation](#documentation)
* [Technical Report and Citing LogAI](#technical-report-and-citing-logai)
* [Contact](#contact)
* [License](#license)

## Introduction
LogAI is a one-stop open source library for log analytics and intelligence. LogAI supports various log analytics and log intelligence tasks such as log summarization, log clustering, log anomaly detection and more. It adopts the OpenTelemetry data model, to enable compatibility with different log management platforms. LogAI provides a unified model interface and integrates popular time-series, statistical learning and deep 
learning models. Alongside this, LogAI also provides an out-of-the-box GUI toolkit for users to conduct interactive log 
analysis. With LogAI, we can also easily benchmark popular ML and deep-learning algorithms for log anomaly detection 
without putting in redundant effort to process the logs. We have opensourced LogAI to facilitate a wide range of 
applications benefiting both academic research and industrial prototyping. 
 
The following table compares LogAI with several existing log analysis Tools, include both commercial log management platforms
like NewRelic and DataDog, or popular log analysis open source tools on Github. 

| Coverage  | LogAI | NewRelic Log Monitoring | DataDog Log Explorer | logparser | loglizer | deep-loglizer | log3C | 
| ------------- | ------------- | ------------- |  ------------- | ------------- | ------------- | ------------- | ------------- |
| OpenTelemetry log data model  | :white_check_mark:  | :white_check_mark:  | :white_check_mark:  | | | | | | 
| Unified data loader and preprocessing | :white_check_mark:  | :white_check_mark:  | :white_check_mark:  | :white_check_mark:  | :white_check_mark:  | :white_check_mark: | |
| Auto log parsing  | :white_check_mark:  | :white_check_mark: | :white_check_mark: | | | 
| Log clustering | :white_check_mark: | :white_check_mark:  | :white_check_mark:  | | | | :white_check_mark: | 
| Log anomaly detection - time-series | :white_check_mark: | :white_check_mark:  | :white_check_mark:  | | | | | | 
| Log anomaly detection - traditional ML | :white_check_mark: |  |  | | :white_check_mark: |  |  |  
| Log anomaly detection - deep Learning | :white_check_mark: |  | | | :white_check_mark: | :white_check_mark: |  |  
| Huggingface integration | :white_check_mark: | | | | | |  |
| GUI for result visualization | :white_check_mark: | :white_check_mark: | :white_check_mark: | | | | |

## Installation

### Quick Install
You can install LogAI core library using `pip install`:

```shell

# Check out LogAI code repo from Github
git clone https://github.com/salesforce/logai.git
cd logai

# [Optional] Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install LogAI
pip install logai

```

### Install Optional Dependencies
LogAI core library is light-weight with limited dependent packages installed. Users can install optional dependencies
to enable extended functionalities of LogAI.

**Deep Learning Log Analysis**. To conduct deep learning model related tasks and run benchmarking,
please install extra requirements by `pip install "logai[deep-learning]"`.

**Enable LogAI GUI Portal***. To use LogAI GUI portal, 
please install extra requirements by `pip install "logai[gui]"`.

**LogAI Development**. To contribute to LogAI development, build and test code changes, 
please install extra requirements by `pip install "logai[dev]"`.

**Complete Installation**. you can install the full list of dependencies by `pip install "logai[all]"`.

### Known Issues

> :warning: You may see `Resource punkt not found` while using LogAI. You can download `punkt`
> package from NLTK to solve the problem.
> ```shell
> python -m nltk.downloader punkt
> ```

## Getting Started

Below we briefly introduce several ways to explore and use LogAI, including exploring LogAI GUI
portal, benchmarking deep-learning based log anomaly detection using LogAI, and building your 
own log analysis application with LogAI.

### Explore LogAI GUI Portal 

You can also start a local LogAI service and use the GUI portal to explore LogAI.

```shell

# Check out LogAI code repo from Github
git clone https://github.com/salesforce/logai.git
cd logai

# [Optional] Create virtual environment
python3 -m venv venv # create virtual environment
source venv/bin/activate # activate virtual env

# install LogAI and GUI dependencies
pip install ".[dev]"
pip install ".[gui]"

# Start LogAI service
export PYTHONPATH='.'  # make sure to add current root to PYTHONPATH
python3 gui/application.py # Run local plotly dash server.
```

Then open the LogAI portal via http://localhost:8050/ or http://127.0.0.1:8050/ in your browser:

![portal](img/logai_gui_landing.png)

The control panel is on the left side of the page. There are three applications you can choose from:
Log Summarization, Log Clustering and Anomaly Detection.

#### Control Panel

**File Setting**. You can select the log type and log files to be processed. Now LogAI supports three 
public datasets: HDFS, BGL and HealthApp. For each log type we included several sample log data. After
log file selected, you can choose the attributes you want to be involved in log processing. The selected 
attributes will be treated as structured log attributes.

**Algorithm Setting**. For different applications, the algorithms options may be different.
For example, auto-log parsing algorithms are utilized for log summarization, but log clustering uses auto-parsing algorithms,
 vectorization algorithms, categorical encoding and clustering algorithms. You can select an
 algorithm and change the parameters in each algorithm section. After algorithm configuration, simply click "run" to
run the application.

#### Log Summarization

Log summarization App summarize and group the raw logs by log patterns and attributes. You can click on 
each log pattern and see what the pattern looks like and the dynamic values in each position. You can also 
see the chart of occurrance trend of this pattern on the right side.

![log summarization](img/logai_summarization_res.png)

#### Log Clustering

Log clustering App groups raw logs into clusters by calculating the semantic representation of each logline. 
Then using clustering algorithms to generate log clusters. In this example, we choose k-mean where `k==8` to
generate 8 clusters. The result is shown as a pie chart and you can click each portion of the pie chart to check
the raw logs in this cluster.

![log clustering](img/logai_clustering_res.png)

#### Anomaly Detection

Log anomaly detection App conduct log anomaly detection tasks. Similar to log clustering, log anomaly detection
also needs to extract information from raw logs and generate representation of loglines. Depend on the type of anomaly detection,
The representation can be different. 
**Time-series anomaly detection**. If we use time-series algorithm like ETS, the raw logs will be converted
into log counter vectors by given time interval. Then ETS is performed on the generated log counter vectors and detect
anomalous timestamps on the counter vector time-series. 

**Semantic anomaly detection**. If we use unsupervised outlier detection algorithms such as One-class SVM, the raw logs will 
be converted into semantic vectors and feed the One-class SVM model. Then the model will detect anomalous loglines.

![log anomaly detection](img/logai_anomaly_detection.png)

LogAI GUI portal is just an example to demo LogAI capabilities. We know this may not be the best way to visualize the 
results and there might be bugs in how the results are displayed. We will keep working with the open source community
to improve usability of the portal. Any feedbacks and contributions are welcome :blush:. 

### Run Simple Time-series Anomaly Detection Application

You can also use LogAI in more programtic ways. LogAI supports configuration files in `.json` or `.yaml`. 
Below is a sample `log_anomaly_detection_config.json` configuration for anomaly detection application. 
Make sure to set `filepath` to the target log dataset file path.

```json
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
```

Then to run log anomaly detection. You can simply create below python script:

```python
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

```

Then you can check anomaly detection results by calling `app.anomaly_results`. 

For full context of this example please check
[Tutorial: Use Log Anomaly Detection Application](./examples/jupyter_notebook/log_anomaly_detection_example.ipynb).

### Build Customized LogAI Applications
You can build your own customized log analysis applications using LogAI. Here we show two examples:

* [Tutorial: Log Clustering Using LogAI](./examples/jupyter_notebook/tutorial_log_clustering.ipynb)
* [Tutorial: Log Anomaly Detection Using LogAI](./examples/jupyter_notebook/tutorial_log_anomaly_detection.ipynb)

### Deep-learning Anomaly Detection Benchmarking

LogAI can be used to benchmark deep-learning anomaly detection results. 
A [tutorial](examples/jupyter_notebook/tutorial_deep_ad.md) is provided for 
Anomaly Detection Benchmarking using LSTM anomaly detector for HDFS Dataset. More examples of deep-learning anomaly 
detection benchmarking on different datasets and algorithms can be found in 
[Deep Anomaly Detection Benchmarking Examples](examples/jupyter_notebook/nn_ad_benchmarking).

## Documentation

For more detail about LogAI library and advanced use cases, please visit 
[LogAI Documentation](https://opensource.salesforce.com/logai).

## Technical Report and Citing LogAI

You can find more details about LogAI in the [technical report](https://arxiv.org/abs/2301.13415). 
If you're using LogAI in your research or applications, please cite using this BibTeX:

```
@misc{https://doi.org/10.48550/arxiv.2301.13415,
    title = {LogAI: A Library for Log Analytics and Intelligence},
    author = {Cheng, Qian and Saha, Amrita and Yang, Wenzhuo and Liu, Chenghao and Sahoo, Doyen and Hoi, Steven},
    publisher = {arXiv},
    year = {2023},
    doi = {10.48550/ARXIV.2301.13415},
    url = {https://arxiv.org/abs/2301.13415},
    copyright = {arXiv.org perpetual, non-exclusive license}
}

```

## Contact
If you have any questions, comments or suggestions, 
please do not hesitate to contact us at [logai@salesforce.com](logai@salesforce.com). 

## License
[BSD 3-Clause License](LICENSE.txt)

