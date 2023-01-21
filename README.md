<!--
Copyright (c) 2022 Salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

-->

<div>
    <img src="./img/logai_logo.jpg" alt="logo" width="200"/>
</div>


[![tests](https://github.com/salesforce/logai/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/salesforce/logai/actions/workflows/tests.yml)

# Log-AI: Log Analytics and Intelligence Library

## What is LogAI

LogAI is a library that provides machine learning (ML) based log analytics and intelligent tools. LogAI can be used to  solve a variety of log analysis related problems, such as log clustering log search, log based anomaly detection and log based failure prediction. LogAI is used in multiple scenarios in Salesforce core stack, and it can be easily expandable to external use scenarios. 

LogAI helps the user to easily perform popular log tasks:
* Log Summarization
* Log Clustering
* Log anomaly Detection
* ...

## LogAI compare to other open-source projects about AI-based log analytics

There are a few existing open-source code repos which conducts AI-based log analytics taks. 
Below table compares LogAI with these projects on several aspects. 

| Coverage  | LogAI | NewRelic Log Monitoring | DataDog Log Explorer | logparser | loglizer | deep-loglizer | log3C | 
| ------------- | ------------- | ------------- |  ------------- | ------------- | ------------- | ------------- | ------------- |
| OpenTelemetry log data model  | :white_check_mark:  | :white_check_mark:  | :white_check_mark:  | | | | | | 
| Unified data loader and preprocessing | :white_check_mark:  | :white_check_mark:  | :white_check_mark:  | :white_check_mark:  | :white_check_mark:  | :white_check_mark: | |
| Auto log parsing  | :white_check_mark:  | :white_check_mark: | :white_check_mark: | | | 
| Log clustering | :white_check_mark: | :white_check_mark:  | :white_check_mark:  | | | | :white_check_mark: | 
| Log anomaly detection - time-series | :white_check_mark: | :white_check_mark:  | :white_check_mark:  | | | | | | 
| Log anomaly detection - traditional ML | :white_check_mark: |  |  | | :white_check_mark: |  |  |  
| Log anomaly detection - deep Learning | :white_check_mark: |  | | | :white_check_mark: | :white_check_mark: |  |  
| Huggingface integration (TBD) | :white_check_mark: | | | | | |  |
| GUI for result visualization (TBD) | :white_check_mark: | :white_check_mark: | :white_check_mark: | | | | | 


## How to Use

#### Install LogAI:

```shell
git clone https://git.soma.salesforce.com/SalesforceResearch/logai.git
cd logai
python3 -m venv venv # create virtual environment
source venv/bin/activate # activate virtual env
pip install ./ # install LogAI from root directory
```

#### Use GUI to explore LogAI

```shell
export PYTHONPATH='.'  # make sure to add current root to PYTHONPATH
python3 gui/application.py # Run local plotly dash server.

```

