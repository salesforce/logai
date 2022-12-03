<!--
Copyright (c) 2022 Salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

-->

![logo](./img/logai_logo.jpg)

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

### Architecture

Log-AI architecture includes several layers: Data layer, preprocessing layer, information extraction layer,
analysis layer, visualization layer and evaluation layer. Components of these layers covers functionalities that
widely used in current log analysis tasks. 

![design](./img/LogAIDesign.png)

## How to Use

#### Install dependencies:

```shell
git clone https://git.soma.salesforce.com/SalesforceResearch/logai.git
cd logai
python3 -m venv venv # create virtual environment
source venv/bin/activate # activate virtual env
pip install -r requirements.txt
```

#### Build wheels package:

```shell
python setup.py bdist_wheel
```
Then you can find the .whl package in `./dist/`.

#### Install Log-AI from wheels:

```shell
pip install logai-{version}-py2.py3-none-any.whl
```

#### Use GUI to explore LogAI

```shell
export PYTHONPATH='.'  # make sure to add current root to PYTHONPATH
python3 gui/application.py # Run local plotly dash server.

```
## How to Build Your Own LogAI application

### Jupyter Notebooks
Please refer to [Instruction in Use Case Notebooks](./use_case_notebooks/README.md) for more information about how to
use LogAI modules to create E2E applications in Jupyter Notebook.


## Miscs

### Run Unit Tests

```shell
./run_unittests.sh 
```

### Update Liscence Header

```shell
python -m licenseheaders -t .copyright.tmpl -y "2022" -o "Salesforce.com, inc."

```

### Build Doc using sphinx

```shell
cd docs 
make clean
make html
```


