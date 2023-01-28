<!--
Copyright (c) 2023 Salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

-->
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

## How to Build Your Own LogAI application

### Jupyter Notebooks
Please refer to [Instruction in Use Case Notebooks](./tutorial/how_to_use.md) for more information about how to
use LogAI modules to create E2E applications in Jupyter Notebook.



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
