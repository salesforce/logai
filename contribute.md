<!--
Copyright (c) 2023 Salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

-->

## How to Contribute [TBA]

Contributing to Log-AI under the current framework can be breakdown into three parts:
* contributing to algorithms
* contributing to data connectors
* contributing to log preprocess

The library leverages `pytest` for unit testing. So you can write associated tests with functional code when 
contributing. To run tests, you can execute:

```shell
./run_unittests.sh
```

### Contribute to Algorithms
Develop more algorithms in information extraction and analytics layers. Adding new algorithms is 
fairly simple as long as we can wrap it to fit the corresponding algorithm interface in `algo_interface`. 

We can either wrap algorithm functions from a reliable library or implement the algorithm from the scratch
if such opensource library does not exist or is not qualified. The only thing to be aware is all algorithm parameters
are wrapped in a `dataclass`, rather than using `**kwargs` to directly pass through the wrapper functions. The reason is
to add an extra level of control on how users could turn the params as well as making the code cleaner and simpler.

In the parameter `dataclass` we can also define the default parameter values, which may not necessarily match the default
parameter values of the original algorithm library. 

### Contribute to Data Connectors
Given the fact that reading logs in different format and from different platforms are very painful, we can implement 
connectors for users to read logs a little more easily. This includes wrapping queries to call log platforms like Splunk,
identifying delimiters of different log format and assign the right fields to create log record object.

### Contribute to Log Preprocess
Recent research shows proper preprocessing will significantly improve the end results of log analytics, which is understandable
since domain knowledge are easily involved in preprocessing while such information is usually difficult to learn by generic 
machine learning models. Thus, providing easy-to-use preprocess functions could significantly boost the efficiency of
log analytics for the library users.


