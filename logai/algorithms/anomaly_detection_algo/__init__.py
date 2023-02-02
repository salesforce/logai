#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from .dbl import DBLDetector
from .distribution_divergence import DistributionDivergence
from .ets import ETSDetector
from .isolation_forest import IsolationForestDetector
from .local_outlier_factor import LOFDetector
from .one_class_svm import OneClassSVMDetector
from logai.utils.misc import is_torch_available, \
    is_transformers_available

_MODULES = [
    "DBLDetector",
    "DistributionDivergence",
    "ETSDetector",
    "IsolationForestDetector",
    "LOFDetector",
    "OneClassSVMDetector"
]

if is_torch_available() and is_transformers_available():
    from .forecast_nn import ForecastBasedLSTM, ForecastBasedCNN, ForecastBasedTransformer
    from .logbert import LogBERT

    _MODULES += [
        "LogBERT",
        "ForecastBasedLSTM",
        "ForecastBasedCNN",
        "ForecastBasedTransformer"
    ]

__all__ = _MODULES
