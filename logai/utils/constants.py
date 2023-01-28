#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from enum import Enum

DIGITS_SUB = "[DIGITS]"
TIMESTAMP = "[TIMESTAMP]"


# Log record object fields
class Field(str, Enum):
    TIMESTAMP = "timestamp"
    BODY = "body"
    ATTRIBUTES = "attributes"
    RESOURCE = "resource"
    SPAN_ID = "span_id"
    LABELS = "labels"


# Attribute names
LOGLINE_NAME = "logline"
NEXT_LOGLINE_NAME = "next_logline"
PARSED_LOGLINE_NAME = "parsed_logline"
PARAMETER_LIST_NAME = "parameter_list"
LOG_EVENTS = "log_events"
LOG_TIMESTAMPS = "timestamp"
SPAN_ID = "span_id"
EVENT_INDEX = "event_index"
LABELS = "labels"

# Counts
LOG_COUNTS = "counts"


# HYPER PARAMS
MIN_TS_LENGTH = 10
COUNTER_AD_ALGO = ["ets", "dbl"]
