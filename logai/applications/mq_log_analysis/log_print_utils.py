#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#


def get_topk_most_anomalous(predictions, sort_by_field, log_metric, topk=None):
    for k,v in predictions.items():
        if sort_by_field not in v:
            v[sort_by_field][log_metric] = 0. 
        elif log_metric not in v[sort_by_field]:
            v[sort_by_field][log_metric] = 0. 

    sorted_predictions = [(k,v[sort_by_field][log_metric]) for k,v in sorted(predictions.items(), key=lambda item: item[1][sort_by_field][log_metric] if log_metric in item[1][sort_by_field] else 0., reverse=True)]
    if topk is not None:
        sorted_predictions = sorted_predictions[:topk]
    return sorted_predictions

def print_topk_most_anomalous(predictions, sort_by_field, log_metric, topk=None):
    i=0
    print('Top-%s most anomalous as per %s-%s' % (topk, sort_by_field, log_metric))
    sorted_predictions = get_topk_most_anomalous(predictions, sort_by_field, log_metric, topk)
    for k,v in sorted_predictions:
        print('\t%s : %s-%s : %s' % (k, sort_by_field, log_metric, v))
        

