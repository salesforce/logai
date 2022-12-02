#
# Copyright (c) 2022 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
"""
Create an application workflow for log pattern discovery
"""

import itertools
import json
import logging
import os
import pickle
from os.path import dirname, exists

import numpy as np
import pandas as pd
import yaml

from tqdm import tqdm
from logai.applications.application_interfaces import WorkFlowConfig
from logai.dataloader.data_loader import FileDataLoader
from logai.information_extraction.feature_extractor import FeatureExtractor
from logai.information_extraction.log_parser import LogParser
from logai.preprocess.preprocess import Preprocessor
from logai.dataloader.data_model import LogRecordObject
from logai.utils import constants
from logai.utils.functions import get_parameter_list


# Functions for log pattern discovery
def levenshtein_distance(token1, token2):
    """
    Calculating the edit distance (levenshtein distance) between two loglines (patterns).
    :param token1: list of log strings
    :param token2: list of log strings
    :return: int: distance
    """
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if token1[t1 - 1] == token2[t2 - 1] \
                    or token1[t1 - 1] == "*" or token2[t2 - 1] == "*":
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if a <= b and a <= c:
                    distances[t1][t2] = a + 1
                elif b <= a and b <= c:
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]


def calc_distance(base, p):
    """
    Calculate edit distance between base pattern and current pattern
    :param base: base pattern
    :param p: current pattern
    :return: distance: int, similarity :float
    """
    b_token = base.split(" ")
    p_token = p.split(" ")

    l = max(len(b_token), len(p_token))
    dis = levenshtein_distance(b_token, p_token)
    sim = (l - dis) * 1.0 / l
    return dis, sim


def get_sim_table(parsed_loglines: pd.Series, lrt, cluster):
    """
    Calculate similarity table for all loglines
    :param parsed_loglines: pd.Series(str): parsed loglines
    :param lrt: str: log record type
    :param cluster: int: cluster label
    :return: res: pd.DataFrame, base_pattern: str
    """
    pattern_counts = parsed_loglines.value_counts(normalize=True, sort=True, ascending=False).reset_index()
    pattern_counts.columns = ["pattern", "portion"]
    pattern_counts.sort_values(by="portion", ascending=False, inplace=True)
    base_pattern = pattern_counts["pattern"][0]

    similarity_table = []
    for index, row in pattern_counts.iterrows():
        pattern = row["pattern"]
        portion = row["portion"]
        dis, sim = calc_distance(base_pattern, pattern)
        similarity_table.append([lrt, cluster, portion, dis, sim, pattern, base_pattern])
    res = pd.DataFrame.from_dict(similarity_table)
    res.columns = ["lrt", "cluster", "portion", "distance", "similarity", "pattern", "base_pattern"]
    return res, base_pattern


def log_pattern_discovery_workflow(logrecord: LogRecordObject, config: WorkFlowConfig, lrt):
    """
    Take log records and configurations, conduct log pattern dicovery tasks
    :param logrecord: logRecordObjects
    :param config: WorkFlowConfig
    :param lrt:
    :return:
    """
    parsing_res = []
    logline_map = pd.DataFrame()
    non_qualified_clusters = []
    if not config.preprocessor_config:
        logging.warning("Preprocessor config is None")
        raise ValueError("Preprocessor config is None")

    # Preprocessor cleans the loglines
    preprocessor = Preprocessor(config.preprocessor_config)
    preprocessed_loglines, _ = preprocessor.clean_log(logrecord.body[constants.LOGLINE_NAME])

    logging.info("Preprocess completed.")

    # Feature extractor groups the target loglines and return group indices
    feature_extractor = FeatureExtractor(config.feature_extractor_config)

    index_groups, _ = feature_extractor.convert_to_sequence(log_pattern=preprocessed_loglines, attributes=logrecord.attributes)

    logging.debug("Parsing logs...")
    for i in tqdm(index_groups.index):
        cluster_label = index_groups['cluster_label'].iloc[i]
        indices = index_groups['event_index'][i]
        if index_groups['cluster_label'].iloc[i] == -1:
            continue
        if len(indices) == 1:
            continue
        loglines_in_group = preprocessed_loglines.iloc[indices]

        if not config.log_parser_config:
            logging.warning("Log parser config is None")
            raise ValueError("Log parser config is None")

        model_path = os.path.join(
            config.workflow_config["output_dir"],
            "models/model_{}_{}.pkl".format(lrt, cluster_label),
        )

        parser = LogParser(config.log_parser_config)

        # load model
        if exists(model_path):
            parser.load(model_path)

        parsed_result = parser.parse(loglines_in_group.dropna())

        # Save model
        parser.save(model_path)


        parsed_result['cluster_label'] = cluster_label

        logline_map = logline_map.append(parsed_result)
        uniq_patterns = parsed_result[constants.PARSED_LOGLINE_NAME].unique()
        num_p = len(uniq_patterns)

        if num_p > 1:
            similarity_table, base_pattern = get_sim_table(parsed_result[constants.PARSED_LOGLINE_NAME], lrt,
                                                           cluster_label)
            if min(similarity_table['similarity']) < config.workflow_config['similarity_threshold']:
                non_qualified_clusters.append((lrt, cluster_label, min(similarity_table['similarity'])))

        else:
            base_pattern = uniq_patterns[0]

        if "*" in base_pattern:
            parsed_result[constants.PARSED_LOGLINE_NAME] = base_pattern
            parsed_result[constants.PARAMETER_LIST_NAME] = parsed_result.apply(get_parameter_list, axis=1)

        para_list = parsed_result[constants.PARAMETER_LIST_NAME]
        para_list = list(map(set, itertools.zip_longest(*para_list, fillvalue=None)))
        para_list = [set(r) for r in para_list]

        ps_res = {
            "lrt": lrt,
            "cluster_label": str(cluster_label),
            "base_pattern": base_pattern,
            "parameter_list": str(para_list)
        }
        parsing_res.append(ps_res)

        map_out_path = os.path.join(
            config.workflow_config["output_dir"],
            "pattern_map/pattern_{}_{}.json".format(lrt, cluster_label),
        )

        if not exists(dirname(map_out_path)):
            try:
                os.makedirs(dirname(map_out_path))
            except OSError as exc:
                if exc.errno != exc.errno.EEXIST:
                    raise RuntimeError("{} is not a valid output directory!".format(map_out_path))

        pd.DataFrame.from_dict([ps_res]).to_json(map_out_path, orient='records')
        #pd.DataFrame.from_dict([ps_res]).to_csv(map_out_path)

    logging.debug("Parsing complete.")
    return parsing_res, logline_map, non_qualified_clusters

def main():
    # config logger
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s::%(module)s::%(funcName)s() %(message)s',
        level=logging.DEBUG
    )

    # Read Configuration from ./test_config.yaml
    # Please make sure test_config.yaml is under the same directory

    CONFIG_PATH = os.path.join(os.path.dirname(__file__), "lpd_config.yaml")
    with open(CONFIG_PATH, "r") as f:
        config_yaml = yaml.full_load(f)
        f.close()
    config = WorkFlowConfig()
    config.from_dict(config_yaml)

    # Load data and create log record object
    # Currently read from local file. We can form LogRecordObject from data stream instead.
    data_loader = FileDataLoader(config.data_loader_config)
    logrecord = data_loader.load_data()

    # Get Log record type name. currently it is from the file name.
    lrt = config.data_loader_config.filepath.split("/")[-1].split('_')[0]

    # Execute Log Pattern Discovery Task
    parsing_res, logline_map, non_qualified_clusters = log_pattern_discovery_workflow(
        logrecord,
        config,
        lrt
    )

    # output results
    res_dir = config.workflow_config['output_dir']

    if not res_dir:
        raise RuntimeError("Need to provide an output directory!".format(res_dir))
    if not exists(dirname(res_dir)):
        try:
            os.makedirs(dirname(res_dir))
        except OSError as exc:
            if exc.errno != exc.errno.EEXIST:
                raise RuntimeError("{} is not a valid output directory!".format(res_dir))

    res_pattern_dir = os.path.join(config.workflow_config['output_dir'], "cluster_patterns.json")
    res_logline_map_dir = logline_map.to_csv(os.path.join(config.workflow_config['output_dir'], "logline_map.csv"))
    res_non_qualified_dir = os.path.join(config.workflow_config['output_dir'], "non_qualified_clusters.csv")

    pd.DataFrame.from_dict(parsing_res).to_json(res_pattern_dir, orient='records')
    logline_map.to_csv(res_logline_map_dir)
    pd.DataFrame(non_qualified_clusters, columns=["lrt", "cluster", "min_sim"]).to_csv(
        res_non_qualified_dir, index=False, header=True)

    return


if __name__ == "__main__":
    main()
