#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
"""
This file is wrapping the IMB/Drain3 implementation.
Link: https://github.com/IBM/Drain3/blob/master/drain3/drain.py
"""
from typing import List, Dict

import pandas as pd
from dataclasses import dataclass
from cachetools import LRUCache, Cache

from abc import ABC, abstractmethod

from logai.algorithms.algo_interfaces import ParsingAlgo
from logai.config_interfaces import Config
from logai.algorithms.factory import factory


@dataclass
class DrainParams(Config):
    """Parameters for Drain Log Parser. 
    For more details on parameters see 
    https://github.com/logpai/Drain3/blob/master/drain3/drain.py.

    :param depth: The depth of tree.
    :param sim_th: The similarity threshold.
    :param max_children: The max number of children nodes.
    :param max_clusters: The max number of clusters.
    :param extra_delimiters: Extra delimiters.
    :param param_str: The wildcard parameter string.
    """
    depth: int = 3
    sim_th: float = 0.4
    max_children: int = 100
    max_clusters: int = None
    extra_delimiters: tuple = ()
    param_str: str = "*"

    @classmethod
    def from_dict(cls, config_dict):
        config = super(DrainParams, cls).from_dict(config_dict)
        if config.extra_delimiters:
            config.extra_delimiters = tuple(config.extra_delimiters)
        return config


class Profiler(ABC):
    @abstractmethod
    def start_section(self, section_name: str):
        pass

    @abstractmethod
    def end_section(self, section_name=""):
        pass

    @abstractmethod
    def report(self, period_sec=30):
        pass


class NullProfiler(Profiler):
    """A no-op profiler. Use it instead of SimpleProfiler in case you want to disable profiling."""

    def start_section(self, section_name: str):
        pass

    def end_section(self, section_name=""):
        pass

    def report(self, period_sec=30):
        pass


class LogCluster:

    __slots__ = ["log_template_tokens", "cluster_id", "size"]

    def __init__(self, log_template_tokens: list, cluster_id: int):
        self.log_template_tokens = tuple(log_template_tokens)
        self.cluster_id = cluster_id
        self.size = 1

    def get_template(self):
        return " ".join(self.log_template_tokens)

    def __str__(self):
        return f"ID={str(self.cluster_id).ljust(5)} : size={str(self.size).ljust(10)}: {self.get_template()}"


class LogClusterCache(LRUCache):
    """
    Least Recently Used (LRU) cache which allows callers to conditionally skip
    cache eviction algorithm when accessing elements.
    """

    def __missing__(self, key):
        return None

    def get(self, key):
        """
        Returns the value of the item with the specified key without updating
        the cache eviction algorithm.
        """
        return Cache.__getitem__(self, key)


class Node:
    __slots__ = ["key_to_child_node", "cluster_ids"]

    def __init__(self):
        self.key_to_child_node: Dict[str, Node] = {}
        self.cluster_ids: List[int] = []


@factory.register("parsing", "drain", DrainParams)
class Drain(ParsingAlgo):
    def __init__(self, params: DrainParams, profiler=NullProfiler()):
        if params.depth < 3:
            raise ValueError("depth argument must be at least 3")

        self.log_cluster_depth = params.depth
        self.max_node_depth = (
            params.depth - 2
        )  # max depth of a prefix tree node, starting from zero
        self.sim_th = params.sim_th
        self.max_children = params.max_children
        self.root_node = Node()
        self.profiler = profiler
        self.extra_delimiters = params.extra_delimiters
        self.max_clusters = params.max_clusters
        self.param_str = params.param_str

        # key: int, value: LogCluster
        self.id_to_cluster = (
            {}
            if params.max_clusters is None
            else LogClusterCache(maxsize=params.max_clusters)
        )
        self.clusters_counter = 0

    @property
    def clusters(self):
        return self.id_to_cluster.values()

    @staticmethod
    def has_numbers(s):
        return any(char.isdigit() for char in s)

    def _tree_search(
        self, root_node: Node, tokens: list, sim_th: float, include_params: bool
    ):

        # at first level, children are grouped by token (word) count
        token_count = len(tokens)
        cur_node = root_node.key_to_child_node.get(str(token_count))

        # no template with same token count yet
        if cur_node is None:
            return None

        # handle case of empty log string - return the single cluster in that group
        if token_count == 0:
            return self.id_to_cluster.get(cur_node.cluster_ids[0])

        # find the leaf node for this log - a path of nodes matching the first N tokens (N=tree depth)
        cur_node_depth = 1
        for token in tokens:
            # at max depth
            if cur_node_depth >= self.max_node_depth:
                break

            # this is last token
            if cur_node_depth == token_count:
                break

            key_to_child_node = cur_node.key_to_child_node
            cur_node = key_to_child_node.get(token)
            if cur_node is None:  # no exact next token exist, try wildcard node
                cur_node = key_to_child_node.get(self.param_str)
            if cur_node is None:  # no wildcard node exist
                return None

            cur_node_depth += 1

        # get best match among all clusters with same prefix, or None if no match is above sim_th
        cluster = self._fast_match(cur_node.cluster_ids, tokens, sim_th, include_params)
        return cluster

    def _add_seq_to_prefix_tree(self, root_node, cluster: LogCluster):
        token_count = len(cluster.log_template_tokens)
        token_count_str = str(token_count)
        if token_count_str not in root_node.key_to_child_node:
            first_layer_node = Node()
            root_node.key_to_child_node[token_count_str] = first_layer_node
        else:
            first_layer_node = root_node.key_to_child_node[token_count_str]

        cur_node = first_layer_node

        # handle case of empty log string
        if token_count == 0:
            cur_node.cluster_ids = [cluster.cluster_id]
            return

        current_depth = 1
        for token in cluster.log_template_tokens:

            # if at max depth or this is last token in template - add current log cluster to the leaf node
            if current_depth >= self.max_node_depth or current_depth >= token_count:
                # clean up stale clusters before adding a new one.
                new_cluster_ids = []
                for cluster_id in cur_node.cluster_ids:
                    if cluster_id in self.id_to_cluster:
                        new_cluster_ids.append(cluster_id)
                new_cluster_ids.append(cluster.cluster_id)
                cur_node.cluster_ids = new_cluster_ids
                break

            # if token not matched in this layer of existing tree.
            if token not in cur_node.key_to_child_node:
                if not self.has_numbers(token):
                    if self.param_str in cur_node.key_to_child_node:
                        if len(cur_node.key_to_child_node) < self.max_children:
                            new_node = Node()
                            cur_node.key_to_child_node[token] = new_node
                            cur_node = new_node
                        else:
                            cur_node = cur_node.key_to_child_node[self.param_str]
                    else:
                        if len(cur_node.key_to_child_node) + 1 < self.max_children:
                            new_node = Node()
                            cur_node.key_to_child_node[token] = new_node
                            cur_node = new_node
                        elif len(cur_node.key_to_child_node) + 1 == self.max_children:
                            new_node = Node()
                            cur_node.key_to_child_node[self.param_str] = new_node
                            cur_node = new_node
                        else:
                            cur_node = cur_node.key_to_child_node[self.param_str]

                else:
                    if self.param_str not in cur_node.key_to_child_node:
                        new_node = Node()
                        cur_node.key_to_child_node[self.param_str] = new_node
                        cur_node = new_node
                    else:
                        cur_node = cur_node.key_to_child_node[self.param_str]

            # if the token is matched
            else:
                cur_node = cur_node.key_to_child_node[token]

            current_depth += 1

    # seq1 is template
    def _get_seq_distance(self, seq1, seq2, include_params: bool):
        assert len(seq1) == len(seq2)
        sim_tokens = 0
        param_count = 0

        for token1, token2 in zip(seq1, seq2):
            if token1 == self.param_str:
                param_count += 1
                continue
            if token1 == token2:
                sim_tokens += 1

        if include_params:
            sim_tokens += param_count

        ret_val = float(sim_tokens) / len(seq1)

        return ret_val, param_count

    def _fast_match(
        self, cluster_ids: list, tokens: list, sim_th: float, include_params: bool
    ):
        """
        Find the best match for a log message (represented as tokens) versus a list of clusters.

        :param cluster_ids: List of clusters to match against (represented by their IDs).
        :param tokens: the log message, separated to tokens.
        :param sim_th: minimum required similarity threshold (None will be returned in no clusters reached it).
        :param include_params: consider tokens matched to wildcard parameters in similarity threshold.
        :return: Best match cluster or None.
        """
        match_cluster = None

        max_sim = -1
        max_param_count = -1
        max_cluster = None

        for cluster_id in cluster_ids:
            # Try to retrieve cluster from cache with bypassing eviction
            # algorithm as we are only testing candidates for a match.
            cluster = self.id_to_cluster.get(cluster_id)
            if cluster is None:
                continue
            cur_sim, param_count = self._get_seq_distance(
                cluster.log_template_tokens, tokens, include_params
            )
            if cur_sim > max_sim or (
                cur_sim == max_sim and param_count > max_param_count
            ):
                max_sim = cur_sim
                max_param_count = param_count
                max_cluster = cluster

        if max_sim >= sim_th:
            match_cluster = max_cluster

        return match_cluster

    def _create_template(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        ret_val = list(seq2)

        for i, (token1, token2) in enumerate(zip(seq1, seq2)):
            if token1 != token2:
                ret_val[i] = self.param_str

        return ret_val

    def _print_tree(self, file=None, max_clusters=5):
        self._print_node("root", self.root_node, 0, file, max_clusters)

    def _print_node(self, token, node, depth, file, max_clusters):
        out_str = "\t" * depth

        if depth == 0:
            out_str += f"<{token}>"
        elif depth == 1:
            out_str += f"<L={token}>"
        else:
            out_str += f'"{token}"'

        if len(node.cluster_ids) > 0:
            out_str += f" (cluster_count={len(node.cluster_ids)})"

        print(out_str, file=file)

        for token, child in node.key_to_child_node.items():
            self._print_node(token, child, depth + 1, file, max_clusters)

        for cid in node.cluster_ids[:max_clusters]:
            cluster = self.id_to_cluster[cid]
            out_str = "\t" * (depth + 1) + str(cluster)
            print(out_str, file=file)

    def _get_content_as_tokens(self, content):
        content = content.strip()
        for delimiter in self.extra_delimiters:
            content = content.replace(delimiter, " ")
        content_tokens = content.split()
        return content_tokens

    def _add_log_message(self, content: str):
        content_tokens = self._get_content_as_tokens(content)

        if self.profiler:
            self.profiler.start_section("tree_search")
        match_cluster = self._tree_search(
            self.root_node, content_tokens, self.sim_th, False
        )
        if self.profiler:
            self.profiler.end_section()

        # Match no existing log cluster
        if match_cluster is None:
            if self.profiler:
                self.profiler.start_section("create_cluster")
            self.clusters_counter += 1
            cluster_id = self.clusters_counter
            match_cluster = LogCluster(content_tokens, cluster_id)
            self.id_to_cluster[cluster_id] = match_cluster
            self._add_seq_to_prefix_tree(self.root_node, match_cluster)
            update_type = "cluster_created"

        # Add the new log message to the existing cluster
        else:
            if self.profiler:
                self.profiler.start_section("cluster_exist")
            new_template_tokens = self._create_template(
                content_tokens, match_cluster.log_template_tokens
            )
            if tuple(new_template_tokens) == match_cluster.log_template_tokens:
                update_type = "none"
            else:
                match_cluster.log_template_tokens = tuple(new_template_tokens)
                update_type = "cluster_template_changed"
            match_cluster.size += 1
            # Touch cluster to update its state in the cache.
            # noinspection PyStatementEffect
            self.id_to_cluster[match_cluster.cluster_id]

        if self.profiler:
            self.profiler.end_section()

        return match_cluster, update_type

    def match(self, content: str):
        """
        Match against an already existing cluster. Match shall be perfect (sim_th=1.0).
        New cluster will not be created as a result of this call, nor any cluster modifications.

        :param content: The log message to match.
        :return: Matched cluster or None of no match found.
        """
        content_tokens = self._get_content_as_tokens(content)
        match_cluster = self._tree_search(self.root_node, content_tokens, 1.0, True)
        return match_cluster

    def _get_total_cluster_size(self):
        return self.clusters_counter

    def fit(self, logline: pd.Series):
        for l in logline:
            if not isinstance(l, str):
                continue
            self._add_log_message(l)

    def parse(self, logline: pd.Series) -> pd.Series:
        """Parse method to run log parser on a given log data.

        :param logline: The raw log data to be parsed.
        :returns: The parsed log data.
        """
        self.fit(logline)
        parsed_logline = []
        for line in logline:
            parsed_logline.append(" ".join(self.match(line).log_template_tokens))
        return pd.Series(parsed_logline, index=logline.index)
