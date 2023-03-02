#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
from logai.applications.openset.anomaly_detection.openset_anomaly_detection_workflow import (
    OpenSetADWorkflow,
    get_openset_ad_config,
)
import os
import pytest

TEST_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "test_data/BGL_AD/BGL_11k.log"
)
TEST_OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "test_data/BGL_AD/output"
)


class TestOpenSetLogAnomalyDetection:
    def _setup(self, config):
        config.data_loader_config.filepath = TEST_DATA_PATH
        config.output_dir = TEST_OUTPUT_PATH
        if not os.path.exists(config.output_dir):
            os.makedirs(config.output_dir)

    @pytest.mark.skip(reason="currently not testing this as it is time and memory consuming")
    def test_bgl_logbert_ad(self):
        kwargs = {
            "config_filename": "bgl",
            "anomaly_detection_type": "logbert_AD",
            "vectorizer_type": "logbert",
            "parse_logline": False,
            "training_type": "unsupervised"
        }

        config = get_openset_ad_config(**kwargs)
        self._setup(config)
        workflow = OpenSetADWorkflow(config)
        workflow.execute()

    def test_bgl_lstm_sequential_unsupervised_parsed_ad(self):
        kwargs = {
            "config_filename": "bgl",
            "anomaly_detection_type": "lstm_sequential_unsupervised_parsed_AD",
            "vectorizer_type": "forecast_nn_sequential",
            "parse_logline": True,
            "training_type": "unsupervised",
        }
        config = get_openset_ad_config(**kwargs)
        self._setup(config)
        workflow = OpenSetADWorkflow(config)
        workflow.execute()

    @pytest.mark.skip(reason="currently not testing this")
    def test_bgl_lstm_sequential_supervised_parsed_ad(self):
        kwargs = {
            "config_filename": "bgl",
            "anomaly_detection_type": "lstm_sequential_supervised_parsed_AD",
            "vectorizer_type": "forecast_nn_sequential",
            "parse_logline": True,
            "training_type": "supervised"
        }
        config = get_openset_ad_config(**kwargs)
        self._setup(config)
        workflow = OpenSetADWorkflow(config)
        workflow.execute()

    @pytest.mark.skip(reason="currently not testing this")
    def test_bgl_lstm_sequential_unsupervised_nonparsed_ad(self):
        kwargs = {
            "config_filename": "bgl",
            "anomaly_detection_type": "lstm_sequential_unsupervised_nonparsed_AD",
            "vectorizer_type": "forecast_nn_sequential",
            "parse_logline": False,
            "training_type": "unsupervised"
        }
        config = get_openset_ad_config(**kwargs)
        self._setup(config)
        workflow = OpenSetADWorkflow(config)
        workflow.execute()

    @pytest.mark.skip(reason="currently not testing this")
    def test_bgl_lstm_sequential_supervised_nonparsed_ad(self):
        kwargs = {
            "config_filename": "bgl",
            "anomaly_detection_type": "lstm_sequential_supervised_nonparsed_AD",
            "vectorizer_type": "forecast_nn_sequential",
            "parse_logline": False,
            "training_type": "supervised"
        }
        config = get_openset_ad_config(**kwargs)
        self._setup(config)
        workflow = OpenSetADWorkflow(config)
        workflow.execute()

    @pytest.mark.skip(reason="currently not testing this")
    def test_bgl_lstm_semantics_supervised_nonparsed_ad(self):
        kwargs = {
            "config_filename": "bgl",
            "anomaly_detection_type": "lstm_semantics_supervised_nonparsed_AD",
            "vectorizer_type": "forecast_nn_sequential",
            "parse_logline": True,
            "training_type": "supervised"
        }
        config = get_openset_ad_config(**kwargs)
        self._setup(config)
        workflow = OpenSetADWorkflow(config)
        workflow.execute()

    @pytest.mark.skip(reason="currently not testing this")
    def test_bgl_lstm_semantics_unsupervised_nonparsed_ad(self):
        kwargs = {
            "config_filename": "bgl",
            "anomaly_detection_type": "lstm_semantics_unsupervised_nonparsed_AD",
            "vectorizer_type": "forecast_nn_sequential",
            "parse_logline": True,
            "training_type": "unsupervised"
        }
        config = get_openset_ad_config(**kwargs)
        self._setup(config)
        workflow = OpenSetADWorkflow(config)
        workflow.execute()

    def test_bgl_transformer_sequential_unsupervised_parsed_ad(self):
        kwargs = {
            "config_filename": "bgl",
            "anomaly_detection_type": "transformer_sequential_unsupervised_parsed_AD",
            "vectorizer_type": "forecast_nn_sequential",
            "parse_logline": True,
            "training_type": "unsupervised"
        }
        config = get_openset_ad_config(**kwargs)
        self._setup(config)
        workflow = OpenSetADWorkflow(config)
        workflow.execute()

    @pytest.mark.skip(reason="currently not testing this")
    def test_bgl_transformer_sequential_supervised_parsed_ad(self):
        kwargs = {
            "config_filename": "bgl",
            "anomaly_detection_type": "transformer_sequential_supervised_parsed_AD",
            "vectorizer_type": "forecast_nn_sequential",
            "parse_logline": True,
            "training_type": "supervised"
        }
        config = get_openset_ad_config(**kwargs)
        self._setup(config)
        workflow = OpenSetADWorkflow(config)
        workflow.execute()

    @pytest.mark.skip(reason="currently not testing this")
    def test_bgl_transformer_sequential_unsupervised_nonparsed_ad(self):
        kwargs = {
            "config_filename": "bgl",
            "anomaly_detection_type": "transformer_sequential_unsupervised_nonparsed_AD",
            "vectorizer_type": "forecast_nn_sequential",
            "parse_logline": False,
            "training_type": "unsupervised"
        }
        config = get_openset_ad_config(**kwargs)
        self._setup(config)
        workflow = OpenSetADWorkflow(config)
        workflow.execute()

    @pytest.mark.skip(reason="currently not testing this")
    def test_bgl_transformer_sequential_supervised_nonparsed_ad(self):
        kwargs = {
            "config_filename": "bgl",
            "anomaly_detection_type": "transformer_sequential_supervised_nonparsed_AD",
            "vectorizer_type": "forecast_nn_sequential",
            "parse_logline": False,
            "training_type": "supervised"
        }
        config = get_openset_ad_config(**kwargs)
        self._setup(config)
        workflow = OpenSetADWorkflow(config)
        workflow.execute()

    @pytest.mark.skip(reason="currently not testing this")
    def test_bgl_transformer_semantics_supervised_nonparsed_ad(self):
        kwargs = {
            "config_filename": "bgl",
            "anomaly_detection_type": "transformer_semantics_supervised_nonparsed_AD",
            "vectorizer_type": "forecast_nn_sequential",
            "parse_logline": True,
            "training_type": "supervised"
        }
        config = get_openset_ad_config(**kwargs)
        self._setup(config)
        workflow = OpenSetADWorkflow(config)
        workflow.execute()

    @pytest.mark.skip(reason="currently not testing this")
    def test_bgl_transformer_semantics_unsupervised_nonparsed_ad(self):
        kwargs = {
            "config_filename": "bgl",
            "anomaly_detection_type": "transformer_semantics_unsupervised_nonparsed_AD",
            "vectorizer_type": "forecast_nn_sequential",
            "parse_logline": True,
            "training_type": "unsupervised"
        }
        config = get_openset_ad_config(**kwargs)
        self._setup(config)
        workflow = OpenSetADWorkflow(config)
        workflow.execute()

    def test_bgl_cnn_sequential_unsupervised_parsed_ad(self):
        kwargs = {
            "config_filename": "bgl",
            "anomaly_detection_type": "cnn_sequential_unsupervised_parsed_AD",
            "vectorizer_type": "forecast_nn_sequential",
            "parse_logline": True,
            "training_type": "unsupervised"
        }
        config = get_openset_ad_config(**kwargs)
        self._setup(config)
        workflow = OpenSetADWorkflow(config)
        workflow.execute()

    @pytest.mark.skip(reason="currently not testing this")
    def test_bgl_cnn_sequential_supervised_parsed_ad(self):
        kwargs = {
            "config_filename": "bgl",
            "anomaly_detection_type": "cnn_sequential_supervised_parsed_AD",
            "vectorizer_type": "forecast_nn_sequential",
            "parse_logline": True,
            "training_type": "supervised"
        }
        config = get_openset_ad_config(**kwargs)
        self._setup(config)
        workflow = OpenSetADWorkflow(config)
        workflow.execute()

    @pytest.mark.skip(reason="currently not testing this")
    def test_bgl_cnn_sequential_unsupervised_nonparsed_ad(self):
        kwargs = {
            "config_filename": "bgl",
            "anomaly_detection_type": "cnn_sequential_unsupervised_nonparsed_AD",
            "vectorizer_type": "forecast_nn_sequential",
            "parse_logline": False,
            "training_type": "unsupervised"
        }
        config = get_openset_ad_config(**kwargs)
        self._setup(config)
        workflow = OpenSetADWorkflow(config)
        workflow.execute()

    @pytest.mark.skip(reason="currently not testing this")
    def test_bgl_cnn_sequential_supervised_nonparsed_ad(self):
        kwargs = {
            "config_filename": "bgl",
            "anomaly_detection_type": "cnn_sequential_supervised_nonparsed_AD",
            "vectorizer_type": "forecast_nn_sequential",
            "parse_logline": False,
            "training_type": "supervised"
        }
        config = get_openset_ad_config(**kwargs)
        self._setup(config)
        workflow = OpenSetADWorkflow(config)
        workflow.execute()

    @pytest.mark.skip(reason="currently not testing this")
    def test_bgl_cnn_semantics_supervised_nonparsed_ad(self):
        kwargs = {
            "config_filename": "bgl",
            "anomaly_detection_type": "cnn_semantics_supervised_nonparsed_AD",
            "vectorizer_type": "forecast_nn_sequential",
            "parse_logline": True,
            "training_type": "supervised"
        }
        config = get_openset_ad_config(**kwargs)
        self._setup(config)
        workflow = OpenSetADWorkflow(config)
        workflow.execute()

    @pytest.mark.skip(reason="currently not testing this")
    def test_bgl_cnn_semantics_unsupervised_nonparsed_ad(self):
        kwargs = {
            "config_filename": "bgl",
            "anomaly_detection_type": "cnn_semantics_unsupervised_nonparsed_AD",
            "vectorizer_type": "forecast_nn_sequential",
            "parse_logline": True,
            "training_type": "unsupervised"
        }
        config = get_openset_ad_config(**kwargs)
        self._setup(config)
        workflow = OpenSetADWorkflow(config)
        workflow.execute()
