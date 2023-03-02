#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import pandas as pd
import pytest
from pytest import approx

from logai.algorithms.parsing_algo.drain import DrainParams, Drain
from tests.logai.test_utils.fixtures import logrecord_body


class TestDrainConfig:
    def setup(self):
        self.ERROR = 0.01

    def test_default_drain_config(self):
        params = DrainParams()
        assert isinstance(params, DrainParams)
        assert params.sim_th == pytest.approx(0.4, self.ERROR)
        assert not params.extra_delimiters

    def test_assigned_drain_config(self):
        params = DrainParams(sim_th=0.6, extra_delimiters=tuple(','))
        assert isinstance(params, DrainParams)
        assert params.sim_th == pytest.approx(0.6, self.ERROR)
        assert ',' in params.extra_delimiters

    def test_from_dict(self):
        config_dict = {
            "sim_th": 0.2,
            "extra_delimiters": [',']
        }

        config = DrainParams.from_dict(config_dict)
        assert config.sim_th == approx(0.2), "sim_th is not 0.2"
        assert type(config.extra_delimiters) is tuple, "extra_delimiters is not tuple"
        assert "," in config.extra_delimiters, "extra_delimiters doesn't contain comma"


class TestDrain:
    def setup(self):
        self.params = DrainParams()

    def test_fit_parse(self, logrecord_body):
        parser = Drain(self.params)
        assert isinstance(parser, Drain)
        parsed_loglines = parser.parse(logrecord_body['logline'])
        assert parser.clusters_counter > 0, "log cluster number should be greater than zero after fit"
        assert isinstance(parser, Drain)
        assert isinstance(parsed_loglines, pd.Series), 'parse returns pandas.Series'
