#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import html
import os

import dash
import pandas as pd
from dash import html, Input, Output, State, callback, dash_table
import plotly.express as px

from gui.demo.log_pattern import LogPattern
from gui.demo.log_anomaly import LogAnomaly
from gui.file_manager import FileManager
from logai.analysis.anomaly_detector import AnomalyDetectionConfig
from logai.applications.application_interfaces import WorkFlowConfig
from logai.dataloader.openset_data_loader import OpenSetDataLoaderConfig
from logai.information_extraction.categorical_encoder import CategoricalEncoderConfig
from logai.information_extraction.feature_extractor import FeatureExtractorConfig
from logai.information_extraction.log_parser import LogParserConfig
from logai.information_extraction.log_vectorizer import VectorizerConfig
from logai.preprocess.preprocessor import PreprocessorConfig

from logai.utils import constants
from ..pages.utils import create_param_table

log_anomaly_demo = LogAnomaly()
file_manager = FileManager()


def _ad_config_sample():
    config = WorkFlowConfig(
        open_set_data_loader_config=OpenSetDataLoaderConfig(),
        preprocessor_config=PreprocessorConfig(),
        feature_extractor_config=FeatureExtractorConfig(group_by_time="1s"),
        log_parser_config=LogParserConfig(),
        log_vectorizer_config=VectorizerConfig(),
        categorical_encoder_config=CategoricalEncoderConfig(),
        anomaly_detection_config=AnomalyDetectionConfig(algo_name="lof"),
    )
    return config


def create_attribute_component(attributes):
    table = dash_table.DataTable(
        id="anomaly-attribute-table",
        data=attributes.iloc[:1].to_dict("records"),
        columns=[
            {"id": c, "name": c, "presentation": "dropdown"} for c in attributes.columns
        ],
        editable=True,
        dropdown={
            c: {"options": [{"label": i, "value": i} for i in attributes[c].unique()]}
            for c in attributes.columns
        },
        style_header_conditional=[{"textAlign": "left"}],
        style_cell_conditional=[{"textAlign": "left"}],
    )
    # print(table)
    return html.Div(children=[table, html.Div(id="table-dropdown-container")])


@callback(
    Output("anomaly-attribute-options", "children"),
    Output("anomaly_exception_modal", "is_open"),
    Output("anomaly_exception_modal_content", "children"),
    [
        Input("anomaly-btn", "n_clicks"),
        Input("anomaly_exception_modal_close", "n_clicks"),
    ],
    [
        State("log-type-select", "value"),
        State("attribute-name-options", "value"),
        State("file-select", "value"),
        State("parsing-algo-select", "value"),
        State("vectorization-algo-select", "value"),
        State("categorical-encoder-select", "value"),
        State("ad-algo-select", "value"),
        State("time-interval", "value"),
        State("ad-param-table", "children"),
        State("ad-parsing-param-table", "children"),
    ],
)
def click_run(
    btn_click,
    modal_close,
    log_type,
    attributes,
    filename,
    parsing_algo,
    vectorization_algo,
    categorical_encoder,
    ad_algo,
    time_interval,
    ad_param_table,
    parsing_param_table,
):
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "anomaly-btn":
            try:
                interval_map = {0: "1s", 1: "1min", 2: "1h", 3: "1d"}
                freq = interval_map[time_interval]

                file_path = os.path.join(file_manager.base_directory, filename)
                ad_params = log_anomaly_demo.parse_parameters(
                    param_info=log_anomaly_demo.get_parameter_info(ad_algo),
                    params={
                        p["Parameter"]: p["Value"]
                        for p in ad_param_table["props"]["data"]
                        if p["Parameter"]
                    },
                )
                parsing_params = LogPattern().parse_parameters(
                    param_info=LogPattern().get_parameter_info(parsing_algo),
                    params={
                        p["Parameter"]: p["Value"]
                        for p in parsing_param_table["props"]["data"]
                        if p["Parameter"]
                    },
                )

                config = _ad_config_sample()
                config.open_set_data_loader_config.filepath = (
                    file_path  # overwrite the file path.
                )
                config.open_set_data_loader_config.dataset_name = log_type
                config.feature_extractor_config.group_by_category = attributes
                config.feature_extractor_config.group_by_time = freq
                config.log_parser_config.parsing_algorithm = parsing_algo

                config_class = LogPattern().get_config_class(parsing_algo)
                config.log_parser_config.parsing_algo_params = config_class.from_dict(
                    parsing_params
                )

                config.log_vectorizer_config.algo_name = vectorization_algo
                config.categorical_encoder_config.algo_name = categorical_encoder
                config.anomaly_detection_config.algo_name = ad_algo

                config_class = log_anomaly_demo.get_config_class(ad_algo)
                config.anomaly_detection_config.algo_params = config_class.from_dict(
                    ad_params
                )

                log_anomaly_demo.execute_anomaly_detection(config)

                return (
                    create_attribute_component(log_anomaly_demo.get_attributes()),
                    False,
                    "",
                )
            except Exception as error:
                return html.Div(), True, str(error)
        elif prop_id == "anomaly_exception_modal_close":
            return html.Div(), False, ""
    else:
        return html.Div(), False, ""


@callback(
    Output("anomaly-summary", "children"),
    [
        Input("anomaly-attribute-table", "data"),
    ],
)
def summary(data):
    if len(data) == 0:
        return html.Div()

    res_df = log_anomaly_demo.get_results()
    anomaly_df = res_df[res_df["is_anomaly"]]
    total_loglines = res_df.shape[0]
    total_groups = len(res_df["group_id"].unique())
    total_anomalies = anomaly_df.shape[0]
    anomaly_group = len(anomaly_df["group_id"].unique())

    return html.Div(
        [
            html.P("Total Number Of Loglines: {}".format(total_loglines)),
            html.P("Total Number Of Log Groups: {}".format(total_groups)),
            html.P("Total Number Of Anomalous Groups: {}".format(anomaly_group)),
            html.P("Total Number of Anomalous Loglines: {}".format(total_anomalies)),
        ]
    )


@callback(
    Output("time_chart", "figure"),
    [Input("anomaly-attribute-table", "data"), Input("time-interval", "value")],
)
def update_graph(data, interval):
    interval_map = {0: "1s", 1: "1min", 2: "1h", 3: "1d"}
    freq = interval_map[interval]
    if len(data) > 0:
        attributes = data[0]
        df = log_anomaly_demo.get_results(attributes)
        ts_df = (
            df[["timestamp", "is_anomaly"]]
            .groupby(pd.Grouper(key="timestamp", freq=freq, offset=0, label="right"))
            .apply(lambda x: pd.Series((x.shape[0], x[x["is_anomaly"]].shape[0])))
        )

        ts_df.columns = ["counts", "anomaly_counts"]

        ts_df = ts_df.reset_index()

        fig = px.line(
            ts_df,
            x="timestamp",
            y=["counts", "anomaly_counts"],
            line_shape="linear",
            markers=True,
        )

        return fig
    else:
        return {}


@callback(
    Output("counter_table", "children"), [Input("anomaly-attribute-table", "data")]
)
def update_counter_table(data):
    # print(attributes)
    if len(data) > 0:
        df = log_anomaly_demo.get_counter(data[0])[
            ["timestamp", constants.LOGLINE_COUNTS]
        ]
        columns = [{"name": c, "id": c} for c in df.columns]
        return dash_table.DataTable(
            data=df.to_dict("records"),
            columns=columns,
            style_table={"overflowX": "scroll"},
            style_cell={
                "max-width": "900px",
                "textAlign": "left",
            },
            editable=False,
            row_selectable="multi",
            sort_action="native",
            sort_mode="multi",
            column_selectable="single",
            page_action="native",
            page_size=20,
            page_current=0,
        )
    else:
        return dash_table.DataTable()


@callback(
    Output("anomaly-table", "children"), [Input("anomaly-attribute-table", "data")]
)
def update_anomaly_table(data):
    if len(data) > 0:
        df = log_anomaly_demo.get_anomalies(data[0])[
            [constants.LOG_TIMESTAMPS, constants.LOGLINE_NAME]
        ]
        columns = [{"name": c, "id": c} for c in df.columns]
        return dash_table.DataTable(
            data=df.to_dict("records"),
            columns=columns,
            style_table={"overflowX": "scroll"},
            style_cell={
                "max-width": "900px",
                "textAlign": "left",
            },
            editable=False,
            row_selectable="multi",
            sort_action="native",
            sort_mode="multi",
            column_selectable="single",
            page_action="native",
            page_size=20,
            page_current=0,
        )

    else:
        return dash_table.DataTable()


@callback(Output("ad-param-table", "children"), Input("ad-algo-select", "value"))
def select_ad_algorithm(algorithm):
    param_info = log_anomaly_demo.get_parameter_info(algorithm)
    param_table = create_param_table(param_info)
    return param_table


@callback(
    Output("ad-parsing-param-table", "children"), Input("parsing-algo-select", "value")
)
def select_parsing_algorithm(algorithm):
    param_info = LogPattern().get_parameter_info(algorithm)
    param_table = create_param_table(param_info)
    return param_table
