#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os

import dash
import pandas as pd
from dash import html, Input, Output, State, callback, dash_table
import plotly.express as px

from gui.demo.log_pattern import LogPattern
from gui.demo.log_clustering import Clustering
from gui.file_manager import FileManager
from logai.analysis.clustering import ClusteringConfig
from logai.applications.application_interfaces import WorkFlowConfig
from logai.dataloader.openset_data_loader import OpenSetDataLoaderConfig
from logai.information_extraction.categorical_encoder import CategoricalEncoderConfig
from logai.information_extraction.feature_extractor import FeatureExtractorConfig
from logai.information_extraction.log_parser import LogParserConfig
from logai.information_extraction.log_vectorizer import VectorizerConfig
from logai.preprocess.preprocessor import PreprocessorConfig

from ..pages.utils import create_param_table

log_clustering = Clustering()
file_manager = FileManager()


def _clustering_config():
    config = WorkFlowConfig(
        open_set_data_loader_config=OpenSetDataLoaderConfig(),
        preprocessor_config=PreprocessorConfig(),
        feature_extractor_config=FeatureExtractorConfig(group_by_time="1s"),
        log_parser_config=LogParserConfig(),
        log_vectorizer_config=VectorizerConfig(),
        categorical_encoder_config=CategoricalEncoderConfig(),
        clustering_config=ClusteringConfig(),
    )
    return config


def create_attribute_component(attributes):
    print(attributes)
    table = dash_table.DataTable(
        id="cluster-attribute-table",
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

    return html.Div(children=[table, html.Div(id="table-dropdown-container")])


@callback(
    Output("clustering-attribute-options", "children"),
    Output("clustering_exception_modal", "is_open"),
    Output("clustering_exception_modal_content", "children"),
    [
        Input("clustering-btn", "n_clicks"),
        Input("clustering_exception_modal_close", "n_clicks"),
    ],
    [
        State("log-type-select", "value"),
        State("attribute-name-options", "value"),
        State("file-select", "value"),
        State("parsing-algo-select", "value"),
        State("vectorization-algo-select", "value"),
        State("categorical-encoder-select", "value"),
        State("clustering-algo-select", "value"),
        State("clustering-param-table", "children"),
        State("clustering-parsing-param-table", "children"),
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
    clustering_algo,
    clustering_param_table,
    parsing_param_table,
):
    ctx = dash.callback_context
    if ctx.triggered:
        prop_id = ctx.triggered[0]["prop_id"].split(".")[0]
        if prop_id == "clustering-btn":
            try:
                file_path = os.path.join(file_manager.base_directory, filename)
                clustering_params = log_clustering.parse_parameters(
                    param_info=log_clustering.get_parameter_info(clustering_algo),
                    params={
                        p["Parameter"]: p["Value"]
                        for p in clustering_param_table["props"]["data"]
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

                config = _clustering_config()
                config.open_set_data_loader_config.filepath = (
                    file_path  # overwrite the file path.
                )
                config.open_set_data_loader_config.dataset_name = log_type
                config.feature_extractor_config.group_by_category = attributes
                config.log_parser_config.parsing_algorithm = parsing_algo

                config_class = LogPattern().get_config_class(parsing_algo)
                config.log_parser_config.parsing_algo_params = config_class.from_dict(
                    parsing_params
                )

                config.log_vectorizer_config.algo_name = vectorization_algo
                config.categorical_encoder_config.algo_name = categorical_encoder
                config.clustering_config.algo_name = clustering_algo

                config_class = log_clustering.get_config_class(clustering_algo)
                config.clustering_config.algo_params = config_class.from_dict(
                    clustering_params
                )

                log_clustering.execute_clustering(config)

                return (
                    create_attribute_component(log_clustering.get_attributes()),
                    False,
                    "",
                )
            except Exception as error:
                return html.Div(), True, str(error)
        elif prop_id == "clustering_exception_modal_close":
            return html.Div(), False, ""
    else:
        return html.Div(), False, ""


@callback(Output("cluster-hist", "figure"), [Input("cluster-attribute-table", "data")])
def update_hist(data):
    res = log_clustering.get_unique_clusters()

    df = pd.DataFrame.from_dict(res, orient="index")
    df.index.name = "Cluster"
    df.columns = ["Size"]
    df["Cluster"] = df.index.values
    return generate_pie_chart(df)


def generate_pie_chart(df):
    fig = px.pie(df, names="Cluster", values="Size")

    return fig


@callback(
    Output("clustering-loglines", "children"), [Input("cluster-hist", "clickData")]
)
def update_logline_list(data):
    if len(data) > 0:
        cluster_label = data["points"][0]["label"]
        # return html.Div(str(data['points'][0])) # for debug
        df = log_clustering.get_loglines(cluster_label)

        columns = [{"name": c, "id": c} for c in df.columns]
        return dash_table.DataTable(
            data=df.to_dict("records"),
            columns=columns,
            style_table={"overflowX": "scroll"},
            style_cell={
                "max-width": "1020px",
                "textAlign": "left",
            },
            editable=True,
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
    Output("clustering-summary", "children"),
    [
        Input("cluster-attribute-table", "data"),
    ],
)
def clustering_summary(data):
    if len(data) == 0:
        return html.Div()

    result_table = log_clustering.result_table

    total_loglines = result_table.shape[0]
    total_num_cluster = len(result_table["cluster_id"].unique())

    return html.Div(
        [
            html.P("Total Number Of Loglines: {}".format(total_loglines)),
            html.P("Total Number Of Log Clusters: {}".format(total_num_cluster)),
        ]
    )


@callback(
    Output("clustering-param-table", "children"),
    Input("clustering-algo-select", "value"),
)
def select_clustering_algorithm(algorithm):
    param_info = log_clustering.get_parameter_info(algorithm)
    param_table = create_param_table(param_info)
    return param_table


@callback(
    Output("clustering-parsing-param-table", "children"),
    Input("parsing-algo-select", "value"),
)
def select_parsing_algorithm(algorithm):
    param_info = LogPattern().get_parameter_info(algorithm)
    param_table = create_param_table(param_info)
    return param_table
