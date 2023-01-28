#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import dash_bootstrap_components as dbc
from dash import dcc, html

from .utils import (
    create_description_card,
    create_modal,
    create_file_setting_layout,
    create_param_table,
)


def create_control_card():
    return html.Div(
        id="control-card",
        children=[
            # create_upload_file_layout(),
            create_file_setting_layout(),
            create_ad_algo_setting_layout(),
            html.Hr(),
            html.Div(
                children=[html.Button(id="anomaly-btn", children="Run", n_clicks=0)],
                style={"textAlign": "center"},
            ),
            create_modal(
                modal_id="anomaly_exception_modal",
                header="An Exception Occurred",
                content="An exception occurred. Please click OK to continue.",
                content_id="anomaly_exception_modal_content",
                button_id="anomaly_exception_modal_close",
            ),
        ],
    )


def create_ad_algo_setting_layout():
    return html.Div(
        id="algo-setting-layout",
        children=[
            html.Br(),
            html.B("Parsing Algortihm"),
            dcc.Dropdown(
                id="parsing-algo-select",
                options=["DRAIN", "IPLoM", "AEL"],
                value="DRAIN",
            ),
            html.Div(id="ad-parsing-param-table", children=[create_param_table()]),
            html.Br(),
            html.B("Vectorization Algortihm"),
            dcc.Dropdown(
                id="vectorization-algo-select",
                options=["word2vec", "tfidf", "fasttext"],
                value="word2vec",
            ),
            html.Br(),
            html.B("Categorical Encoder"),
            dcc.Dropdown(
                id="categorical-encoder-select",
                options=["label_encoder", "one_hot_encoder", "ordinal_encoder"],
                value="label_encoder",
            ),
            html.Br(),
            html.B("Anomaly Detection Algortihm"),
            dcc.Dropdown(
                id="ad-algo-select",
                options=[
                    "one_class_svm",
                    "isolation_forest",
                    "LOF",
                    "distribution_divergence",
                    "dbl",
                    "ets",
                ],
                value="one_class_svm",
            ),
            html.Div(id="ad-param-table", children=[create_param_table()]),
        ],
    )


def create_display_layout():
    return html.Div(
        id="result_table_card_anomaly",
        children=[
            html.B("Timeseries"),
            html.Hr(),
            dbc.Card(
                dbc.CardBody(
                    [
                        dcc.Loading(
                            id="loading-timechart",
                            children=[dbc.Row(dcc.Graph(id="time_chart"))],
                            type="default",
                        )
                    ],
                    style={"marginTop": 0, "marginBottom": 0},
                ),
            ),
            html.B("Anomalies"),
            html.Hr(),
            dbc.Card(
                dbc.CardBody(
                    [
                        dcc.Loading(
                            id="loading-anomaly-table",
                            children=[html.Div(id="anomaly-table")],
                            type="default",
                        )
                    ]
                ),
            ),
        ],
    )


def create_anomaly_detection_layout():
    return dbc.Row(
        [
            # Left column
            dbc.Col(
                html.Div(
                    [
                        create_description_card(),
                        create_control_card(),
                        html.Div(
                            ["initial child"],
                            id="output-clientside",
                            style={"display": "none"},
                        ),
                    ]
                ),
                width=2,
            ),
            dbc.Col(
                [
                    dbc.Row(
                        [
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H4("Summary"),
                                            html.Div(
                                                id="anomaly-summary"
                                            ),  # Add log AD summary
                                        ]
                                    )
                                ),
                                width=4,
                            ),
                            dbc.Col(
                                dbc.Card(
                                    dbc.CardBody(
                                        [
                                            html.H4("Attributes"),
                                            html.Div(id="anomaly-attribute-options"),
                                        ]
                                    )
                                ),
                                width=8,
                            ),
                        ]
                    ),
                    create_display_layout(),
                ]
            ),
        ],
    )


layout = create_anomaly_detection_layout()
