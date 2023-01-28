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
    create_modal,
    create_description_card,
    create_file_setting_layout,
    create_param_table,
)


def create_control_card():
    return html.Div(
        id="control-card",
        children=[
            # create_upload_file_layout(),
            create_file_setting_layout(),
            create_clustering_algo_setting_layout(),
            html.Hr(),
            html.Div(
                children=[html.Button(id="clustering-btn", children="Run", n_clicks=0)],
                style={"textAlign": "center"},
            ),
            create_modal(
                modal_id="clustering_exception_modal",
                header="An Exception Occurred",
                content="An exception occurred. Please click OK to continue.",
                content_id="clustering_exception_modal_content",
                button_id="clustering_exception_modal_close",
            ),
        ],
    )


def create_clustering_algo_setting_layout():
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
            html.Div(
                id="clustering-parsing-param-table", children=[create_param_table()]
            ),
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
            html.B("Clustering Algortihm"),
            dcc.Dropdown(
                id="clustering-algo-select",
                options=["DBSCAN", "kmeans"],
                value="kmeans",
            ),
            html.Div(id="clustering-param-table", children=[create_param_table()]),
        ],
        style={
            "display": "inline-block",
            "width": "100%",
        },
    )


def create_display_layout():
    return html.Div(
        id="result-clustering-table-card",
        children=[
            html.B("Clustering Summary"),
            html.Hr(),
            dbc.Card(
                dbc.CardBody(
                    [
                        dcc.Loading(
                            id="clstering-summary",
                            children=[dbc.Row(dcc.Graph(id="cluster-hist"))],
                            type="default",
                        )
                    ],
                    style={"marginTop": 0, "marginBottom": 0},
                )
            ),
            html.B("Loglines in Cluster"),
            html.Hr(),
            html.Div(id="clustering-loglines"),
        ],
    )


def create_clustering_layout():
    return dbc.Row(
        [
            dbc.Col(
                [
                    # Left column
                    create_description_card(),
                    create_control_card(),
                    html.Div(
                        ["initial child"],
                        id="clustering-output-clientside",
                        style={"display": "none"},
                    ),
                ],
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
                                                id="clustering-summary"
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
                                            html.Div(id="clustering-attribute-options"),
                                        ]
                                    )
                                ),
                                width=8,
                            ),
                        ]
                    ),
                    dbc.Row(
                        [
                            dbc.Col(
                                [
                                    create_display_layout(),
                                ]
                            )
                        ]
                    ),
                ]
            ),
        ],
    )


layout = create_clustering_layout()
