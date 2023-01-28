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
            create_summarization_algo_setting_layout(),
            html.Hr(),
            html.Div(
                children=[html.Button(id="pattern-btn", children="Run", n_clicks=0)],
                style={"textAlign": "center"},
            ),
            create_modal(
                modal_id="pattern_exception_modal",
                header="An Exception Occurred",
                content="An exception occurred. Please click OK to continue.",
                content_id="pattern_exception_modal_content",
                button_id="pattern_exception_modal_close",
            ),
        ],
    )


def create_summarization_algo_setting_layout():
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
            html.Div(id="parsing-param-table", children=[create_param_table()]),
        ],
    )


def create_summary_graph_layout():
    return html.Div(
        dcc.Graph(id="summary-scatter"),
        # style={'width': '39%', 'display': 'inline-block', 'padding': '0 20'}
    )


def create_timeseries_grapy_layout():
    return html.Div(
        children=[
            dcc.Graph(id="pattern-time-series"),
        ],
        # style={
        #     'display': 'inline-block',
        #     'width': '59%'
        # },
    )


def create_pattern_layout():
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
            # Right column
            dbc.Col(
                html.Div(
                    [
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                html.H4("Summary"),
                                                html.Div(
                                                    id="log-summarization-summary"
                                                ),
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
                                                html.Div(id="attribute-options"),
                                            ]
                                        )
                                    ),
                                    width=8,
                                ),
                            ]
                        ),
                        html.B("Charts"),
                        html.Hr(),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                dcc.Loading(
                                                    [
                                                        create_summary_graph_layout(),
                                                    ]
                                                )
                                            ]
                                        )
                                    ),
                                    width=4,
                                ),
                                dbc.Col(
                                    dbc.Card(
                                        dbc.CardBody(
                                            [
                                                dcc.Loading(
                                                    [
                                                        create_timeseries_grapy_layout(),
                                                    ]
                                                )
                                            ]
                                        )
                                    ),
                                    width=8,
                                ),
                            ],
                        ),
                        html.B("Log Patterns"),
                        html.Hr(),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.Div(
                                        id="log-patterns",
                                    )
                                ],
                            ),
                            id="pattern-log-card",
                        ),
                        html.B("Dynamic Values"),
                        html.Hr(),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dcc.Loading(
                                        id="loading-dynamic-values",
                                        children=[html.Div(id="log-dynamic-lists")],
                                        type="default",
                                    )
                                ],
                            ),
                            id="pattern-dynamic-values",
                        ),
                        html.B("Log Lines"),
                        html.Hr(),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dcc.Loading(
                                        id="loading-loglines",
                                        children=[
                                            dbc.Row(
                                                dbc.Col(html.Div(id="select-loglines"))
                                            )
                                        ],
                                        type="default",
                                    )
                                ]
                            ),
                            id="result_table_card",
                            style={"maxwidth": "900px"},
                        ),
                    ]
                )
            ),
        ]
    )


layout = create_pattern_layout()
