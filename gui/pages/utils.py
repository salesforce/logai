#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import logging
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table

STYLE = {
    "json-output": {
        "overflow-y": "scroll",
        "height": "calc(90% - 25px)",
        "border": "thin lightgrey solid",
    },
    "tab": {"height": "calc(98vh - 80px)"},
    "log-output": {
        "overflow-y": "scroll",
        "height": "calc(90% - 25px)",
        "border": "thin lightgrey solid",
        "white-space": "pre-wrap",
    },
}
TABLE_HEADER_COLOR = "lightskyblue"
TABLE_DATA_COLOR = "rgb(239, 243, 255)"


class DashLogger(logging.StreamHandler):
    def __init__(self, stream=None):
        super().__init__(stream=stream)
        self.logs = list()

    def emit(self, record):
        try:
            msg = self.format(record)
            self.logs.append(msg)
            self.logs = self.logs[-1000:]
            self.flush()
        except Exception:
            self.handleError(record)


def create_banner(app):
    return html.Div(
        id="banner",
        className="banner",
        children=[
            html.Img(src=app.get_asset_url("logai_logo.jpg")),
            html.Plaintext("  Powered by Salesforce AI Research"),
        ],
    )


def create_description_card():
    return html.Div(
        id="description-card",
        children=[
            html.H4("AI-based Log Analysis"),
            html.Div([create_menu()]),
            html.Div(id="intro", children="  "),
        ],
    )


def create_menu():
    menu = html.Div(
        [
            dbc.Row(
                dcc.Link(
                    "Log Summarization",
                    href="/logai/pattern",
                    className="tab first",
                    style={"font-weight": "bold", "text-decoration": "underline"},
                )
            ),
            dbc.Row(
                dcc.Link(
                    "Log Clustering",
                    href="/logai/clustering",
                    className="tab third",
                    style={"font-weight": "bold", "text-decoration": "underline"},
                )
            ),
            dbc.Row(
                dcc.Link(
                    "Anomaly Detection",
                    href="/logai/anomaly",
                    className="tab second",
                    style={"font-weight": "bold", "text-decoration": "underline"},
                )
            ),
        ],
    )
    return menu


def create_modal(modal_id, header, content, content_id, button_id):
    modal = html.Div(
        [
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle(header)),
                    dbc.ModalBody(content, id=content_id),
                    dbc.ModalFooter(
                        dbc.Button(
                            "Close", id=button_id, className="ml-auto", n_clicks=0
                        )
                    ),
                ],
                id=modal_id,
                is_open=False,
            ),
        ]
    )
    return modal


def create_upload_file_layout():
    return html.Div(
        id="upload-file-layout",
        children=[
            html.Br(),
            html.P("Upload Log File"),
            dcc.Upload(
                id="upload-data",
                children=html.Div(["Drag and Drop or Select a File"]),
                style={
                    # "width": "300px",
                    "height": "50px",
                    "lineHeight": "50px",
                    "borderWidth": "1px",
                    "borderStyle": "dashed",
                    "borderRadius": "5px",
                    "textAlign": "center",
                    "margin": "10px",
                },
                multiple=True,
            ),
        ],
    )


def create_file_setting_layout():
    return html.Div(
        id="file-setting-layout",
        children=[
            html.Br(),
            html.B("File Settings"),
            html.Hr(),
            html.Label("Log Type"),
            dcc.Dropdown(
                id="log-type-select",
                options=["HDFS", "BGL", "HealthApp", "Custom"],
                value="HDFS",
            ),
            dbc.Row(dbc.Col([html.Div(id="custom-file-setting")])),
            html.Label("Log File"),
            dcc.Dropdown(id="file-select", style={"width": "100%"}),
            html.Label("Attributes"),
            dcc.Dropdown(id="attribute-name-options", multi=True),
            html.Label("Time Interval"),
            dcc.Slider(
                0,
                3,
                step=None,
                marks={0: "1s", 1: "1min", 2: "1h", 3: "1d"},
                value=0,
                id="time-interval",
            ),
            html.Hr(),
        ],
        # style={
        #     "display": "inline-block",
        #     "width": "300px",
        # }
    )


def create_param_table(params=None, height=100):
    if params is None or len(params) == 0:
        data = [{"Parameter": "", "Value": ""}]
    else:
        data = [
            {"Parameter": key, "Value": str(value["default"])}
            for key, value in params.items()
        ]

    table = dash_table.DataTable(
        data=data,
        columns=[
            {"id": "Parameter", "name": "Parameter"},
            {"id": "Value", "name": "Value"},
        ],
        editable=True,
        style_header_conditional=[{"textAlign": "center"}],
        style_cell_conditional=[{"textAlign": "center"}],
        style_table={"overflowX": "scroll", "overflowY": "scroll", "height": height},
        style_header=dict(backgroundColor=TABLE_HEADER_COLOR),
        style_data=dict(backgroundColor=TABLE_DATA_COLOR),
    )
    return table
