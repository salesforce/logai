#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, callback

from gui.pages.utils import create_banner
from gui.pages import pattern as pattern_page
from gui.pages import anomaly_detection as anomaly_page
from gui.pages import clustering as clustering_page
from gui.callbacks import pattern, anomaly_detection, clustering, utils


app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    title="LogAI",
)
server = app.server
app.config["suppress_callback_exceptions"] = True

app.layout = dbc.Container(
    [
        dcc.Location(id="url", refresh=False),
        dbc.Container(id="page-content", fluid=True),
    ],
    fluid=True,
)


@callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/logai/pattern":
        return dbc.Container(
            [dbc.Row(create_banner(app)), pattern_page.layout], fluid=True
        )
    elif pathname == "/logai/anomaly":
        return dbc.Container(
            [dbc.Row(create_banner(app)), anomaly_page.layout], fluid=True
        )
    elif pathname == "/logai/clustering":
        return dbc.Container(
            [dbc.Row(dbc.Col(create_banner(app))), clustering_page.layout], fluid=True
        )
    else:
        return dbc.Container(
            [dbc.Row(dbc.Col(create_banner(app))), pattern_page.layout], fluid=True
        )


if __name__ == "__main__":
    app.run_server(debug=False)
