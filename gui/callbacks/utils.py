#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import dash_bootstrap_components as dbc

from dash import html, Input, Output, State, callback, dash_table
from gui.file_manager import FileManager


file_manager = FileManager()


@callback(
    Output("upload-status", "children"),
    [Input("upload-data", "filename"), Input("upload-data", "contents")],
)
def upload_file(uploaded_filenames, uploaded_file_contents):
    if uploaded_filenames is not None and uploaded_file_contents is not None:
        for name, data in zip(uploaded_filenames, uploaded_file_contents):
            file_manager.save_file(name, data)
        return html.Div("Upload Success!")
    else:
        return html.Div("File Already Exists!")


@callback(
    Output("file-select", "options"),
    Output("file-select", "value"),
    [Input("log-type-select", "value")],
)
def select_file(dataset_name):
    options = []
    files = file_manager.uploaded_files()
    if dataset_name.lower == "custom":
        for filename in files:
            options.append({"label": filename, "value": filename})
    else:
        for filename in files:
            if dataset_name.lower() in filename.lower():
                options.append({"label": filename, "value": filename})

    if len(options) > 0:
        return options, options[0]["label"]
    else:
        return options, ""


@callback(
    Output("custom-file-setting", "children"),
    [Input("log-type-select", "value")],
)
def custom_file_setting(dataset_name):
    if dataset_name.lower() == "custom":
        return html.Div(
            [
                dbc.Textarea(
                    id="custom-file-config",
                    size="lg",
                    className="mb-5",
                    placeholder="custom file loader config",
                )
            ]
        )
    else:
        return html.Div()
