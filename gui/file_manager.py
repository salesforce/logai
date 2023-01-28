#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os
import base64
from dash import html
from urllib.parse import quote as urlquote


class FileManager:
    def __init__(self, directory=None):
        if directory is None:
            self.directory = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "uploaded_files"
            )
        else:
            self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

    def save_file(self, name, content):
        data = content.encode("utf8").split(b";base64,")[1]
        with open(os.path.join(self.directory, name), "wb") as fp:
            fp.write(base64.decodebytes(data))

    def uploaded_files(self):
        files = []
        for filename in os.listdir(self.directory):
            path = os.path.join(self.directory, filename)
            if os.path.isfile(path):
                files.append(filename)
        return files

    def file_download_link(self, filename):
        location = "/download/{}".format(urlquote(filename))
        return html.A(filename, href=location)

    @property
    def base_directory(self):
        return self.directory
