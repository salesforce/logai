#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#
import os.path

from gui.file_manager import FileManager

TEST_FILE_DIR = os.path.join(os.path.dirname(__file__), "data/HealthApp_format_2000.csv")

class TestFileManager:
    def setup(self):
        self.file_manager = FileManager()

        return

    def test_base_directory(self):

        print(os.path.exists(self.file_manager.base_directory))
        print(self.file_manager.base_directory)
        pass

