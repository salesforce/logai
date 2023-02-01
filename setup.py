# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#


# Learn more: https://github.com/kennethreitz/setup.py
import glob

from setuptools import setup, find_packages

with open("requirements.txt") as f:
    REQUIRED = f.read().splitlines()

with open("README.md") as f:
    readme = f.read()

with open("LICENSE.txt") as f:
    license = f.read()

setup(
    name="logai",
    version="v0.1.2",
    description="LogAI is unified framework for AI-based log analytics",
    long_description_content_type="text/markdown",
    long_description=readme,
    author="Qian Cheng, Amrita Saha, Wenzhuo Yang, Chenghao Liu, Gerald Woo, Doyen Sahoo, Steven Hoi",
    author_email="logai@salesforce.com",
    python_requires=">=3.8.0",
    install_requires=REQUIRED,
    license=license,
    packages=find_packages(exclude=["tests", "tests.*", "docs", "gui", "gui.*"]),
    data_files=glob.glob('logai/dataloader/openset_configs/*.json'),
)
