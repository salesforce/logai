# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Salesforce.com, inc.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
#
#


# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open("README.md") as f:
    readme = f.read()

with open("LICENSE.txt") as f:
    license = f.read()

extras_require = {
    "gui": [
        "dash-bootstrap-components>=1.2.1",
        "plotly>=5.9.0",
        "dash>=2.5.1",
    ],
    "deep-learning": [
        "tokenizers>=0.11.6",
        "datasets>=1.18.3",
        "torch>=1.10.1",
        "transformers>=4.17.0,<=4.23",
    ],
    "dev": [
        "Sphinx>=3.5.3",
        "docutils>=0.18.1",
        "wheel>=0.37.1",
        "toml>=0.10.2",
        "build>=0.7.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.16",
        "pytest>=6.2.5",
    ]
}
extras_require["all"] = sum(extras_require.values(), [])

setup(
    name="logai",
    version="0.1.5",
    description="LogAI is unified framework for AI-based log analytics",
    long_description_content_type="text/markdown",
    long_description=readme,
    author="Qian Cheng, Amrita Saha, Wenzhuo Yang, Chenghao Liu, Gerald Woo, Doyen Sahoo, Steven Hoi",
    author_email="logai@salesforce.com",
    python_requires=">=3.7.0,<4",
    install_requires=[
        "schema>=0.7.5",
        "salesforce-merlion>=1.0.0",
        "Cython>=0.29.30",
        "nltk>=3.6.5",
        "gensim>=4.1.2",
        "scikit-learn>=1.0.1",
        "pandas>=1.2.0",
        "numpy>=1.21.4",
        "spacy>=3.2.2",
        "attrs>=21.2.0",
        "dataclasses>=0.6",
        "PyYAML>=6.0",
        "tqdm>=4.62.3",
        "cachetools>=4.2.4",
        "matplotlib>=3.5.1",
        "seaborn>=0.11.2",
        "Jinja2>=3.0.0",
        "numba>=0.56.3",
    ],
    extras_require=extras_require,
    license=license,
    packages=find_packages(exclude=["tests", "tests.*", "docs", "gui", "gui.*"]),
    include_package_data=True,
)
