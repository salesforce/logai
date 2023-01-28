<!--
Copyright (c) 2023 Salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

-->
To generate documentation using Sphinx, just run the script build_docs.sh. 
The build/html directory will be populated with searchable, indexed HTML documentation.

Note that our documentation also depends on Pandoc to render Jupyter notebooks. 
For Ubuntu, call `sudo apt-get install pandoc`. For Mac OS, install Homebrew and call `brew install pandoc`.