#!/bin/bash
##
## Copyright (c) 2023 Salesforce.com, inc.
## All rights reserved.
## SPDX-License-Identifier: BSD-3-Clause
## For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
##
##


# Change to root directory of repo
DIRNAME=$(cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd)
cd "${DIRNAME}/.."

# Get current git branch & stash unsaved changes
#GIT_BRANCH=$(git branch --show-current)
#if [ -z "${GIT_BRANCH}" ]; then
#    GIT_BRANCH="main"
#fi
#git stash

# Set up virtual environment
python3 -m pip install --upgrade pip setuptools wheel virtualenv
if [ ! -d venv ]; then
  rm -f venv
  virtualenv venv
fi
source venv/bin/activate
python3 -m pip install -r "${DIRNAME}/requirements.txt"

# Clean up build directory and install Sphinx requirements
sphinx-build -M clean "${DIRNAME}/source" "${DIRNAME}/_build"

# Build API docs for current head
export current_version="latest"
python3 -m pip install "${DIRNAME}/../[all]"
sphinx-build -b html "${DIRNAME}/source" "${DIRNAME}/_build/html/${current_version}" -W --keep-going
rm -rf "${DIRNAME}/_build/html/${current_version}/.doctrees"

# Create dummy HTML's for the stable version in the base directory
export stable_version="latest"
while read -r filename; do
    filename=$(echo "$filename" | sed "s/\.\///")
    n_sub=$(echo "$filename" | (grep -o "/" || true) | wc -l)
    prefix=""
    for (( i=0; i<n_sub; i++ )); do
        prefix+="../"
    done
    url="${prefix}${stable_version}/$filename"
    mkdir -p "${DIRNAME}/_build/html/$(dirname "$filename")"
    cat > "${DIRNAME}/_build/html/$filename" <<EOF
<!DOCTYPE html>
<html>
   <head>
      <title>LogAI Documentation</title>
      <meta http-equiv = "refresh" content="0; url='$url'" />
   </head>
   <body>
      <p>Please wait while you're redirected to our <a href="$url">documentation</a>.</p>
   </body>
</html>
EOF
done < <(cd "${DIRNAME}/_build/html/$stable_version" && find . -name "*.html")
echo "Finished writing to _build/html."
