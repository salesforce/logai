#!/bin/bash
##
## Copyright (c) 2023 Salesforce.com, inc.
## All rights reserved.
## SPDX-License-Identifier: BSD-3-Clause
## For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
##
##


# Example usages of the script:
# To run flake8 on files changed in this branch compared to the main branch, run:
# ./run_flake8.sh
# To run flake8 on files changed in this branch compared to some other branch, run:
# ./run_flake8.sh other-branch-name

# get the current git branch
CUR=`git rev-parse --abbrev-ref HEAD`
# The name of the branch that you want to compare against. This is typically the
# branch that you branched off from.
BRANCH=${1:-origin/main}
# identify the files that differ from the remote branch
FILES=`git diff-tree -r --diff-filter=d --no-commit-id --name-only $BRANCH $CUR`

# apply flake8 to python files
for f in ${FILES[@]}
do
   if [ ${f: -3} == ".py" ]
   then
   	   echo $f
       flake8 $f
   fi
done
