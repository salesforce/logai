<!--
Copyright (c) 2022 Salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

-->
# AI Library Template

A template repository to create an AI library. It contains common functions based on the [standards and best practices](https://salesforce.quip.com/LVbDAhptBEaX) for creating a new project, including dependency adding, code style formatting, tests, documentation, and license.

## How To Use the Template

1. Navigate to the main page of the repository.
2. Above the file list, click 'Use this template'.
3. Follow the instructions to finish your repository creation. (Checkout [here](https://docs.github.com/en/enterprise-server@2.22/github/creating-cloning-and-archiving-repositories/creating-a-repository-from-a-template#creating-a-repository-from-a-template) for more details. )
4. After created your repository, run ```pip install -r requirements.txt``` to install dependencies so that all the functions from the template can work properly. 


## Add Dependencies to Requirements.txt
Add dependencies to requirements.txt when the library depends on a new library. First, you need to fill the [3PP](https://confluence.internal.salesforce.com/pages/viewpage.action?spaceKey=SECURITY&title=Third+Party+%283PP%29+and+Open+Source+Usage+Program). (Check out an example of the 3PP filling GUS ticket [here](https://gus.lightning.force.com/lightning/r/ADM_Third_Party_Software__c/a0qB0000000KpWmIAK/view), and how Pennyworth keeps track of all the 3PP [here](https://salesforce.quip.com/n5AcABHW7zbi#QYTACAO2dGR)). And then, you need to add the specific version to the requirements.txt. For example, if you are using scikit-learn with version 0.23.1, then append it to requirements.txt
```
scikit-learn==0.23.1
```

You can install the dependencies by running:
```
pip install -r requirements.txt
```

## Code Style Formatting

To format your Python code, use [_Black_](https://black.readthedocs.io/en/stable/index.html)
by using the commands below in your terminal. _Black_ automatically formats your code using a consistent style.
```
# Run Black on a particular file
black your_filename.py

# Run Black on all the files modified in the current branch compared to the main branch
./run_black.sh

# Run Black on the files modified in the current branch compared to another branch
# You can also use this for remote branches like origin/parallel-hpo.
./run_black.sh other-branch-name
```

## PEP 8 Style Enforcement
To check that your code follows [PEP 8](https://www.python.org/dev/peps/pep-0008/), use 
[Flake8](https://flake8.pycqa.org/en/latest/) by using the commands below in your terminal.
Flake8 prints out style guide violations in each file. Please correct these violations before you submit your
pull request. The configuration for Flake8 to make it compatible with _Black_ is in `setup.cfg` in this directory.
You don't need to do any extra setup to make Flake8 use `setup.cfg`.

```
# Run Flake8 on a particular file
black your_filename.py

# Run Flake8 on all the files modified in the current branch compared to the main branch
./run_flake8.sh

# Run Flake8 on the files modified in the current branch compared to another branch
# You can also use this for remote branches like origin/parallel-hpo.
./run_flake8.sh other-branch-name
```

## Tests

Please add unit tests under the `tests` folder as you develop your project. To allow your source code successfully imported while testing, you need to create an empty `__init__.py`  under every source code folder, so that these folders will be considered as python packages and thus can be imported. Similarly, please also include `__init__.py` in the tests folder so that it can be imported externally. 

You can run following command to run all of the tests:

```
./run_unittests.sh
```

## Auto Documentation by Sphinx:

Please use [Sphinx](https://www.sphinx-doc.org/en/master/) to document your library and put documents in `docs` folder. 

### Generate Documentation From Source Files
The `docs` folder provides example source files to generate project documentation. To generate it, follow the steps below:
```
# Go to the docs folder
cd docs

# Remove previously generated documentation
make clean

# Generate html documentation.
make html
```
The generated documentation files will be in `./docs/build/html`. You can view the
documentation by opening `./docs/build/html/index.html` in your favorite browser.

### Generate New Source Files

If you are adding new files to the repo, you can use [sphinx-apidoc](https://www.sphinx-doc.org/en/master/man/sphinx-apidoc.html)
to automatically generate documentation from docstrings using the command below.

```
cd docs
sphinx-apidoc -o ./source ../<folder_included_new_files>
```
For example, in this project, you can run:
```
cd docs
sphinx-apidoc -o ./source ../helloworld
```
to generate all the description docs for the files in helloworld folder.


You can modify the `.rst` files to add more information about your code such as information on 
why someone would want to use your code and example usages of your code. After you are 
done, add your new `.rst` file to the `toctree` in `./docs/source/index.rst` to 
add it to the navigation sidebar on the documentation website.

Here's an example of how you would add a new file `./docs/source/new_file.rst`
to the `toctree` in `index.rst`. An excerpt of `index.rst` where the `toctree` is 
configured is below. To add `new_file.rst`to the navigation bar, simply add `new_file`
to the list of pages with the same indentation and without the `.rst` extension. The
order of the pages in the navigation bar is the same as the order of the pages in 
the `toctree` in `index.rst`.
```
.. toctree::
   :maxdepth: 2
   :caption: User Guide

   helloworld
   new_file
```

## Update License

Please add the correct license of your project to your repository following [here](https://docs.github.com/en/github/building-a-strong-community/adding-a-license-to-a-repository).

## Helpful Resources to Add to the Documentation:
- [Getting started with Sphinx tutorial](https://sphinx-rtd-tutorial.readthedocs.io/en/latest/index.html)
- [reStructured text and Sphinx cheatsheet](https://thomas-cokelaer.info/tutorials/sphinx/rest_syntax.html) -
useful reference on how to format documentation
