.. role:: file (code)
  :language: shell
  :class: highlight


Developers' Guide
============================

This guide if for people who want to contribute to LogAI codebase.
The guide includes how to run and test LogAI in local environment.

Install dependencies
----------------------------

.. code-block:: shell

    git clone https://github.com/salesforce/logai.git
    cd logai
    python3 -m venv venv # create virtual environment
    source venv/bin/activate # activate virtual env
    pip install -r requirement.txt


Build wheels package
----------------------------

.. code-block:: shell

    python setup.py bdist_wheel

Then you can find the .whl package in :file:`./dist/`.

Install Log-AI from wheels
----------------------------

.. code-block:: shell

    pip install logai-{version}-py2.py3-none-any.whl


Use GUI to explore LogAI
----------------------------

.. code-block:: shell

    export PYTHONPATH='.'  # make sure to add current root to PYTHONPATH
    python3 gui/application.py # Run local plotly dash server.

Then open the LogAI portal via :file:`http://localhost:8050/` or :file:`http://127.0.0.1:8050/` in your browser:

.. image:: _static/log_summarization.png
   :width: 750


