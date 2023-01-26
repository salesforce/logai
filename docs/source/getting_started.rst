
.. role:: file (code)
  :language: shell
  :class: highlight

.. image:: _static/logai_logo.jpg
   :width: 650
   :align: center

Getting Started
===============================================

Installation
-----------------------------------------------

You can install LogAI using :file:`pip install` with the instruction below:

.. code-block:: shell

   git clone https://git.soma.salesforce.com/SalesforceResearch/logai.git
   cd logai
   python3 -m venv venv # create virtual environment
   source venv/bin/activate # activate virtual env
   pip install ./ # install LogAI from root directory

Setup LogAI GUI Portal
-----------------------------------------------

You can also start a local LogAI service and use the GUI portal to explore LogAI.

.. code-block:: shell

   export PYTHONPATH='.'  # make sure to add current root to PYTHONPATH
   python3 gui/application.py # Run local plotly dash server.

Then open the LogAI portal via :file:`http://localhost:8050/` or :file:`http://127.0.0.1:8050/` in your browser:

.. image:: _static/logai_summarization_res.png
   :width: 750

