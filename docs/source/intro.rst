.. image:: _static/logai_logo.jpg
   :width: 650
   :align: center

Introduction
===============================================

Logs are the machine generated text messages of a computer program. In modern computer systems, logs are one of the most
critical observability data for developers to understand system behavior, monitor system health and resolve issues.
The volume of logs are huge for complex distributed systems, such as cloud, search engine, social media, etc. Log analytics,
are tools for developers to process huge volume of raw logs and generate insights, in order to better handle system
operations. While artificial intelligence (AI) and machine learning (ML) technologies are proven to be capable to improve
productivity in a lot of domains, recently more and more AI tools are integrated in log analytics solutions, in both
commercial and opensource software. However, there is still no sufficient toolkit that can handle multiple AI-based
log analysis tasks in uniform way. We introduce LogAI, an one-stop toolkit for AI-based log analytics.
LogAI provides AI and ML capabilities for log analysis. LogAI can be used for a variety of tasks such as log summarization,
log clustering and log anomaly detection. LogAI adopts the same log data model as OpenTelemetry so the developed applications
and models are eligible to logs from different log management platforms. LogAI provides a unified model interface and
integrates with popular time-series models, statistical learning models and deep learning models. LogAI also provides
an out-of-the-box GUI for users to conduct interactive analysis. With LogAI, we can also easily benchmark popular deep
learning algorithms for log anomaly detection without putting in redundant effort to process the logs. LogAI can be used
for different purposes from academic research to industrial prototyping.

LogAI Architecture
-----------------------------------------------
LogAI is separated into the GUI module and core library module.

LogAI Core Library
-----------------------------------------------

The core library module contains four main layers: data layer, pre-processing layer, information extraction layer and
analysis layer. Each layer contains the components to process logs in a standard way. LogAI applications, such as log
summarization, log clustering, unsupervised log anomaly detection, are created on top of the components of the four
layers.

.. image:: _static/LogAIDesign.png
  :width: 750

LogAI GUI Portal
-----------------------------------------------

The GUI module contains the implementation
of a GUI portal that talks to backend analysis applications. The portal is supported using `Plotly Dash
<https://github.com/plotly/dash>`_.

.. image:: _static/logai_summarization_res.png
   :width: 750
