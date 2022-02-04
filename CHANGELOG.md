# Changelog

All notable changes to this project will be documented in this file.

The format is based on `Keep a Changelog`_, and this project adheres to
`Semantic Versioning`_.

.. \_Keep a Changelog: https://keepachangelog.com/
.. \_Semantic Versioning: https://semver.org/spec/v2.0.0.html

## [Unreleased] - yyyy-mm-dd

### Added

- [Streamlit](streamlit.io/) app.
- Module for text generation using a [GPT-Neo](https://zenodo.org/record/5297715) model.
- [Jupyter notebook](https://colab.research.google.com/drive/1CHByFGc2LKSPaW6X6_MqXqJRmWnlesjA) to train the text generation model.
- Module for semantic text search.
- Function to extract inventories and wills from data and convert them to text to be
  used to train the generator model.
- API for text generation and semantic search.

## [0.1.0] - 2022-01-10

### Added

- Module to download, generate and process data.
- Module to transform the data, add new features and perform named entity recognition.
- Module to train and tune a topic model and assign topics to the data.
- Command line application.