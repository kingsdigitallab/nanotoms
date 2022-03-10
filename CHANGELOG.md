# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - yyyy-mm-dd

## Added

- Word embeddings to version control.

## [0.2.1] - 2022-03-10

### Fixed

- Call to topic extraction tune command from the streamlit app.

## [0.2.0] - 2022-03-10

### Added

- [Streamlit](streamlit.io/) app.
- Module for text generation using a [GPT-Neo](https://zenodo.org/record/5297715) model.
- [Jupyter notebook](https://colab.research.google.com/drive/1CHByFGc2LKSPaW6X6_MqXqJRmWnlesjA) to train the text generation model.
- Module for semantic text search.
- Function to extract inventories and wills from data and convert them to text to be
  used to train the generator model.
- API for text generation and semantic search.
- [Dockerfile](https://docs.docker.com/) for API deployment.
- Scripts to start the API, Docker and Streamlit apps.
- [Mermaid](https://mermaid-js.github.io/) flowchart to the [README](README.md).
- [bumpversion](https://github.com/c4urself/bump2version) configuration.

## [0.1.0] - 2022-01-10

### Added

- Module to download, generate and process data.
- Module to transform the data, add new features and perform named entity recognition.
- Module to train and tune a topic model and assign topics to the data.
- Command line application.
