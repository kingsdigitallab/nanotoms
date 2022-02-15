# Narrative Atoms

The Narrative Atoms project contains a set of tools for topic extraction and text generation,
using as a test case data about objects from the [Middling Culture](https://middlingculture.com/) project.

This repository provides a command line tool and a web interface to process and interact with the data.

## Workflow

```mermaid
flowchart LR
    data_raw[/Raw data/] --> prepare(Clean/prepare)
    data_raw -.- comment_data_raw[CSV data curated and\n provided by the research team\n with information about objects\n from XVI-XVII period]
    class comment_data_raw comment

    prepare --> data_prepared[/Prepared data/]
    prepare --> data_scraped[/Scraped/extracted data/]
    prepare -.- comment_prepare[Clean the data,\n validate/parse URLs,\n scrape URLs]
    class comment_prepare comment

    data_prepared -.- comment_data_prepared[CSV with cleaned data]
    data_prepared --> transform(Transform)
    class comment_data_prepared comment

    data_scraped --> transform(Transform)
    data_scraped -.- comment_data_scrapped[CSV with data scraped from the URLs]
    class comment_data_scrapped comment

    transform -.- comment_transform[Merge the scraped and prepared data, \nadd features to the prepared data,\n apply named entity recognition]
    transform --> data_transformed[/Transformed data/]
    transform --> data_train_gpt[/GPT training data/]  
    class comment_transform comment

    data_transformed -.- comment_data_transformed[CSV with transformed data, including\n the scrapped data and extracted entities]
    data_transformed --> train(Train/tune topic model)
    class comment_data_transformed comment

    train --> data_final[/Trained data/]
    train --> topic_model[/Trained topic model/]
    train -.- comment_train[Train topic model and extract topics from the data]
    data_final -.- comment_data_final[CSV with final data, including topics]
    topic_model -.- comment_topic_model[Topic model for topic extraction]
    class comment_train comment
    class comment_data_final comment
    class comment_topic_model comment

    data_train_gpt --> gpt(Train GPT model)
    data_train_gpt -.- comment_data_train_gpt[Text file containing all the objects descriptions]
    gpt -.- comment_gpt[Notebook to train GPT model for text generation]
    class comment_data_train_gpt comment
    class comment_gpt comment
    
    gpt --> gpt_model[/Trained GPT model/]
    gpt_model -.- comment_gpt_model[GPT model for text generation]
    class comment_gpt_model comment

    click gpt "https://colab.research.google.com/drive/1CHByFGc2LKSPaW6X6_MqXqJRmWnlesjA" "Train GPT model"
    click gpt_model "https://github.com/kingsdigitallab/gpt-neo-125M-nanotoms" "GPT model"

    classDef comment fill:lightyellow,stroke-width:0px;
```

## Set up

Install [poetry](https://python-poetry.org/docs/#installation) and the requirements:

    poetry install

## Run the cli

    poetry run python cli.py

## Run the gui

    poetry run streamlit run streamlit_app.py

## Development

    poetry install --dev
    poetry shell
