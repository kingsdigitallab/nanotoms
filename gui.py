import ast

import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models
import spacy
import streamlit as st
from gensim.models import LdaModel
from streamlit import components
from wordcloud import WordCloud

import cli
import settings
from nanotoms import data as dm
from nanotoms import train as tm


def streamlit(datadir: str = settings.DATA_DIR.name):
    st.set_page_config(page_title=settings.PROJECT_TITLE, layout="wide")
    st.title(settings.PROJECT_TITLE)

    with st.sidebar:
        sidebar(datadir)

    st.header("Data")
    with st.container():
        data_section(datadir)

    with st.container():
        trained_section(datadir)


def sidebar(datadir: str):
    st.title("Configuration")

    with st.expander("Data", expanded=False):
        uploaded_file = st.file_uploader("Upload new data", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df.to_csv(dm.get_raw_data_path(datadir), index=False)

    if dm.get_raw_data(datadir) is not None:
        with st.expander("1. Clean/prepare", expanded=False):
            with st.form("prepare_form"):
                scrape = st.checkbox("Scrape URLs in the data?")

                if st.form_submit_button("Prepare data"):
                    with st.spinner("Preparing data"):
                        cli.prepare(datadir, scrape=scrape)

    if dm.get_clean_data(datadir) is not None:
        with st.expander("2. Transform", expanded=False):
            language_models = spacy.util.get_installed_models()
            if not language_models:
                st.warning("No language models available")
                if st.button("Download language model"):
                    with st.spinner("Downloading language model"):
                        cli.spacy()
                    st.info("Language model downloaded, please restart the app")

            if language_models:
                with st.form("transform_form"):
                    language_model = st.selectbox(
                        "Select language model", language_models
                    )

                    stop_words = st.text_area(
                        "Additional stop words",
                        value=", ".join(settings.SPACY_EXTRA_STOP_WORDS),
                    )

                    labels = st.multiselect(
                        "Select entity labels",
                        options=spacy.load(language_model).get_pipe("ner").labels,
                        default=settings.SPACY_ENTITY_TYPES,
                    )

                    if st.form_submit_button("Transform data"):
                        with st.spinner("Transforming data"):
                            cli.transform(
                                datadir, language_model, stop_words.split(", "), labels
                            )

    if dm.get_transformed_data(datadir) is not None:
        with st.expander("3. Train", expanded=False):
            with st.form("train_form"):
                number_of_topics = st.number_input(
                    "Number of topics",
                    min_value=2,
                    max_value=20,
                    value=settings.NUMBER_OF_TOPICS,
                    step=1,
                )
                passes = st.number_input(
                    "Passes",
                    min_value=1,
                    max_value=50,
                    value=settings.NUMBER_OF_PASSES,
                    step=1,
                )
                minimum_probability = st.number_input(
                    "Minimum probability",
                    min_value=0.01,
                    max_value=1.0,
                    step=0.01,
                    value=settings.TOPICS_MINIMUM_PROBABILITY,
                )

                if st.form_submit_button("Train"):
                    with st.spinner("Training"):
                        cli.train(
                            datadir, number_of_topics, passes, minimum_probability
                        )


def data_section(datadir: str):
    data = dm.get_raw_data(datadir)
    if data is not None:
        with st.expander(
            f"View raw data, {dm.lastModified(dm.get_raw_data_path(datadir))}",
            expanded=False,
        ):
            show_data(data)

    data = dm.get_clean_data(datadir)
    if data is not None:
        modified = dm.lastModified(dm.get_clean_data_path(datadir))

        with st.expander(
            f"View cleaned/prepared data, {modified}",
            expanded=False,
        ):
            show_data(data)

    data = dm.get_transformed_data(datadir)
    if data is not None:
        modified = dm.lastModified(dm.get_transformed_data_path(datadir))

        with st.expander(f"View transformed data, {modified}", expanded=False):
            show_data(data)


def trained_section(datadir: str):
    final_data = dm.list_final_data(datadir)
    if not final_data:
        return

    st.header("Trained data/model")

    model_name = st.selectbox("Select trained data", final_data)
    model_name = "_".join(model_name.split(", ")[0].split("_")[1:])

    with st.expander("View data", expanded=False):
        data = dm.get_final_data(datadir, model_name)
        if data is not None:
            show_data(data)

    model = dm.get_model(datadir, model_name)
    if not model:
        return

    pyldavis(datadir, model, model_name)
    topic_terms_chart(model)

    with st.container():
        explore_by_topic(data, model)

    with st.container():
        explore_by_entities(data)


def pyldavis(datadir: str, model: LdaModel, model_name: str):
    with st.expander("View model"):
        try:
            model_data = pyLDAvis.gensim_models.prepare(
                model,
                dm.get_bow_corpus(datadir),
                dm.get_model_id2word(datadir, model_name),
                n_jobs=2,
                sort_topics=False,
            )
            html = pyLDAvis.prepared_data_to_html(model_data)
            components.v1.html(html, height=800)
        except Exception as e:
            st.error(e)


def topic_terms_chart(model: LdaModel):
    number_of_topics = tm.get_number_of_topics(model)

    with st.expander("View top words in topics"):
        number_of_words = st.slider(
            "Number of words",
            min_value=5,
            max_value=50,
            step=1,
            value=10,
        )
        topics = model.show_topics(
            number_of_topics, formatted=False, num_words=number_of_words
        )

        topics_data = []
        for idx, topic in topics:
            topics_data.extend(
                [{"term": item[0], f"topic {idx:02n}": item[1]} for item in topic]
            )

        df = pd.DataFrame(topics_data).set_index("term")
        st.bar_chart(df)


def explore_by_topic(data: pd.DataFrame, model: LdaModel):
    st.header("Explore data by topic")

    number_of_topics = tm.get_number_of_topics(model)

    topic = st.slider(
        "Topic", min_value=0, max_value=number_of_topics - 1, step=1, value=0
    )

    with st.expander("View terms in topic", expanded=False):
        frequencies = {}
        for term, weight in model.show_topic(topic, 50):
            frequencies[term] = int(weight * 1000)

        wc = WordCloud(
            background_color="white",
            width=1000,
            height=150,
            min_font_size=12,
        )
        wc.generate_from_frequencies(frequencies)
        st.image(
            wc.to_image(),
            caption="Terms in topic",
            use_column_width="always",
        )

    st.subheader("Objects")
    sort_by = st.selectbox(
        "Sort objects by",
        [f"topic:{topic}", "title", "index"],
    )

    objects = data[~data[f"topic:{topic}"].isin([np.nan])]
    if sort_by != "index":
        objects = objects.sort_values([sort_by], ascending=(sort_by == "title"))

    for _, row in objects.iterrows():
        with st.expander(row["title"], expanded=False):
            st.write(row["description"])


def explore_by_entities(data: pd.DataFrame):
    st.header("Explore data by entities")

    entities = []
    entities_df = data.filter(regex="^entity", axis=1)
    for label, column in entities_df.iteritems():
        label = label.split(":")[1]
        for value in column.dropna().values:
            entities.extend([f"{label}: {v}" for v in ast.literal_eval(value)])

    entities = sorted(list(set(entities)))

    st.multiselect("Select entities", options=entities)


def show_data(data: pd.DataFrame):
    st.dataframe(data)


if __name__ == "__main__":
    streamlit()
