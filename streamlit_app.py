from typing import Any

import numpy as np
import pandas as pd
import pyLDAvis
import pyLDAvis.gensim_models
import spacy
import streamlit as st
from gensim.models import LdaModel
from streamlit import components
from transformers import GPT2Tokenizer, GPTNeoForCausalLM
from wordcloud import WordCloud

import cli
import settings
from nanotoms import data as dm
from nanotoms import generate as gm
from nanotoms import train as tm


def streamlit(datadir: str = settings.DATA_DIR.name):
    st.set_page_config(page_title=settings.PROJECT_TITLE, layout="wide")
    st.title(settings.PROJECT_TITLE)

    with st.sidebar:
        sidebar(datadir)

    if show_topics_view():
        with st.container():
            data_section(datadir)

        with st.container():
            trained_section(datadir)

    if show_explore_view():
        with st.container():
            explore_section()

    if show_generation_view():
        with st.container():
            generator_section(datadir)


def show_topics_view():
    return st.session_state.view == "Topic extraction"


def show_explore_view():
    return st.session_state.view == "Explore topic data"


def show_generation_view():
    return st.session_state.view == "Text generation"


def sidebar(datadir: str):
    st.title("Configuration")

    st.session_state.view = st.radio(
        "Choose view", ("Topic extraction", "Explore topic data", "Text generation")
    )

    if show_topics_view():
        topics_sidebar(datadir)

    if show_explore_view():
        explore_sidebar(datadir)

    if show_generation_view():
        generation_sidebar()


def topics_sidebar(datadir: str):
    st.header("Topics")

    with st.expander("Data", expanded=False):
        uploaded_file = st.file_uploader("Upload new data", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            df.to_csv(dm.get_raw_data_path(datadir), index=False)

    if dm.get_raw_data(datadir) is not None:
        with st.expander("1. Clean/prepare", expanded=False):
            st.write(get_help(cli.prepare))
            with st.form("prepare_form"):
                scrape = st.checkbox("Scrape URLs in the data?")

                if st.form_submit_button("Prepare data"):
                    with st.spinner("Preparing data"):
                        cli.prepare(datadir, scrape=scrape)

    if dm.get_clean_data(datadir) is not None:
        with st.expander("2. Transform", expanded=False):
            st.write(get_help(cli.transform))
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
                        help="Words to be excluded from the processing.",
                    )

                    labels = st.multiselect(
                        "Select entity labels",
                        options=spacy.load(language_model).get_pipe("ner").labels,
                        default=settings.SPACY_ENTITY_TYPES,
                        help="Entitly labels to extract.",
                    )

                    if st.form_submit_button("Transform data"):
                        with st.spinner("Transforming data"):
                            cli.transform(
                                datadir, language_model, stop_words.split(", "), labels
                            )

    if dm.get_transformed_data(datadir) is not None:
        with st.expander("3. Train", expanded=False):
            st.write(get_help(cli.train))
            with st.form("train_form"):
                number_of_topics = st.number_input(
                    "Number of topics",
                    min_value=2,
                    max_value=20,
                    value=settings.NUMBER_OF_TOPICS,
                    step=1,
                    help="Number of topics to extract from the data.",
                )
                passes = st.number_input(
                    "Passes",
                    min_value=1,
                    max_value=50,
                    value=settings.NUMBER_OF_PASSES,
                    step=1,
                    help="Number of passes through the data during training.",
                )
                minimum_probability = st.number_input(
                    "Minimum probability",
                    min_value=0.01,
                    max_value=1.0,
                    step=0.01,
                    value=settings.TOPICS_MINIMUM_PROBABILITY,
                    help=(
                        "Topics with a probability lower than this threshold "
                        " will be excluded."
                    ),
                )
                multicore = st.checkbox(
                    "Multiprocessing",
                    value=True,
                    help=(
                        "Use multiprocessing to speed up training process. "
                        "Turn off in streamlit cloud or if having issues training."
                    ),
                )

                if st.form_submit_button("Train"):
                    with st.spinner("Training"):
                        cli.train(
                            datadir,
                            number_of_topics,
                            passes,
                            minimum_probability,
                            multicore,
                            True,
                        )

        with st.expander("3.1 Tune", expanded=False):
            st.write(get_help(cli.tune))
            with st.form("tune_form"):
                min_number_of_topics = st.number_input(
                    "Minimum number of topics",
                    min_value=2,
                    max_value=20,
                    value=settings.MIN_NUMBER_OF_TOPICS,
                    step=1,
                    help="Minimum number of topics to extract from the data.",
                )
                max_number_of_topics = st.number_input(
                    "Maximum number of topics",
                    min_value=2,
                    max_value=20,
                    value=settings.MAX_NUMBER_OF_TOPICS,
                    step=1,
                    help="Maximum number of topics to extract from the data.",
                )
                passes = st.number_input(
                    "Passes",
                    min_value=1,
                    max_value=50,
                    value=settings.NUMBER_OF_PASSES,
                    step=1,
                    help="Number of passes through the data during training.",
                )
                minimum_probability = st.number_input(
                    "Minimum probability",
                    min_value=0.01,
                    max_value=1.0,
                    step=0.01,
                    value=settings.TOPICS_MINIMUM_PROBABILITY,
                    help=(
                        "Topics with a probability lower than this threshold "
                        " will be excluded."
                    ),
                )
                multicore = st.checkbox(
                    "Multiprocessing",
                    value=True,
                    help=(
                        "Use multiprocessing to speed up training process. "
                        "Turn off in streamlit cloud or if having issues training."
                    ),
                )

                if st.form_submit_button("Tune"):
                    with st.spinner("Training"):
                        cli.tune(
                            datadir,
                            min_number_of_topics,
                            max_number_of_topics,
                            passes,
                            minimum_probability,
                            multicore,
                            True,
                        )


def get_help(f) -> str:
    return f.__doc__.split(":param")[0].strip()


def explore_sidebar(datadir: str):
    final_data = dm.list_final_data(datadir)
    if not final_data:
        return

    st.header("Explore topic data")

    with st.expander("Data", expanded=True):
        model_name = st.selectbox("Select data", final_data)
        model_name = "_".join(model_name.split(", ")[0].split("_")[1:])

        data = dm.get_final_data(datadir, model_name)
        if data is not None:
            st.session_state.topic_data = data

        model = dm.get_model(datadir, model_name)
        if model:
            st.session_state.topic_model = model


def generation_sidebar():
    st.header("Generation")
    with st.expander("Generator settings", expanded=True):
        length = st.slider(
            "Number of tokens",
            min_value=10,
            max_value=1000,
            step=10,
            value=100,
            help="Maximum number of generated tokens",
        )
        no_repeat_ngram_size = st.slider(
            "No repeat ngrams",
            min_value=0,
            max_value=5,
            value=2,
            help=(
                "N-gram size that can't occur more than once. "
                "This helps avoid repetition in the generated text."
            ),
        )

        temperature = st.slider(
            "Temperature",
            min_value=0.1,
            max_value=1.0,
            step=0.1,
            value=0.7,
            help=(
                "How sensitive the algorithm is to selecting least common options "
                "for the generated text."
            ),
        )
        top_k = st.slider(
            "Top k",
            min_value=1,
            max_value=100,
            step=1,
            value=50,
            help="How many potential outcomes are considered before generating the text",
        )
        early_stopping = st.checkbox(
            "Stop at last full sentence (if possible)?", value=True
        )

        st.session_state.gen_settings = dict(
            do_sample=True,
            early_stopping=early_stopping,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_length=length,
            temperature=temperature,
            top_k=top_k,
        )


def data_section(datadir: str):
    st.header("Data")

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


def show_data(data: pd.DataFrame):
    st.dataframe(data)


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


def explore_section():
    st.header("Explore topic data")

    data = st.session_state.topic_data
    if data is not None:
        show_data(data)

    model = st.session_state.topic_model

    with st.container():
        explore_by_topic(data, model)

    with st.container():
        explore_by_entities(data)


def explore_by_topic(data: pd.DataFrame, model: LdaModel):
    st.header("Data by topic")

    number_of_topics = tm.get_number_of_topics(model)
    topics = [
        ", ".join([term for term, _ in model.show_topic(idx, 15)])
        for idx in range(number_of_topics)
    ]
    topic = st.selectbox(
        "Select topic", range(len(topics)), format_func=lambda x: f"{x}: {topics[x]}"
    )

    # topic = st.slider(
    # "Topic", min_value=0, max_value=number_of_topics - 1, step=1, value=0
    # )

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

    sort_by = st.selectbox(
        "Sort objects by",
        [f"topic:{topic}", "title", "index"],
    )

    objects = data[~data[f"topic:{topic}"].isin([np.nan])]
    if sort_by != "index":
        objects = objects.sort_values([sort_by], ascending=(sort_by == "title"))

    show_objects(objects)


def show_objects(objects: pd.DataFrame):
    st.subheader("Objects")
    for idx, row in enumerate(objects.iterrows()):
        obj = row[1]
        with st.expander(f"{idx + 1}. {obj['title']}", expanded=False):
            st.write(obj["description"])

            for url in obj["url"]:
                st.markdown(f"- [{url}]({url})", unsafe_allow_html=True)


def explore_by_entities(data: pd.DataFrame):
    st.header("Data by entities")

    entities = np.concatenate(data["entities"].values.tolist()).flat
    entities = sorted(list(set(entities)))

    selected = st.multiselect("Select entities", options=entities)
    if selected:
        contains_entities = set(selected).issubset

        objects = data[data["entities"].map(contains_entities)]
        show_objects(objects)


def generator_section(datadir: str):
    st.header("Text generation")

    if "prompt" not in st.session_state:
        st.session_state.prompt = ""

    with st.container():
        prompt = st.text_area(
            "Prompt",
            placeholder="Prompt to generate text",
        )

        col1, col2, col3 = st.columns([0.2, 0.125, 1])

        with col1:
            if st.button("Generate text"):
                if prompt:
                    text = generate_text(prompt)
                    st.session_state.prompt = text
                else:
                    st.warning("Please fill in the prompt field")

        with col2:
            if st.session_state.prompt and st.button("More"):
                text = generate_text(st.session_state.prompt)
                st.session_state.prompt = text

        with col3:
            if st.session_state.prompt and st.button("Clear"):
                st.session_state.prompt = ""

        if st.session_state.prompt:
            st.write(st.session_state.prompt)


def get_description(data: pd.DataFrame, title: str) -> str:
    return data[data["title"] == title].iloc[0]["description"].split(".")[-1].strip()


def generate_text(prompt: str) -> str:
    try:
        return gm.generate(
            get_generator_model(),
            get_generator_tokenizer(),
            prompt,
            get_generator_settings(),
        )
    except Exception as e:
        st.error(f"Error generating text: {e}")
        return "Error"


def get_generator_model(
    path: str = settings.TEXT_GENERATOR_MODEL_PATH,
) -> GPTNeoForCausalLM:
    if "gen_model" not in st.session_state:
        st.session_state.gen_model = gm.get_model(path)
        st.session_state.gen_tokenizer = gm.get_tokenizer(path)

    return st.session_state.gen_model


def get_generator_tokenizer(
    path: str = settings.TEXT_GENERATOR_MODEL_PATH,
) -> GPT2Tokenizer:
    if "gen_tokenizer" not in st.session_state:
        st.session_state.gen_tokenizer = gm.get_tokenizer(path)

    return st.session_state.gen_tokenizer


def get_generator_settings() -> dict[str, Any]:
    return st.session_state.gen_settings


if __name__ == "__main__":
    streamlit()
