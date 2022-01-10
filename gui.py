import pandas as pd
import spacy
import streamlit as st

import cli
import settings
from nanotoms import data as dm


def streamlit(datadir: str = settings.DATA_DIR.name):
    st.set_page_config(page_title=settings.PROJECT_TITLE, layout="wide")
    st.title(settings.PROJECT_TITLE)

    with st.sidebar:
        sidebar(datadir)

    st.header("Data")
    with st.container():
        prepare_section(datadir)

    with st.container():
        transform_section(datadir)

    with st.container():
        train_section(datadir)

    # data = dm.get_raw_data(datadir)
    # if data_selectbox != "raw":
    # data = dm.get_final_data(datadir, data_selectbox)

    # st.header(f"{data_selectbox.title()} data")
    # show_data(data)


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


def prepare_section(datadir: str):
    data = dm.get_raw_data(datadir)

    if data is not None:
        with st.expander(
            f"View raw data, {dm.lastModified(dm.get_raw_data_path(datadir))}",
            expanded=False,
        ):
            show_data(data)


def transform_section(datadir: str):
    data = dm.get_clean_data(datadir)

    if data is not None:
        modified = dm.lastModified(dm.get_clean_data_path(datadir))

        with st.expander(
            f"View cleaned/prepared data, {modified}",
            expanded=False,
        ):
            show_data(data)


def train_section(datadir: str):
    data = dm.get_transformed_data(datadir)

    if data is not None:
        modified = dm.lastModified(dm.get_transformed_data_path(datadir))

        with st.expander(f"View transformed data, {modified}", expanded=False):
            show_data(data)

        st.header("Trained data/model")

        suffix = st.selectbox("Select trained data", dm.list_final_data(datadir))
        suffix = "_".join(suffix.split(", ")[0].split("_")[1:])

        with st.expander(f"View data", expanded=False):
            data = dm.get_final_data(datadir, suffix)
            if data is not None:
                modified = dm.lastModified(dm.get_final_data_path(datadir, suffix))
                show_data(data)

        model = dm.get_model(datadir, suffix)
        if model:
            st.write(model.show_topics(formatted=False, num_words=50))


def show_data(data: pd.DataFrame):
    st.dataframe(data)


if __name__ == "__main__":
    streamlit()
