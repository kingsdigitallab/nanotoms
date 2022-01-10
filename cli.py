import numpy as np
import spacy.cli.download as spacy_download
import typer
from spacy import util
from tqdm import tqdm

import settings
from nanotoms import data as dm
from nanotoms import etl
from nanotoms import features as fm
from nanotoms import train as tm
from nanotoms import visualize as vm

app = typer.Typer()


@app.command()
def prepare(datadir: str = settings.DATA_DIR.name, scrape: bool = False):
    """
    Download, generate, and process data.

    :param datadir: Path to the data directory
    :param scrape: Download data from the URLs in the data?
    """
    with tqdm(total=4, desc="Preparing data...") as progress:
        data = dm.get_raw_data(datadir)
        progress.update(1)

        data = etl.clean(data)
        data.to_csv(dm.get_clean_data_path(datadir), index=False)
        progress.update(1)

        filepath = dm.get_scraped_data_path(datadir)
        if scrape or not filepath.is_file():
            scraped_data = etl.scrape(data, warn)
            dm.dump_data(filepath, scraped_data)
        else:
            scraped_data = dm.load_data(filepath)
        progress.update(1)

        extracted = etl.extract(
            data,
            scraped_data,
            settings.SCRAPE_DOMAINS,
            settings.SCRAPE_TEXT_CLEAN_PATTERN,
            warn,
        )
        extracted.to_csv(dm.get_extracted_data_path(datadir), index=False)
        progress.update(1)


def error(msg: str):
    typer.echo()
    typer.secho(f"Error: {msg}", fg=typer.colors.RED)
    raise typer.Abort()


def warn(msg: str):
    typer.echo()
    typer.secho(f"Warn: {msg}", fg=typer.colors.YELLOW)


@app.command()
def transform(
    datadir: str = settings.DATA_DIR.name,
    language_model: str = settings.SPACY_LANGUAGE_MODEL,
    stop_words: list[str] = settings.SPACY_EXTRA_STOP_WORDS,
    entity_labels: list[str] = settings.SPACY_ENTITY_TYPES,
):
    """
    Transform prepared data into features for modelling.

    :param datadir: Path to the data directory
    :param spacy_language_model: Name of the spacy language model to use
    """
    if language_model not in util.get_installed_models():
        warn(f"The spacy language model {language_model} is not installed")
        download = typer.confirm("Would you like to download the model?")

        if not download:
            error("Language model not available")

        spacy(language_model)

    with tqdm(total=6, desc="Adding features to data...") as progress:
        cleaned_df = dm.get_clean_data(datadir)
        progress.update(1)

        extracted_df = dm.get_extracted_data(datadir)
        progress.update(1)

        data = fm.add_features(
            cleaned_df,
            extracted_df,
            language_model,
            stop_words,
            entity_labels,
        )
        data.to_csv(dm.get_transformed_data_path(datadir), index=False)
        progress.update(1)

        text_corpus = fm.text_corpus(data["lemmas"])
        dm.dump_data(dm.get_text_corpus_path(datadir), text_corpus)
        progress.update(1)

        dict_corpus = fm.corpus_to_dict(
            text_corpus, no_below=settings.MINIMUM_NUMBER_OF_DOCS_WITH_TERM
        )
        dict_corpus.save_as_text(dm.get_dict_corpus_path(datadir))
        progress.update(1)

        bow_corpus = fm.bow_corpus(text_corpus, dict_corpus)
        dm.dump_data(dm.get_bow_corpus_path(datadir), bow_corpus)
        progress.update(1)


@app.command()
def spacy(name: str = settings.SPACY_LANGUAGE_MODEL):
    """
    Download spacy language model.

    :name: Language model name
    """
    spacy_download(name)


@app.command()
def train(
    datadir: str = settings.DATA_DIR.name,
    number_of_topics: int = settings.NUMBER_OF_TOPICS,
    passes: int = settings.NUMBER_OF_PASSES,
    minimum_probability: float = settings.TOPICS_MINIMUM_PROBABILITY,
    show: bool = typer.Option(
        ...,
        prompt="Print topics",
        confirmation_prompt=True,
    ),
):
    """
    Train topic model.

    :param datadir: Path to the data directory
    :param number_of_topics: Number of topics to be extracted from the data
    :param show: Print the extracted topics?
    """
    typer.echo()

    with tqdm(total=6, desc="Training...") as progress:
        data = dm.get_transformed_data(datadir)
        progress.update(1)

        bow_corpus = dm.get_bow_corpus(datadir)
        progress.update(1)

        dict_corpus = dm.get_dict_corpus(datadir)
        progress.update(1)

        text_corpus = dm.get_text_corpus(datadir)
        progress.update(1)

        model = tm.model(
            bow_corpus,
            dict_corpus,
            passes=passes,
            num_topics=number_of_topics,
            minimum_probability=minimum_probability,
        )
        model.save(dm.get_model_path(datadir, f"{number_of_topics}").as_posix())
        progress.update(1)

        data = tm.add_topics_to_documents(model, bow_corpus, data, number_of_topics)
        data.to_csv(
            dm.get_final_data_path(datadir, f"{number_of_topics}"),
            index=False,
        )
        progress.update(1)

        score = tm.coherence_score(
            model,
            text_corpus,
            dict_corpus,
            settings.COHERENCE_MEASURE,
        )
        progress.update(1)

    typer.echo()
    typer.echo(f"Trained model {settings.COHERENCE_MEASURE} score: {score}")

    if show:
        vm.print_topics(model, n=number_of_topics, writer=typer.echo)


@app.command()
def tune(
    datadir: str = settings.DATA_DIR.name,
    number_of_topics: int = settings.NUMBER_OF_TOPICS,
    show: bool = typer.Option(
        ...,
        prompt="Print topics",
        confirmation_prompt=True,
    ),
):
    """
    Trune topic model. Iterate over different settings to try to improve the
    trained model.

    :param datadir: Path to the data directory
    :param number_of_topics: Number of topics to be extracted from the data
    :param show: Print the extracted topics?
    """
    data = dm.get_transformed_data(datadir)

    bow_corpus = dm.get_bow_corpus(datadir)

    dict_corpus = dm.get_dict_corpus(datadir)

    text_corpus = dm.get_text_corpus(datadir)

    number_of_topics = settings.NUMBER_OF_TOPICS

    num_topics_range = range(
        tm.get_min_number_of_topics(number_of_topics),
        tm.get_max_number_of_topics(number_of_topics),
        1,
    )
    alphas = list(np.arange(0.01, 1, 0.3))
    alphas.append("symmetric")
    alphas.append("asymmetric")

    etas = list(np.arange(0.01, 1, 0.3))
    etas.append("symmetric")

    max_score = 0
    model = None
    parameters = {}

    for num_topics in tqdm(num_topics_range, desc="Tunning model..."):
        for alpha in tqdm(alphas, desc=f"n:{num_topics}, alpha"):
            for eta in tqdm(etas, desc=f"n:{num_topics}, alpha:{alpha}, eta"):
                trained_model = tm.model(
                    bow_corpus,
                    dict_corpus,
                    passes=settings.NUMBER_OF_PASSES,
                    num_topics=num_topics,
                    alpha=alpha,
                    eta=eta,
                    minimum_probability=settings.TOPICS_MINIMUM_PROBABILITY,
                )

                score = tm.coherence_score(
                    trained_model,
                    text_corpus,
                    dict_corpus,
                    settings.COHERENCE_MEASURE,
                )

                # alpha_str = f"{alpha:.2f}" if isinstance(alpha, float) else alpha
                # eta_str = f"{eta:.2f}" if isinstance(eta, float) else eta

                # num_topics_progress.label = (
                # f"topics: {num_topics} "
                # f"alpha: {alpha_str} "
                # f"eta: {eta_str} "
                # f"score: {score:.2f}"
                # )

                if score > max_score:
                    max_score = score
                    model = trained_model
                    parameters["topics"] = num_topics
                    parameters["alpha"] = alpha
                    parameters["eta"] = eta

    typer.echo()
    typer.echo(f"max score: {max_score}")
    for k, v in parameters.items():
        typer.echo(f"{k}: {v}")

    if model:
        model.save(
            dm.get_model_path(datadir, f"tunned_{parameters['topics']}").as_posix()
        )

        data = tm.add_topics_to_documents(model, bow_corpus, data, parameters["topics"])
        data.to_csv(
            dm.get_final_data_path(datadir, f"tunned_{parameters['topics']}"),
            index=False,
        )

        if show:
            vm.print_topics(model, n=parameters["topics"], writer=typer.echo)


if __name__ == "__main__":
    app()
