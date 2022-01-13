from collections import defaultdict

import pandas as pd
import spacy
from gensim import corpora
from gensim.models import Phrases
from spacy.tokens import Doc


def add_features(
    cleaned_df: pd.DataFrame,
    extracted_df: pd.DataFrame,
    language_model: str,
    stop_words: list[str],
    entity_types: list[str],
) -> pd.DataFrame:
    data = cleaned_df.copy()
    data = data.join(extracted_df[["web:description"]])

    data["content"] = (
        data["title"]
        .str.cat(data["theme"], sep=".\n")
        .str.cat(data["type"], sep=".\n")
        .str.cat(data["tags"].apply(lambda x: ", ".join(x)), sep=".\n")
        .str.cat(data["description"], sep=".\n")
    )
    data["content"] = data["content"].str.cat(data["web:description"], na_rep="")
    data["content"] = (
        data["content"].fillna("").astype("str").apply(lambda x: x.strip())
    )

    nlp = get_language_model(language_model, stop_words)

    data["doc"] = data["content"].apply(nlp)
    data["lemmas"] = data["doc"].apply(get_lemmas)

    entities = data["doc"].apply(lambda x: get_entities(x, entity_types))
    data["entities"] = entities.apply(
        lambda x: [
            f"{label.split(':')[1]}: {entity}"
            for label, items in x.items()
            for entity in items
        ]
    )

    entities_df = pd.DataFrame(entities.values.tolist())
    entities_df = entities_df[sorted(entities_df.columns)]
    data = data.join(entities_df)

    data = data.drop(columns=["doc"])

    return data  # type: ignore


def get_language_model(
    language_model: str, stop_words: list[str] = []
) -> spacy.language.Language:
    nlp = spacy.load(language_model)
    nlp.Defaults.stop_words.update(stop_words)

    for stop in stop_words:
        nlp.vocab[stop].is_stop = True

    return nlp


def get_lemmas(doc: Doc) -> list:
    return [
        token.lemma_.lower()
        for token in doc
        if not token.is_stop and not token.is_punct and not token.is_space
    ]


def get_entities(doc: Doc, entity_types: list[str] = []) -> dict[str, list[str]]:
    entities = defaultdict(list)

    for entity in [
        (ent.label_, ent.text) for ent in doc.ents if ent.label_ in entity_types
    ]:
        entities[f"entity:{entity[0]}"].append(entity[1])

    for k, v in entities.items():
        entities[k] = sorted(list(set(v)))

    return entities


def get_descriptions(data: pd.DataFrame) -> list[str]:
    descriptions = data[["title", "description"]].copy()
    descriptions["text"] = (
        data["title"].str.cat(data["description"], na_rep="", sep=". ").str.cat(sep="\n")
    )

    return descriptions["text"].values.tolist()


# https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html
def text_corpus(texts: pd.Series) -> list[list[str]]:
    tokens = texts.tolist()
    bigrams = Phrases(tokens)
    trigrams = Phrases(bigrams[tokens], min_count=1)

    return list(trigrams[bigrams[tokens]])


def corpus_to_dict(
    text_corpus: list[list[str]],
    no_below: int,
) -> corpora.Dictionary:
    dictionary = corpora.Dictionary(text_corpus)
    dictionary.filter_extremes(no_below=no_below)

    return dictionary


def bow_corpus(
    text_corpus: list[list[str]], dict_corpus: corpora.Dictionary
) -> list[list[tuple[int, int]]]:
    return [dict_corpus.doc2bow(text) for text in text_corpus]  # type: ignore
