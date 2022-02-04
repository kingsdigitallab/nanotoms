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
        data["title"]
        .str.cat(data["description"], na_rep="", sep=". ")
        .str.cat(sep="\n")
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


def get_inventories(path: str) -> str:
    roles = pd.read_excel(path, sheet_name="Roles1")

    people = pd.read_excel(path, sheet_name="People")
    people = people.dropna(subset=["DocID"])
    people["DocID"] = people["DocID"].astype(int)
    people["person_name"] = people["First name"].str.cat(people["Surname"], sep=" ")

    people = people.merge(roles, on="Person ID", suffixes=("_person", "_role"))

    people_by_id = people.drop_duplicates("Person ID").set_index("Person ID")
    testators = (
        people[people["Role"] == "Testator"].drop_duplicates("DocID").set_index("DocID")
    )

    people = people.dropna(subset=["Occupation"])
    people = people[people["Occupation"] != "Not stated"]

    documents = pd.read_excel(path, sheet_name="Documents")
    documents["Document ID"] = documents["Document ID"].astype(int)

    objects = pd.read_excel(path, sheet_name="Inventory_objects")
    objects = objects.dropna(subset=["DocID"])
    objects["DocID"] = objects["DocID"].astype(int)
    objects["Number of objects"] = objects["Number of objects"].fillna(0).astype(int)
    objects = objects.merge(documents, left_on="DocID", right_on="Document ID")
    objects = objects.merge(people, left_on="Document ID", right_on="DocID")
    objects["Location"] = objects["Location"].fillna("")

    objects["item"] = objects["Object description"]
    objects["place"] = objects.apply(
        lambda x: get_place(x["Location"], x["Region"]), axis=1
    )

    object_groups = objects.groupby(
        by=["Document ID", "Inventory number", "Person ID", "person_name"]
    )

    inventories = object_groups.apply(lambda x: inventory(x))
    inventories = [inv for inv in inventories.values.tolist() if inv]

    recipients = pd.read_excel(path, sheet_name="BequestRecipients")
    recipients["recipient_name"] = recipients["PersonID"].map(
        people_by_id["person_name"]
    )

    bequests = pd.read_excel(path, sheet_name="Will_bequests")
    bequests = bequests.dropna(subset=["DocID"])
    bequests["DocID"] = bequests["DocID"].astype(int)
    bequests["Conditions"] = bequests["Conditions"].fillna("")
    bequests = documents.merge(bequests, left_on="Document ID", right_on="DocID")
    bequests["WillReligPreamble"] = bequests["WillReligPreamble"].fillna("")
    bequests["WillFuneraryProvision"] = bequests["WillFuneraryProvision"].fillna("")
    bequests["bequest_testator"] = bequests["DocID"].map(testators["person_name"])
    bequests = bequests.merge(recipients, left_on="Bequest ID", right_on="BequestID")
    bequests = bequests.drop_duplicates()

    wills = bequests.groupby(by=["Will ID"]).apply(lambda x: will(x))
    inventories.extend(wills.values.tolist())

    return "\n\n".join(inventories)


def get_place(location: str, region: str) -> str:
    if location and region:
        return f"{location}, {region}"

    if location:
        return location

    if region:
        return region

    return ""


def inventory(df) -> str:
    headings = df.groupby(by=["Heading"])
    items = headings.apply(
        lambda x: (f"{x['Heading'].iloc[0]} are {', '.join(x['item'].values.tolist())}")
    )

    if items.empty:
        return ""

    return (
        "Listed on the inventory of "
        f"{df['person_name'].iloc[0]}, "
        f"{df['Occupation'].iloc[0]} "
        f"of {df['place'].iloc[0]}. "
        f"{'; '.join(items.values.tolist())}."
    )


def will(df) -> str:
    bequests = df.groupby(by=["Bequest"])
    items = bequests.apply(
        lambda x: (
            f"{x['Bequest'].iloc[0]} {x['RecipDescription'].iloc[0]} "
            f"{x['Conditions'].iloc[0]}"
        ).strip()
    )

    return modernise(
        f"I {df['bequest_testator'].iloc[0]} bequethe "
        f"{', '.join(items.values.tolist())}. "
        f"{df['WillReligPreamble'].iloc[0]} {df['WillFuneraryProvision'].iloc[0]}."
    )


def modernise(text: str) -> str:
    return text
