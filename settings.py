import re
from pathlib import Path

ROOT_DIR = Path(".")

DATA_DIR = ROOT_DIR.joinpath("data")
if not DATA_DIR.is_dir():
    DATA_DIR.mkdir(parents=True)

PROJECT_TITLE = "Narrative Atoms"

SCRAPE_DOMAINS = {
    "artuk.org": dict(path="div.desc"),
    "britainexpress.com": dict(path="div.article"),
    "collections.museumoflondon.org.uk": dict(path=".object-details"),
    "collections.shakespeare.org.uk": dict(path="div.description"),
    "collections.vam.ac.uk": dict(path="#vam-main-content"),
    "earlymusicshop.com": dict(path=".product-description"),
    "commons.wikimedia.org": dict(path=".mw-mmv-title"),
    "finds.org.uk": dict(path="div", kwargs=dict(property="pas:description")),
    "historicengland.org.uk": dict(path=".markedContent"),
    "luna.folger.edu": dict(path="div.singleValueValue"),
    "rammcollections.org.uk": dict(path="dd"),
    "research.britishmuseum.org": dict(path=".object-detail__data-list"),
    "shakespeareandbeyond.folger.edu": dict(path="div.entry-content"),
    "shakespearedocumented.folger.edu": dict(path="div.field-name-body"),
    "www.archaeology.org": dict(path=".article_content"),
    "www.artuk.org": dict(path="div.desc"),
    "www.bmagic.org.uk": dict(path=".o_info"),
    "www.bonhams.com": dict(path=".core_content"),
    "www.geograph.org.uk": dict(
        path=".caption640", kwargs=dict(itemprop="description")
    ),
    "www.lymeregis-parishchurch.org": dict(path=".paragraph"),
    "www.shakespeare.org.uk": dict(path=".rich-text"),
    "www.wealddown.co.uk": dict(path="#portfolio-extra"),
}

SCRAPE_TEXT_CLEAN_PATTERN: re.Pattern = re.compile(r"(\s\s+)|([\x80-\xff])")

# https://spacy.io/models
SPACY_LANGUAGE_MODEL: str = "en_core_web_md"

SPACY_EXTRA_STOP_WORDS: list[str] = ["Miss", "Mr", "Mrs", "Ms"]

# https://spacy.io/models/en#en_core_web_sm-labels
SPACY_ENTITY_TYPES: list[str] = [
    "DATE",
    "EVENT",
    "GPE",
    "LOC",
    "NORP",
    "PERSON",
]


def get_min_number_of_topics(n: int) -> int:
    min_n: int = 2

    n = int(n / 2)

    if n > 1:
        return n

    return min_n


def get_max_number_of_topics(n: int) -> int:
    return int(n * 1.5)


# https://radimrehurek.com/gensim/models/ldamulticore.html
MINIMUM_NUMBER_OF_DOCS_WITH_TERM: int = 3
NUMBER_OF_TOPICS: int = 10
MIN_NUMBER_OF_TOPICS: int = get_min_number_of_topics(NUMBER_OF_TOPICS)
MAX_NUMBER_OF_TOPICS: int = get_max_number_of_topics(NUMBER_OF_TOPICS)
NUMBER_OF_PASSES: int = 10
TOPICS_MINIMUM_PROBABILITY: float = 1 / NUMBER_OF_TOPICS * 1.5


# https://radimrehurek.com/gensim/models/coherencemodel.html#gensim.models.coherencemodel.CoherenceModel
# one of 'c_v', 'c_uci', 'c_npmi'
COHERENCE_MEASURE: str = "c_v"


TEXT_GENERATOR_MODEL_PATH = "../gpt-neo-125M-nanotoms"
