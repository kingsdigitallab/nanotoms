from typing import Optional

import pandas as pd
from txtai.embeddings import Embeddings


def index(data: pd.DataFrame) -> Embeddings:
    texts = data["content"].fillna(data["title"]).values.tolist()

    embeddings = get_embeddings(None)
    embeddings.index([[idx, text, None] for idx, text in enumerate(texts)])

    return embeddings


def get_embeddings(path: Optional[str]) -> Embeddings:
    embeddings = Embeddings()

    if path and embeddings.exists(path):
        embeddings.load(path)
    else:
        embeddings = Embeddings(
            dict(path="sentence-transformers/all-MiniLM-L6-v2", backend="annoy")
        )

    return embeddings


def search(
    data: pd.DataFrame, embeddings: Embeddings, query: str, limit: int = 10
) -> pd.DataFrame:
    docs = embeddings.search(query, limit)

    index = [d[0] for d in docs]
    scores = [d[1] for d in docs]

    found = data[data.index.isin(index)].copy()
    found["score"] = scores

    return found
