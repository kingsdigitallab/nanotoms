from functools import lru_cache
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

import settings
from nanotoms import data as dm
from nanotoms import generate as gm
from nanotoms import search as sm

datadir = settings.DATA_DIR.name

data = dm.get_transformed_data(datadir)

model = gm.get_model(settings.TEXT_GENERATOR_MODEL_PATH.as_posix())
tokenizer = gm.get_tokenizer(settings.TEXT_GENERATOR_MODEL_PATH.as_posix())

embeddings = sm.get_embeddings(dm.get_embeddings_path(datadir).as_posix())

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.API_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Prompt(BaseModel):
    prompt: str
    do_sample: Optional[bool] = True
    early_stopping: Optional[bool] = False
    no_repeat_ngram_size: Optional[int] = 2
    max_length: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_k: Optional[int] = 50


@app.get("/")
async def docs_redirect():
    return RedirectResponse("/redoc")


@app.post("/generate/")
def generate(prompt: Prompt) -> str:
    """
    Generate text based on the given prompt.

    :param prompt: Prompt to generate text for
    :param do_sample: Choose words based on their conditional probability?
    :param early_stopping: Stop at last full sentence (if possible)?
    :param no_repeat_ngram_size: N-gram size that can't occur more than once
    :param max_length: Maximum length of the generated text
    :param temperature: How sensitive the algorithm is to selecting least common
                        optionsfor the generated text
    :param top_k: How many potential outcomes are considered before generating the text

    @returns str: the generated text.
    """
    text = gm.generate(
        model,
        tokenizer,
        prompt.prompt,
        dict(
            do_sample=prompt.do_sample,
            early_stopping=prompt.early_stopping,
            no_repeat_ngram_size=prompt.no_repeat_ngram_size,
            max_length=prompt.max_length,
            temperature=prompt.temperature,
            top_k=prompt.top_k,
        ),
    )

    return text


@app.get("/search/")
@lru_cache(maxsize=64)
def search(query: str, limit: int = 10) -> list[dict[str, Any]]:
    """
    Search objects in the data using a semantic search, finds by meaning as well as by
    keyword.

    :param query: The query to search for
    :param limit: Maximum number of results to return

    @returns a list of dictionaries containing the found objects.
    """
    if data is None:
        return []

    found = sm.search(data, embeddings, query, limit)
    records = found[["title", "description", "score"]].fillna("").to_dict("records")
    return records
