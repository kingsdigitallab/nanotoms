from typing import Optional

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

import settings
from nanotoms import generate as gm

model = gm.get_model(settings.TEXT_GENERATOR_MODEL_PATH)
tokenizer = gm.get_tokenizer(settings.TEXT_GENERATOR_MODEL_PATH)

app = FastAPI()


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
def generate(prompt: Prompt):
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

    return dict(text=text)
