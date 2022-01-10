import ast
import json
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
from gensim import corpora
from gensim.models import LdaModel


def get_raw_data(datadir: str) -> pd.DataFrame:
    return get_data(get_raw_data_path(datadir))


def get_raw_data_path(datadir: str) -> Path:
    return get_data_path(datadir, "0_raw", "data.csv", True)


def get_data_path(
    dir: str,
    stage: Optional[str] = None,
    filename: Optional[str] = None,
    file_exists: bool = False,
) -> Path:
    path = Path(dir)
    if not path.is_dir():
        raise FileNotFoundError(f"Data directory not found: {path}")

    if not stage:
        return path

    path = path.joinpath(Path(stage))
    if not path.is_dir():
        raise FileNotFoundError(f"Directory not found: {path}")

    if not filename:
        return path

    path = path.joinpath(Path(filename))
    if file_exists and not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    return path


def get_data(filename: Path, kwargs: dict = {}) -> pd.DataFrame:
    return pd.read_csv(filename, **kwargs)  # type: ignore


def get_clean_data(datadir: str) -> pd.DataFrame:
    return get_data(
        get_clean_data_path(datadir), dict(converters=dict(tags=ast.literal_eval))
    )


def get_clean_data_path(datadir: str) -> Path:
    return get_data_path(datadir, "1_interim", "cleaned.csv")


def get_scraped_data_path(datadir: str) -> Path:
    return get_data_path(datadir, "0_external", "scraped.json")


def get_extracted_data(datadir: str) -> pd.DataFrame:
    return get_data(get_extracted_data_path(datadir))


def get_extracted_data_path(datadir: str) -> Path:
    return get_data_path(datadir, "1_interim", "extracted.csv")


def get_transformed_data(datadir: str) -> pd.DataFrame:
    return get_data(get_transformed_data_path(datadir))


def get_transformed_data_path(datadir: str) -> Path:
    return get_data_path(datadir, "1_interim", "transformed.csv")


def get_text_corpus(datadir: str) -> list[list[str]]:
    return load_data(get_text_corpus_path(datadir))


def get_text_corpus_path(datadir: str) -> Path:
    return get_data_path(datadir, "1_interim", "corpus_text.json")


def get_dict_corpus(datadir: str) -> corpora.Dictionary:
    return corpora.Dictionary.load_from_text(get_dict_corpus_path(datadir))


def get_dict_corpus_path(datadir: str) -> Path:
    return get_data_path(datadir, "1_interim", "corpus_dict.txt")


def get_bow_corpus(datadir: str) -> list[list[tuple[int, int]]]:
    return load_data(get_bow_corpus_path(datadir))


def get_bow_corpus_path(datadir: str) -> Path:
    return get_data_path(datadir, "1_interim", "corpus_bow.json")


def get_model(datadir: str, suffix: str) -> LdaModel:
    return LdaModel.load(get_model_path(datadir, suffix).as_posix())


def get_model_path(datadir: str, suffix: str) -> Path:
    return get_data_path(datadir, "2_final", f"topic_model_{suffix}")


def get_final_data(datadir: str, suffix: str) -> pd.DataFrame:
    return get_data(get_final_data_path(datadir, suffix))


def get_final_data_path(datadir: str, suffix: str) -> Path:
    return get_data_path(datadir, "2_final", f"data_{suffix}.csv")


def list_final_data(datadir: str):
    final_data = [
        (os.path.getmtime(p), p.stem, lastModified(p))
        for p in get_data_path(datadir, "2_final").glob("*.csv")
    ]
    final_data.sort(key=lambda x: x[0], reverse=True)

    return [f"{f[1]}, {f[2]}" for f in final_data]


def dump_data(filepath: Path, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_data(filepath: Path):
    with open(filepath, "r") as f:
        return json.load(f)


def lastModified(filepath: Path) -> str:
    modified = os.path.getmtime(filepath)
    return (
        f"last modified: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(modified))}"
    )
