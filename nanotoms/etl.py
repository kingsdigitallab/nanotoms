import re
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
import urlextract
from bs4 import BeautifulSoup


def clean(data: pd.DataFrame) -> pd.DataFrame:
    cleaned = data.copy()
    cleaned.columns = cleaned.columns.str.lower()
    cleaned = cleaned[
        [
            "title",
            "theme",
            "type",
            "media",
            "attribution",
            "tags",
            "url",
            "description",
            "image rights",
        ]
    ]

    if cleaned is not None:
        cleaned["tags"] = cleaned["tags"].fillna("").astype(str).apply(get_tags)
        cleaned["url"] = cleaned["url"].fillna("").astype(str).apply(get_urls)

    return cleaned  # type: ignore


def get_tags(text: str) -> list:
    if not text:
        return []

    return sorted(
        list(
            set(
                [
                    tag
                    for value in text.split("; ")
                    if (tag := value.strip()) and len(tag) > 0
                ]
            )
        )
    )


def get_urls(text: str) -> list:
    if not text:
        return []

    extractor = urlextract.URLExtract()

    return [
        url if url.startswith("http") else f"http://{url}"
        for url in extractor.find_urls(text)
    ]


def scrape(data: pd.DataFrame, writer=print) -> dict[str, str]:
    scraped = {}

    for urls in data["url"]:
        for url in urls:
            try:
                response = requests.get(url)
                if not response.ok:
                    continue

                soup = BeautifulSoup(response.text, "html.parser")

                body = soup.body
                if not body:
                    continue

                scraped[url] = str(body)
            except requests.exceptions.RequestException as e:
                writer(f"{url}: {e}")

    return scraped


def extract(
    data: pd.DataFrame,
    scraped_data: dict[str, str],
    domain_options: dict,
    clean_pattern: re.Pattern,
    writer=print,
) -> pd.DataFrame:
    extracted = pd.DataFrame()

    extracted["url"] = data["url"]
    extracted["web:description"] = data["url"].apply(
        lambda x: get_web_description(
            scraped_data, x, domain_options, clean_pattern, writer
        )
    )

    return extracted


def get_web_description(
    scraped_data: dict[str, str],
    urls: list,
    domain_options: dict,
    clean_pattern: re.Pattern,
    writer=print,
) -> str:
    description = ""

    if not urls:
        return description

    for url in urls:
        if url not in scraped_data:
            continue

        soup = BeautifulSoup(scraped_data[url], "html.parser")

        options = get_domain_options(url, domain_options, writer)
        if not options:
            continue

        path = options["path"]

        if "kwargs" in options:
            kwargs = options["kwargs"]
            nodes = soup.find_all(path, **kwargs)
        else:
            nodes = soup.select(path)

        for node in nodes:
            description = f"{description} {clean_text(clean_pattern, node.text)}"

    return description.strip()


def get_domain_options(url: str, domain_options: dict, writer=print) -> Optional[dict]:
    domain = url.split("/")[2]

    if domain not in domain_options:
        writer(f"domain {domain} not found in domain options")
        return None

    return domain_options[domain]


def clean_text(clean_pattern: re.Pattern, text: str) -> str:
    return clean_pattern.sub(" ", text)
