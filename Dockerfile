FROM python:3.9

ARG poetry_version=">=1.1,<1.2"

ENV PYTHONUNBUFFERED=1

RUN addgroup --system nanotoms \
  && adduser --system --ingroup nanotoms nanotoms

RUN pip install "poetry"

COPY poetry.lock pyproject.toml /app/

WORKDIR /app

RUN poetry config virtualenvs.create false \
  && poetry install --no-ansi --no-dev --no-interaction

COPY --chown=nanotoms:nanotoms . /app/

USER nanotoms

CMD ["uvicorn", "nanotoms.api:app", "--proxy-headers", "--host", "0.0.0.0", "--port", "80", "--workers", "4"]
