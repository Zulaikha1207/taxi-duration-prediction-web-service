FROM python:3.9-slim

RUN apt-get update -qq \
&& apt-get install -q -y make curl unzip git-core \
&& apt-get clean -qq
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=1.2.2 python
WORKDIR /app 

COPY [ "pyproject.toml", "poetry.lock", "./" ]
RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi

COPY [ "predict.py", "xgboost.bin", "./" ]
EXPOSE 9696 
ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]