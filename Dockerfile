FROM python:3.12-slim AS runtime
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES="" TF_CPP_MIN_LOG_LEVEL=3
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY pyproject.toml README.md ./
COPY src ./src
COPY tests ./tests
RUN pip install -e .
ENTRYPOINT ["fraudaudit"]

FROM runtime AS dev
RUN pip install jupyterlab==4.2.5
EXPOSE 8888
