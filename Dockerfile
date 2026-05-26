# syntax=docker/dockerfile:1

FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHON_BIN=python3 \
    MAX_WORKERS=3 \
    DB_PREFIX=/results/betamax

WORKDIR /opt/betamax

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        clang \
        cmake \
        make \
        pkg-config \
        libre2-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x run_bms.sh run_ddmax.sh build_all.sh validators/build_validators.sh \
    && cmake -S betamax_cpp -B betamax_cpp/build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build betamax_cpp/build -j"$(nproc)" \
    && c++ -std=c++17 -O3 -DNDEBUG erepair.cpp -o erepair \
    && ./validators/build_validators.sh \
    && mkdir -p /results

ENTRYPOINT ["./run_bms.sh"]
CMD ["regex"]