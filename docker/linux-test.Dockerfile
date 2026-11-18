# Minimal Linux to run the QUMO test suite: Ubuntu 24.04, the pinned Mojo toolchain from the Makefile, nothing else.
# Build and run from the repo root:  make docker-test   (or see docker/README lines in the Makefile target)
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl git make build-essential libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# uv brings Python 3.12 and the pinned Mojo wheels (make setup)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /qumo
# Toolchain layer first: it only depends on the Makefile, so source edits do not reinstall Mojo.
COPY Makefile ./
RUN make setup

COPY . .
RUN mkdir -p build
CMD ["make", "test"]
