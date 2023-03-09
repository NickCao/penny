FROM docker.io/rust:bullseye
RUN apt-get update && \
    apt-get install -y libclang-dev llvm && \
    rm -rf /var/lib/apt/lists/*
