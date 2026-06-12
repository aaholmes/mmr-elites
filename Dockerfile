# MMR-Elites Docker Image
# 
# Build: docker build -t mmr-elites .
# Run:   docker run -it mmr-elites mmr-elites benchmark --quick

FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Build and install the package; the maturin build backend compiles the
# Rust extension during pip install (maturin develop would fail here
# because it requires a virtualenv).
RUN pip install --no-cache-dir .

# Default command
CMD ["mmr-elites", "--help"]
