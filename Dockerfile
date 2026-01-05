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

# Install Python dependencies
RUN pip install --no-cache-dir maturin numpy scipy matplotlib scikit-learn click gymnasium

# Build Rust extension
RUN maturin develop --release

# Install package
RUN pip install -e .

# Default command
CMD ["mmr-elites", "--help"]
