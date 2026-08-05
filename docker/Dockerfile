# syntax=docker/dockerfile:1
FROM vggfiit/vgg-torch:2.0.0

# Update the APT-GET
RUN apt-get update --fix-missing
RUN apt-get install apt-transport-https
RUN apt-get install ca-certificates
RUN update-ca-certificates

RUN curl -fsSL https://deb.nodesource.com/setup_24.x | bash - \
    && apt-get update --fix-missing \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Claude Code in the container
RUN curl -fsSL https://claude.ai/install.sh | bash \
    && echo 'export PATH="$HOME/.local/bin:$PATH"' >> /root/.bashrc
ENV PATH="/root/.local/bin:${PATH}"

# Install required packages
WORKDIR /setup
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Working dir
ENV PYTHONPATH=/workspace/dp-code
WORKDIR /workspace/dp-code