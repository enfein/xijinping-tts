FROM python:3.7-slim-buster as builder

WORKDIR /app

COPY . .

# Assemble data files
RUN rm -rf .git && \
    cd hifigan && \
    cat parts_* > generator_universal.pth.tar && \
    rm parts_* && \
    cd ../output/ckpt/xi-jinping && \
    cat parts_* > 600000.pth.tar && \
    rm parts_*

FROM python:3.7-slim-buster

WORKDIR /app

COPY packages.txt requirements.txt ./
RUN apt-get update && \
    apt-get install --yes $(cat packages.txt) && \
    pip install --no-cache-dir -r requirements.txt

COPY --from=builder /app /app

ENTRYPOINT [ "streamlit", "run", "app.py" ]

EXPOSE 8501
