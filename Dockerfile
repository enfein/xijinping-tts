FROM python:3.7-slim-buster

WORKDIR /app

COPY packages.txt requirements.txt ./
RUN apt-get update && \
    apt-get install --yes $(cat packages.txt) && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

# Assemble data files
RUN cd hifigan && \
    cat parts_* > generator_universal.pth.tar && \
    cd ../output/ckpt/xi-jinping && \
    cat parts_* > 600000.pth.tar

ENTRYPOINT [ "streamlit", "run", "app.py" ]

EXPOSE 8501
