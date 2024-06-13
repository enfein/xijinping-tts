import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from datetime import datetime
from glob import glob
from os import path

import numpy as np
import streamlit as st
import torch
import yaml
from pypinyin import Style, pinyin

from text import text_to_sequence
from utils.model import get_model, get_vocoder
from utils.tools import pad_1D, synth_samples, to_device

id = str(uuid.uuid4())

st.set_page_config(page_title="习近平语音合成器", page_icon="😅")
st.title("习近平语音合成器")
st.markdown("""
[***关注我的 YouTube 频道！***](https://www.youtube.com/channel/UCMNmgNsEeWRL-KefLGvpqdw)
- 不支持在苹果 Safari 内核的浏览器上直接播放音频，包括 iPhone 和 iPad 上的所有浏览器。你可以把语音下载到本地后再播放；
- 所有非汉字或阿拉伯数字的字符将被忽略，比如英文单词将不会被发音；
- 阿拉伯数字会直接按照逐个数字发音，比如 “123” 会被念作 “一二三” 而不是 “一百二十三”。如果你想要念出 “一百二十三”，请直接输入汉字 “一百二十三”；
- 每一句话不要太长，否则可能会导致发音异常；
- 多音字有概率会念错。
""")
text_to_synthesize = st.text_area("", max_chars=250, placeholder="请在此输入要合成的文本", height=250)
speed = st.slider("时长倍数（越大语速越慢）", min_value=0.5, max_value=2.0, value=1.15, step=0.05)

args = argparse.Namespace(
    mode = "inference",
    source = "text.txt",
    duration_control = 1.1,
    pitch_control = 1.0,
    energy_control = 1.0,
    speaker_id = 218,
    restore_step = 600000,
    preprocess_config = "config/xi-jinping/preprocess.yaml",
    model_config = "config/xi-jinping/model.yaml",
    train_config = "config/xi-jinping/train.yaml",
)


@st.cache_resource
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_configs():
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    return preprocess_config, model_config, train_config, configs


device = get_device()
preprocess_config, model_config, train_config, configs = load_configs()


@st.cache_resource
def load_model():
    return get_model(args, configs, device, train=False)


@st.cache_resource
def load_vocoder():
    return get_vocoder(model_config, device)


@st.cache_resource
def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")

    phones = "{" + " ".join(phones) + "}"
    # print("Raw Text Sequence: {}".format(text))
    # print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(model, step, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    wav_file_paths = []

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            wav_file_paths_batch = synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
            )
            wav_file_paths.extend(wav_file_paths_batch)

    return wav_file_paths


def get_clean_text_lines(dirty_text):
    clean_text = dirty_text

    number_to_hanzi_dict = {
        "0": "零",
        "1": "一",
        "2": "二",
        "3": "三",
        "4": "四",
        "5": "五",
        "6": "六",
        "7": "七",
        "8": "八",
        "9": "九",
    }

    for number, hanzi in number_to_hanzi_dict.items():
        clean_text = clean_text.replace(number, hanzi)

    newline_punctuations = [
        "。", ".",
        "？", "?",
        "！", "!",
        "，", ",", "、",
        "；", ";",
        "—", "-",
        "～", "~",
        "…",
        ":", "：",
        "(", "（", ")", "）"
    ]

    for newline_punctuation in newline_punctuations:
        clean_text = clean_text.replace(newline_punctuation, "\n")

    lines = []
    for line in clean_text.splitlines():
        hanzi_line = "".join(re.findall(r"[\u4e00-\u9fa5]+", line))
        if hanzi_line != "":
            lines.append(hanzi_line)

    return lines


def concat_to_mp3_bytes(wav_file_paths):
    temp_dir = tempfile.gettempdir()
    
    wav_list_file_path = path.join(temp_dir, id + ".txt")
    mp3_file_path = path.join(temp_dir, id + ".mp3")
    silence_file_path = path.abspath("silence.wav")
    with open(wav_list_file_path, "w+", encoding="utf-8") as f:
        for wav_file_path in wav_file_paths:
            f.write(f"file 'file:{path.abspath(wav_file_path)}'\n")
            f.write(f"file 'file:{silence_file_path}'\n")

    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", wav_list_file_path, mp3_file_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with open(mp3_file_path, "rb") as f:
        mp3_bytes = f.read()

    os.remove(wav_list_file_path)
    os.remove(mp3_file_path)
    for wav_file_path in wav_file_paths:
        os.remove(wav_file_path)

    return mp3_bytes


if st.button("开始合成"):
    lines = [line.strip()[:100] for line in get_clean_text_lines(text_to_synthesize)]

    if len(lines) > 0:
        json_log = {
            "time": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
            "text": text_to_synthesize,
        }
        print(json.dumps(json_log, ensure_ascii=False))

        with st.spinner("正在合成…"):
            # Load model
            model = load_model()

            # Load vocoder
            vocoder = load_vocoder()

            raw_texts = lines

            ids = []
            for line in lines:
                ids.append(str(uuid.uuid4()))

            speakers = np.array([args.speaker_id] * len(lines))     
            texts = [preprocess_mandarin(line, preprocess_config) for line in lines]
            text_lens = np.array([len(text) for text in texts])
            max_text_len = max(text_lens)
            texts = pad_1D(texts)
            batchs = [(ids, raw_texts, speakers, texts, text_lens, max_text_len)]

            control_values = args.pitch_control, args.energy_control, speed

            wav_file_paths = synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)

            mp3_bytes = concat_to_mp3_bytes(wav_file_paths)
            st.success("合成完成！")
            st.audio(mp3_bytes, format="audio/mp3")
            st.download_button("下载语音", mp3_bytes, "audio.mp3")
