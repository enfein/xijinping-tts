## 习近平语音合成器

本项目代码从 https://hub.docker.com/r/xijinping615/xi-jinping-tts (已经 404) 使用逆向工程获得，并且更新了 Python 和部分依赖库的版本。

### 下载已编译的容器并运行

```shell
docker pull ghcr.io/enfein/xijinping-tts:latest

docker run --rm -it -p 8501:8501 ghcr.io/enfein/xijinping-tts:latest
```

### 自己编译容器并运行

```shell
git clone https://github.com/enfein/xijinping-tts.git

cd xijinping-tts

docker build . -t <TAG>

docker run --rm -it -p 8501:8501 <TAG>
```
