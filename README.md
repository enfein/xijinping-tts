## 习近平语音合成器

本项目代码从 https://hub.docker.com/r/xijinping615/xi-jinping-tts (已经 404) 使用逆向工程获得，并且更新了 Python 和部分依赖库的版本。

### 编译容器

```shell
docker build . -t <TAG>
```

### 运行容器

```shell
docker run --rm -it -p 8501:8501 <TAG>
```
