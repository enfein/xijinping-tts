## 习近平语音合成器

本项目代码从 https://hub.docker.com/r/xijinping615/xi-jinping-tts 使用逆向工程获得。

编译容器

```shell
docker build . -t <TAG>
```

运行容器

```shell
docker run --rm -it -p 8051:8051 <TAG>
```
