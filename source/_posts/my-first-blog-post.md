---
title: My first blog post
date: 2023-07-05 14:00:27
tags: [one]
# 文章主页背景图片
bgImg: https://i.postimg.cc/d0Sh2PNd/11-280-3.jpg
# 文章图片
img: https://i.postimg.cc/d0Sh2PNd/11-280-3.jpg
categories: [Pointless small talk]
---

# 如何搭建一个属于自己的博客网站？

## 起因

`今天早些的时候我在看Nginx的教程，那位youtuber就使用了Hexo搭建的博客进行反向代理测试，于是我就很感兴趣，便去了解了一下`

{% linkgroup %}
{% link 30分钟Nginx入门教程 - YouTube,
https://youtu.be/sCJcusORiE8, https://pica.zhimg.com/80/v2-970dd5538f106dd6be064c4eafc01c36_1440w.webp %}
{% endlinkgroup %}

访问[Hexo](https://hexo.io/)以了解更多内容。

我选择了一个自己蛮喜欢的主题。

我也修改了很多配置项，不过总的说还是非常容易理解的。

## 遇到的问题

这过程中最最最离谱的问题（对我来说）是：关于git push的问题，经常push不上去，所以我就排查了问题原因，试过了修改`/etc/hosts`的配置

```shell
sudo bash -c 'echo "<IP> <hostname>" >> /etc/hosts'
```

也尝试了修改`http.version`

```shell
git config --global http.version HTTP/1.1
```

发现没有用，最后，我发现是代理的问题，因为在浏览器中打开GitHub没问题，但使用git的时候就`could't connect`，原因在于git默认不使用代理端口

```shell
git config --global http.proxy 127.0.0.1:[port]
git config --global https.proxy 127.0.0.1:[port]
```

`port`为代理端口号

这个问题就如此解决了。

## 后言

说实话，太累了写这东西。