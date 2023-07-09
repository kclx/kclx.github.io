---
title: GPT API使用教程
tags: [ 'GPT' ]
categories: [ Computer Technology ]
top: false
comments: true
lang: en
toc: true
excerpt: GPT API使用教程
swiper: false
swiperDesc: GPT API使用教程
tocOpen: true
onlyTitle: false
share: true
copyright: true
donate: true
bgImgTransition: fade
bgImgDelay: 180000
prismjs: true
mathjax: false
imgTop: ture
date: 2023-07-09 20:49:44
updated: 2023-07-09 20:49:44
swiperImg:
bgImg:
img: https://s1.imagehub.cc/images/2023/07/09/2023-07-09-20.52.31.png
---

# 什么是GPT

ChatGPT是OpenAI开发的一种基于语言模型的聊天代理工具。它建立在强大的GPT（Generative Pre-trained
Transformer）模型的基础上，可以对自然语言进行理解和生成。

GPT是一种基于Transformer架构的语言模型，通过大规模的预训练来学习语言的潜在模式和结构。在预训练阶段，模型使用了大量的公开可用的互联网文本数据，从而获得了丰富的语言知识。在预训练之后，通过微调和特定任务的训练，模型可以用于各种自然语言处理任务，包括对话生成。

ChatGPT是GPT模型在对话生成任务上的应用。它经过训练，使得模型能够理解和生成连贯的对话。您可以向ChatGPT提供一个对话历史或问题，并从模型中获得相应的回复。ChatGPT可以用于构建智能聊天机器人、客服代理、语言翻译等应用。

OpenAI通过对大规模数据集进行训练，使ChatGPT能够生成流畅、有逻辑的回复，并且具备一定的上下文理解能力。然而，它也可能存在生成不准确、含有偏见或不恰当回答的情况，因此在使用ChatGPT时需要注意对输出进行筛选和审查，以确保生成的回复符合期望和要求。

## GPT-3.5与GPT-4的区别

我并没有实际的比较过GPT-3.5与GPT-4，但可以看到的是GPT-4新增了插件支持，支持联网，也就是GPT-4可以进行半自主学习，我在前一段时间就看到了GPT-4的一个插件，声称可以快速的帮你读论文，甚至你可以不给出论文，而是根据GPT-4联网的论文库中查找，你只需要给出名字即可。

网页版GPT-3.5是免费的，而GPT-4是收费的，月费率为$20。

另外，在API价格方面也有一定差别。这个可以看我前段时间发的tokens。

{% link About-GPT-Token,
/2023/07/07/About-GPT-Token/, https://s1.imagehub.cc/images/2023/07/08/2023-07-08-20.32.05.png %}

## 模型使用方法与主观评价

OpenAI即今为止发布了非常多的模型，今天早些的时候我每个都去使用了一下，接下来就说一说我的主观感受与使用方法和体验。

详细的使用教程在OpenAI官网有讲。

{% link API
REFERENCE, https://platform.openai.com/docs/api-reference/introduction, https://yt3.googleusercontent.com/UqT_vCkJIn1P2fH1pchr6lbe3xeEekY61h4bUpJkVuityqKOEtUYcNy3pLiJ5OKdj4uKA81FWE8=s176-c-k-c0x00ffffff-no-rj %}

### GPT

#### 使用方法

GPT是OpenAI发布的最为广为人知的模型，也是我个人认为最强大的模型。根据OpenAI官方网站的介绍，如果要使用GPT的API，最佳选择是使用Node.js和Python。我曾尝试过使用Node.js、Python、Java和Curl这几种方式，总的来说它们之间差别不大。

我通常最常使用的是`gpt-3.5-turbo`模型，不过听说这个模型将在年底下架，并被新一代的3.5模型所取代。对于这个消息，我并没有深入了解。

起初，我并不擅长进行多轮对话，感觉自己很可笑。但今天我学了一下，发现其实非常简单。

首先，让我们了解一下发送请求的组成部分（以Curl为例）：

```shell
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
     "model": "gpt-3.5-turbo",
     "messages": [{"role": "user", "content": "Say this is a test!"}],
     "temperature": 0.7
   }'
```

API接口为：https://api.openai.com/v1/chat/completions

`-H`部分包含了两个头信息参数，分别是`Content-Type`和`Authorization`，用于指定请求体的类型和授权令牌。需要在官网申请获取OPENAI_API_KEY，并将其替换为实际的密钥。

`-d`部分是请求体，根据前文中`Content-Type`为`application/json`，所以这部分需要符合JSON格式。

其中，`model`指定了要使用的模型，`messages`是对话内容，每个消息包含一个`role`角色和`content`内容，用户默认角色为`user`
。这两个参数是必需的。

响应示例：

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1677858242,
  "model": "gpt-3.5-turbo-0301",
  "usage": {
    "prompt_tokens": 13,
    "completion_tokens": 7,
    "total_tokens": 20
  },
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "\n\nThis is a test!"
      },
      "finish_reason": "stop",
      "index": 0
    }
  ]
}
```

最重要的部分是`choices`中的`message`内容，其中`role`为assistant，表示这是AI助手的回复，`content`则是AI助手的回复内容。

以上是基本的请求和响应内容。实现多轮对话也非常简单，只需将前面的对话内容全部传回即可。

```shell
curl https://api.openai.com/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
      "model": "gpt-3.5-turbo",
      "messages": [
        {
          "role": "system",
          "content": "You are a helpful assistant."
        },
        {
          "role": "user",
          "content": "What is 1+1?"
        },
        {
          "role": "assistant",
          "content": "1+1 equals 2."
        },
        {
          "role": "user",
          "content": "And if we add 3?"
        },
        {
          "role": "assistant",
          "content": "1 + 1 + 3 = 5. So, 1+1 plus 3 equals 5."
        },
        {
          "role": "user",
          "content": "What if we multiply it by 5?"
        }
      ]
    }'
```

可以看到，这里出现了多个`assistant`和`user`，还有一个`system`角色。个人而言，我将`user`视为自己，而其他角色都是AI助手。这段对话可以看作是多轮对话的示例。

通过以上方法，我们已经学会了最基本的多轮对话请求方式。

#### 评价

GPT-3.5模型是OpenAI发布的一款强大的自然语言处理模型，它具有以下几个方面的优点：

1. **语言生成能力强大**：GPT-3.5模型在语言生成任务上表现出色，能够生成流畅、准确、连贯的文本。它能够根据上下文理解用户的意图并生成相关回复，具备一定的语义理解和逻辑推理能力。

2. **多领域适用**：GPT-3.5模型在多个领域都有应用潜力，可以用于编写文章、回答问题、提供建议、创作故事等。无论是技术领域、商业应用还是创意创作，该模型都可以提供有用的信息和帮助。

3. **灵活的对话交互**：GPT-3.5模型支持多轮对话，可以进行复杂的对话交互。它能够记住先前的对话历史并根据上下文进行回复，使得对话更加连贯和一致。

4. **可定制性强**：GPT-3.5模型允许用户根据实际需求进行定制，可以通过指定不同的角色、添加系统级指示等方式来控制模型的行为。这种灵活性使得模型可以根据特定的应用场景进行定制化的使用。

尽管GPT-3.5模型具有许多优点，但也存在一些考虑因素：

1. **模型训练成本高**：由于GPT-3.5模型的复杂性，其训练和部署成本相对较高，可能对某些个人开发者或小型团队而言不太可承受。

2. **对大量数据的依赖**：GPT-3.5模型的性能与其所训练的大规模数据集密切相关。在某些领域或特定任务上，模型可能需要更多的领域专业知识或数据来取得最佳效果。

3. **缺乏常识和实际知识**：尽管GPT-3.5模型在生成文本方面表现出色，但它并没有实际的常识和背景知识。在处理需要具备实际知识或复杂推理的任务时，模型可能会出现一些不准确或不合理的回复。

> 当然这是ChatGPT自己对自己的评价。

### Image

OpenAI推出的Image模型还是满垃圾的，也不知道是我自己的问题还是什么。

> 这个Image图像生成模型并不是DALL·E模型

#### 使用方法

```shell
curl https://api.openai.com/v1/images/generations \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -d '{
    "prompt": "A cute baby sea otter",
    "n": 2,
    "size": "1024x1024"
  }'
```

`prompt`是提示词，使用过[midjourney](https://www.midjourney.com/)或者其他的图像生成模型应该满熟悉了，如何使用我不多说了

`n`生成图像的数量

`size`生成图像的大小，现在可选为`256x256`, `512x512`, `1024x1024`

#### 评价

并不推荐使用，这里推荐midjourney、DALL·E、Diffusion Model这三个应该是主流的了

以至于后来的图像编辑、图像变化的api我都不想多说了，这里就给出使用方法示例。

```shell
curl https://api.openai.com/v1/images/edits \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F image="@otter.png" \
  -F mask="@mask.png" \
  -F prompt="A cute baby sea otter wearing a beret" \
  -F n=2 \
  -F size="1024x1024"
```

```shell
curl https://api.openai.com/v1/images/variations \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -F image="@otter.png" \
  -F n=2 \
  -F size="1024x1024"
```

使用方法在官方文档中有详细参数

### Audio

语音转文字这个模型我觉得还是蛮使用的

#### 使用方法

```shell
curl https://api.openai.com/v1/audio/transcriptions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F file="@/path/to/file/audio.mp3" \
  -F model="whisper-1"
```

`-F`代表使用form-data，这就对应了上文中`Content-Type: multipart/form-data`

`file`需要上传的音频文件，支持的格式有`mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `wav`, `webm`

`model`使用的模型，目前，用于音频的模型仅有`whisper-1`

#### 评价

个人使用的次数不多，但总体感受是，蛮好用的。

## 资料分享

关于OpenAI的论文，官网已经发表了连接[在这里](https://openai.com/research)

## 最后的话

目前我还并没有去研究微调模型，不过在今后说不定会感兴趣。