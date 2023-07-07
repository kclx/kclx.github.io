---
title: 关于ChatGPT的tokens收费标准
tags: [ 'gpt' ]
<<<<<<< HEAD
categories: [ 'Pointless small talk' ]
=======
categories: [ 'Pointless small talk category' ]
>>>>>>> origin/main
top: false
comments: true
lang: en
toc: true
excerpt: 这篇文章将会详细的讲解chatGPT各个模型的收费标准以及使用的真实体验
swiper: false
swiperDesc: 这篇文章将会详细的讲解chatGPT各个模型的收费标准以及使用的真实体验
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
date: 2023-07-07 15:14:06
updated: 2023-07-07 15:14:06
swiperImg:
bgImg:
img: https://i.postimg.cc/8c8k0RM9/2023-07-07-17-13-09.png
---

# 关于ChatGPT的tokens收费标准

根据OpenAI官网给出的[计费标准](https://openai.com/pricing)

总结了以下表格：

| 模型            | 最大长度      | 输入（/1k tokens） | 输出（1k tokens） | 使用价格              | 类型   |
|---------------|-----------|----------------|---------------|-------------------|------|
| GPT-4         | 8K        | $0.03          | $0.06         |                   | 语言模型 |
| GPT-4         | 32K       | $0.06          | $0.12         |                   | 语言模型 |
| GPT-3.5 Turbo | 4K        | $0.0015        | $0.002        |                   | 语言模型 |
| GPT-3.5 Turbo | 16K       | $0.003         | $0.004        |                   | 语言模型 |
| Ada           |           | $0.0004        | $0.0016       |                   | 微调模型 |
| Babbage       |           | $0.0006        | $0.0024       |                   | 微调模型 |
| Curie         |           | $0.0030        | $0.012        |                   | 微调模型 |
| Davinci       |           | $0.0300        | $0.12         |                   | 微调模型 |
| Ada v2        | Ada v2    |                |               | $0.0001/1K tokens | 嵌入模型 |
| Image models  | 1024×1024 |                |               | $0.020/image      | 图像模型 |
| Image models  | 512×512   |                |               | $0.018/image      | 图像模型 |
| Image models  | 256×256   |                |               | $0.016/image      | 图像模型 |
| Whisper       |           |                |               | $0.006/min        | 音频模型 |

> 事实上，OpenAI推出的模型不只这些，通过api可以得到最完整的模型列表。
> ```shell
> curl https://api.openai.com/v1/models \
> -H "Authorization: Bearer $OPENAI_API_KEY"
> ```
> 如果没有key或者不想发送请求可以直接获取下面的json数据都是一样的
> 
> 截止2023年7月7日全部模型的json数据：
> {% getFiles citation, openai_models json, %}

## Tokens

上述内容中多次提到了tokens的概念，什么是tokens？

> 您可以将标记视为用于自然语言处理的单词片段。对于英文文本，1 个标记大约为 4 个字符或 0.75 个单词。作为参考，莎士比亚的作品集大约有
> 90 万字或 120 万个符号。
>
> `1000`tokens约等于`750`个单词
>
> 要了解有关代币如何工作并估计您的使用情况的更多信息……
>
> - 使用我们的交互式 [分词器工具](https://platform.openai.com/tokenizer)进行实验。
> - 登录您的帐户并在 Playground 中输入文本。页脚中的计数器将显示文本中有多少个标记。

根据OpenAI官方给出的回答还是无法有一个具体的掌握。

转换为中文，tokens的大小为多少？

使用gpt-3.5-turbo-0301为例：

| Request content                                                                                                                                                                                                                                                                                                                                                                                                               | Response content                                                                                                                                                                                                               | input tokens | output tokens | total |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|---------------|-------|
| Say this is a test!                                                                                                                                                                                                                                                                                                                                                                                                           | This is a test!                                                                                                                                                                                                                | 13           | 5             | 18    |
| 请形容一下小猫                                                                                                                                                                                                                                                                                                                                                                                                                       | 小猫是一种可爱的小动物，通常体型娇小，毛发柔软。它们的眼睛大而明亮，透露出一种天真无邪的神采。小猫的耳朵尖尖，灵敏而警觉，时刻捕捉着周围的声音。它们的小鼻子粉嫩可爱，鼻尖微微动着，好奇地探索着周围的世界。小猫的爪子尖锐而灵活，能够轻松攀爬和抓捕猎物。它们的尾巴修长而灵动，时而垂下，时而竖起，表达出各种情绪。小猫的身体柔软而灵活，它们经常会摆出各种可爱的姿势，让人看了心情愉悦。总之，小猫是一种令人陶醉的可爱生物，它们的存在让人感到温暖和幸福。 | 16           | 319           | 335   |
| 请翻译：This technical report presents GPT-4, a large multimodal model capable of processing image and text inputs and producing text outputs. Such models are an important area of study as they have the potential to be used in a wide range of applications, such as dialogue systems, text summarization, and machine translation. As such, they have been the subject of substantial interest and progress in recent years. | 这份技术报告介绍了GPT-4，一个能够处理图像和文本输入并生成文本输出的大型多模态模型。这样的模型是研究的重要领域，因为它们有潜力在各种应用中使用，如对话系统、文本摘要和机器翻译。因此，近年来它们受到了广泛关注和进展的研究对象。                                                                                                             | 93           | 127           | 220   |

经过计算得出，1字约等于1token，英文状态4个字符约等于1token。

> 假设：我现在有一篇论文要写，需求（input content）输入了100字，需要生成一篇5000字的论文（output
> content），算上中间的润色和修改（约等于重复3遍），那么可以计算写一篇论文大约会消耗tokens = ((100 + 5000) * 5) = 25500
>
> 使用GPT-3.5，费用为 0.03825美元 约等于 0.277人民币，
> 如果内容总长度超过4096k，那么价格会翻一倍，费用为 0.0765美元 约等于 0.5534人民币（汇率为1美元兑7.2337人民币）
>
> 使用GPT-4，费用为 0.765美元 约等于 5.534人民币
>
> 当然这只是猜测。

## 测试

使用gpt-3.5-turbo-16k与gpt-3.5-turbo-4k（并未写完一篇论文）：：

{% getFiles citation, html, %}

整理好的论文（字数：7479）：

{% getFiles citation, 习近平新时代 pdf, %}

根据最后的返回`Total tokens spent: 52087`可以得出本次论文一共花费了`52087/1000*0.003*7.2337=1.1303451957¥`花费1.13人民币。

## 总结与思考

为什么会与预测的花费不一致？

1. 示例中生成的文章为7000+
2. 采用的是gpt对话的形式，也就是说每一次对话都需要把本次对话中的上文全部当作前置传入，这样就增加了tokens的消耗
3. 输入与输出的tokens的价格并不相等，输出的tokens更贵一些