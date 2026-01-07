## Introduction
Prompt Tuning 在实验上通常可以提高 LLM 的表现。 但是为什么 performance 会提升？ 定性上来说，LLM prompt 被优化得更加全面，包含例子等等。
但是从量化的角度来分析，我们想要去了解：优化后的 prompt 在哪些方面有比较大的改变？ 

## Method & Experiment

我用了 TextGrad 和 DSpy 两个框架。他们两个都有对 prompt 多次迭代，验证，然后取最优 作为 final optimized prompt 这么一个过程。我们在 10 个通用数据集上进行 TextGrad 和 DSpy 的 prompt tuning，并且收集整个训练过程中所有的 draft prompt，即使他们没有被最终采用，但是他们参与了迭代中的验证。

我们随后对所有的 draft prompts 用 另外一个 LLM 做 特征提取， 用于指导 LLM 做特征提取的 prompt 在 './dynamic_feats.json'。 之所以把它命名为 dynamic features 是因为用 LLM 提取特征的时候， 由于 LLM 本身的不确定性，导致多次运行 获得的 features 会有不同。

在获取 dynamic features 之后，我再用 ATE 算法去对每个数据集计算，不同 features 对于解决当前类问题有什么影响。

## Codebase Structure

To run DSpy experiments. 

```
chmod +x train_dspy_miprov2.sh
./train_dspy_miprov2.sh
```

To run TextGrad experiments. 

```
chmod +x train_textgrad.sh
./train_textgrad.sh
```
