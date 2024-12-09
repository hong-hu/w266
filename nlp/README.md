# Efficient Financial Named Entity Recognition Without Large Language Models

## One Word

I recommend you just directly read run.ipynb because setting up environment is non-trivial.

## Requirements

I recommend using python3 and a virtual env. 

```
virtualenv -p python3 .env
source .env/bin/activate
pip install -r requirements.txt
```

When you're done working on the project, deactivate the virtual environment with `deactivate`.

## Task

Given a sentence, give a tag to each word ([Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition))

```
John   lives in New   York
B-PER  O     O  B-LOC I-LOC
```

## Dataset

There are two datasets. Both are provided by [The Fin AI](https://thefin.ai). I have unified the data format for the two datasets. I have also made minor changes to split very long sentences into multiple small sentences so that it will not exceed the max sequence length of 512 for no matter Pytorch, Bert or other encoder frameworks. 

[FiNER-ORD: Financial Named Entity Recognition Open Research Dataset](https://arxiv.org/abs/2302.11157). The original dataset comes from [Hugging Face](https://huggingface.co/datasets/TheFinAI/flare-finer-ord?row=0).

