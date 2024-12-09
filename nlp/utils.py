import json
import logging
import os
import platform
import shutil
from heapq import heappop
from subprocess import check_call
from typing import Tuple, List, Sequence, Dict

import fasttext
import pandas as pd
import torch
from matplotlib import pyplot as plt
from transformers import BertTokenizer, BertModel, PreTrainedModel, BatchEncoding, BertTokenizerFast


def get_static_model_root_path():
    if platform.system() == 'Windows':
        return "C:\\Users\\O772985\\OneDrive - JPMorgan Chase\\MSDE\\data\\study\\pyanalysis\\model"
    elif platform.system() == 'Linux':
        return "/home/omniai-jupyter/nlp/pyanalysis/model"
    else:
        return "/Users/honghu/MSDE/data/study/pyanalysis/model"


def get_result_save_path():
    if platform.system() == 'Windows':
        return "C:\\Users\\O772985\\OneDrive - JPMorgan Chase\\MSDE\\tmp\\"
    elif platform.system() == 'Linux':
        return "/home/omniai-jupyter/nlp/pyanalysis/data_backup"
    else:
        return "/Users/honghu/Doc/w266/Project/data_backup"


class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def __call__(self):
        return self.total / float(self.steps)


def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        json.dump(d, f, indent=4)


def save_checkpoint(state, optimizer, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        optimizer: the optimizer
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        logging.debug("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        logging.debug("Checkpoint Directory exists! ")
    if optimizer is not None:
        state['optim_dict'] = optimizer.state_dict()
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise ("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, weights_only=True)
    model.load_state_dict(checkpoint['state_dict'])

    # If your intent is to resume training, it's always a good practice to also restore the optimizer state
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


def get_local_device():
    if torch.cuda.is_available():
        return 'cuda'
    # return 'cpu'
    return 'mps' if torch.backends.mps.is_available() else 'cpu'


class StaticModelCategory:
    nlp = "nlp"
    torch = "torch"
    tensorflow = "tensorflow"


def get_static_model_path(subject: str, category: str):
    return os.path.join(get_static_model_root_path(), category, subject)


def load_fasttext_model(model_checkpoint: str = 'cc.en.300.bin') -> fasttext.FastText:
    root_path = get_static_model_path("fasttext", StaticModelCategory.nlp)
    save_path = str(os.path.join(root_path, model_checkpoint))
    return fasttext.load_model(save_path)


def get_fasttext_embeddings(model: fasttext.FastText, sentences: Sequence[Sequence[str]]) -> List[List[float]]:
    return [[model.get_word_vector(word) for word in sentence] for sentence in sentences]


class ReportConstants:
    mistakes_top_n = 50


class DataTypeConstant:
    train = "train"
    val = "val"
    test = "test"


NER_DEF_COLOR_MAP = {
    "O": "transparent",
    "B-ORG": "#ffcccc",
    "I-ORG": "#ffcccc",
    "B-PER": "#cce5ff",
    "I-PER": "#cce5ff",
    "B-LOC": "#ccffcc",
    "I-LOC": "#ccffcc"
}


def gen_ner_html_header(color_map: Dict[str, str] = None, specific_tag: str = None) -> str:
    if color_map is None:
        color_map = NER_DEF_COLOR_MAP
    html_content = ["<div style=\"font-family: 'Arial', sans-serif; line-height: 1.6;\">"]
    for i, k in enumerate(color_map.keys()):
        if i % 2 == 1:
            color = color_map.get(k)
            if specific_tag:
                if k.endswith(specific_tag.upper()):
                    html_content.append(f'<span style="background-color: {color}; padding: 4px 6px; margin: 2px; border-radius: 5px; font-weight: bold;">{specific_tag}</span> ')
            else:
                html_content.append(f'<span style="background-color: {color}; padding: 4px 6px; margin: 2px; border-radius: 5px; font-weight: bold;">{k.split("-")[1]}</span> ')
    html_content.append("</div>")
    html_content.append("<hr>\n")
    return ''.join(html_content)


def gen_ner_html_body(words: List[str], tag_list: List[str], color_map: Dict[str, str] = None) -> str:
    if color_map is None:
        color_map = NER_DEF_COLOR_MAP

    # Generate Markdown content with inline HTML
    html_content = ["<div style=\"font-family: 'Arial', sans-serif; line-height: 1.6;\">"]
    for word, tag in zip(words, tag_list):
        color = color_map.get(tag, "transparent")
        html_content.append(f'<span style="background-color: {color}; padding: 4px 6px; margin: 2px; border-radius: 5px;">{word}</span> ')
    html_content.append("</div>")
    return ''.join(html_content)


def get_resources(data_dir: str, file_name: str) -> List[str]:
    # loading tags (we require this to map tags to their indices)
    file_path = os.path.join(data_dir, file_name)
    lines = []
    with open(file_path) as f:
        line = f.readline()
        while line:
            if len(line.rstrip()) != 0:
                lines.append(line.rstrip())
            line = f.readline()
    return lines


def gen_ner_mistakes_html(model_dir: str, data_dir: str, data_type: str, mistakes: List[List[List[Tuple[float, int, List[int], List[int]]]]]) -> None:
    tags = get_resources(data_dir, "tags.txt")
    sentences = get_resources(os.path.join(data_dir, data_type), "sentences.txt")
    tag_names = ["ALL", "ORG", "PER", "LOC"]
    metrics = ["Precision", "Recall", "F1"]
    for tag_id, tag_name in enumerate(tag_names):
        if tag_name == "ALL":
            tag_map = NER_DEF_COLOR_MAP
        else:
            tag_map = NER_DEF_COLOR_MAP.copy()
            for tag in NER_DEF_COLOR_MAP.keys():
                if not tag.endswith(tag_name.upper()):
                    tag_map[tag] = "transparent"
        mistake = mistakes[tag_id]
        for metric_id, metric_name in enumerate(metrics):
            write_dir = os.path.join(model_dir, "mistakes", data_type, tag_name)
            os.makedirs(write_dir, exist_ok=True)
            file_path = os.path.join(write_dir, f"{metric_name}.html")
            pq = mistake[metric_id]
            ordered = []
            while pq:
                ordered.append(heappop(pq))
            ordered.reverse()
            with open(file_path, "w") as f:
                f.write(gen_ner_html_header(tag_map, None if tag_name == "ALL" else tag_name))
                for neg_metric_val, neg_sentence_idx, label_ids, output_ids in ordered:
                    sentence = sentences[-neg_sentence_idx]
                    label_sentence = sentence.split(' ')
                    output_sentence = sentence.split(' ')
                    labels = [tags[label_id] for label_id in label_ids]
                    outputs = [tags[output_id] for output_id in output_ids]
                    f.write(f"<h5 style=\"font-family: 'Arial', sans-serif; color: #666;\">{tag_name} {metric_name}={-neg_metric_val}</h5>")
                    f.write(gen_ner_html_body(label_sentence, labels, tag_map))
                    f.write("<br>")
                    f.write(gen_ner_html_body(output_sentence, outputs, tag_map))
                    f.write("<hr>\n")


def load_bert_torch_model(model_checkpoint: str = 'bert-base-uncased') -> Tuple[PreTrainedModel, BertTokenizerFast]:
    root_path = get_static_model_path(model_checkpoint, StaticModelCategory.nlp)
    save_path = str(os.path.join(root_path, StaticModelCategory.torch))
    if not os.path.exists(save_path):
        bert_model = BertModel.from_pretrained(model_checkpoint)
        bert_model.save_pretrained(save_path)
        bert_tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
        bert_tokenizer.save_pretrained(save_path)
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    bert_model = BertModel.from_pretrained(save_path)
    bert_tokenizer = BertTokenizerFast.from_pretrained(save_path)
    return bert_model, bert_tokenizer


def tokenize_and_align_labels(
        sentences_tokens: List[List[str]], tags: List[List[str]], tokenizer: BertTokenizer, is_label_1st_token: bool
) -> Tuple[BatchEncoding, List[list]]:
    tokenized_inputs = tokenizer(sentences_tokens, truncation=True, is_split_into_words=True)
    labels = []
    for i, labels_per_sen in enumerate(tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(labels_per_sen[word_idx])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(-100 if is_label_1st_token else labels_per_sen[word_idx])
            previous_word_idx = word_idx

        labels.append(label_ids)
    return tokenized_inputs, labels


def launch_job(dataset: str, embedding: str, model: str, lr: float, n_head: int, n_layer: int) -> str:
    root_dir = get_result_save_path()
    with open("params.json", "r") as f:
        params = json.load(f)
    dir_name = "train"
    if "pre_trained_tar" in params and params["pre_trained_tar"] is not None:
        dir_name = "transfer"
    if model == "transformer":
        comb_dir = f"{embedding}_{model}_{lr}_{n_head}x{n_layer}"
    else:
        comb_dir = f"{embedding}_{model}_{lr}"
    model_dir = os.path.join(root_dir, dir_name, dataset, comb_dir)

    # # Create a new folder in parent_dir with unique_name "job_name"
    # if not os.path.exists(model_dir):
    #     os.makedirs(model_dir)
    #
    # # Write parameters in json file
    # json_path = os.path.join(model_dir, 'params.json')
    # with open(json_path, "w") as f:
    #     json.dump(params, f)

    cmd = f"./run.sh {dataset} {embedding} {model} {lr} {n_head} {n_layer}"
    logging.info(cmd)
    check_call(cmd, shell=True)
    return model_dir


def plot_train_loss_f1(model_dir: str, model_name: str) -> None:
    image_dir = os.path.join(model_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    history_path = os.path.join(model_dir, "train_metrics_history.json")
    with open(history_path, "r") as f:
        history = json.load(f)

    # Placeholder lists for extracting data
    train_loss = []
    val_loss = []
    train_f1 = []
    val_f1 = []

    # Extract the data from the dictionary
    for epoch in history:
        train_loss.append(epoch['train']['loss'])
        val_loss.append(epoch['val']['loss'])
        train_f1.append(epoch['train']['entity_classification']['overall_f1'])
        val_f1.append(epoch['val']['entity_classification']['overall_f1'])

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plotting Loss
    axes[0].plot(train_loss, label='Train Loss', marker='o')
    axes[0].plot(val_loss, label='Validation Loss', marker='o')
    axes[0].set_title(f'{model_name} Loss Change')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    # Plotting F1 Score
    axes[1].plot(train_f1, label='Train Entity F1', marker='o')
    axes[1].plot(val_f1, label='Validation Entity F1', marker='o')
    axes[1].set_title(f'{model_name} Entity F1 Change')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('F1 Score')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(image_dir, "train_loss_f1.png"))
    plt.show()
    plt.close()


def metric_table(model_dir: str, model_name: str) -> pd.DataFrame:
    metric_path = os.path.join(model_dir, "metrics_test_best.json")
    with open(metric_path, "r") as f:
        metric = json.load(f)
        table = pd.DataFrame(
            metric["entity_classification"]["overall"],
            index=["ALL", "ORG", "PER", "LOC"],
            columns=["TP", "FP", "FN", "Support", "Precision", "Recall", "F1"]
        )
        table.reset_index(inplace=True)
        table.rename(columns={'index': 'Tag'}, inplace=True)  # Rename the index column to "Tag"
        table.insert(0, "Model", model_name)
        return table


def gen_top_n_mistakes(model_dir: str, model_name: str, top_n: int = 3, tag: str = "ALL", metric="F1") -> str:
    model_path = os.path.join(model_dir, "mistakes", "test", tag)
    file_name = f"{metric}.html"
    lines = get_resources(model_path, file_name)
    res = [
        f"<h3 style=\"font-family: 'Arial', sans-serif; color: #444; border-bottom: 2px solid #eaeaea; padding-bottom: 5px; margin-bottom: 15px;\">Top {top_n} Lowest {metric} on {tag} for {model_name}</h3>"]
    # res = []
    res.extend(lines[0: min(len(lines), top_n) + 1])
    return "".join(res)


if __name__ == '__main__':
    md = "/Users/honghu/Doc/w266/Project/data_backup/train/small/fasttext_lstm_0.01"
    mn = "LSTM"
    plot_train_loss_f1(md, mn)
    df = metric_table(md, mn)
    print(df)
    html = gen_top_n_mistakes(md, mn)
    print(f'{html}')
