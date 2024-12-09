"""Defines the neural network, losss function and metrics"""
import logging
import os
from heapq import heappush, heappop
from typing import Set, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertForTokenClassification

from utils import load_bert_torch_model, ReportConstants, get_static_model_path, StaticModelCategory


class Net(nn.Module):
    """
    This is the standard way to define your own network in PyTorch. You typically choose the components
    (e.g. LSTMs, linear layers etc.) of your network in the __init__ function. You then apply these layers
    on the input step-by-step in the forward function. You can use torch.nn.functional to apply functions
    such as F.relu, F.sigmoid, F.softmax. Be careful to ensure your dimensions are correct after each step.

    You are encouraged to have a look at the network in pytorch/vision/model/net.py to get a better sense of how
    you can go about defining your own network.

    The documentation for all the various components available to you is here: http://pytorch.org/docs/master/nn.html
    """

    def __init__(self, params):
        """
        We define an recurrent network that predicts the NER tags for each token in the sentence. The components
        required are:

        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags

        Args:
            params: (Params) contains vocab_size, embedding_dim, lstm_hidden_dim
        """
        super(Net, self).__init__()
        self.params = params
        self.tokenizer = None
        if params.embedding == "fasttext":
            self.embedding = lambda tokens: tokens
        elif params.embedding == "vocab":
            # the embedding takes as input the vocab_size and the embedding_dim
            self.embedding = nn.Embedding(params.vocab_size, params.embedding_dim).to(params.device)
        else:  # bert
            bert_model, self.tokenizer = load_bert_torch_model(f"bert-base-{params.embedding}")
            self.embedding = bert_model.embeddings

        if params.model == "lstm":
            # Define the LSTM
            self.model = nn.LSTM(params.embedding_dim, params.lstm_hidden_dim, batch_first=True).to(params.device)

            # the fully connected layer transforms the output to give the final output layer
            self.fc = nn.Linear(params.lstm_hidden_dim, params.number_of_tags).to(params.device)
        elif params.model == "bert":
            model_checkpoint = f"bert-base-{params.embedding}"
            root_path = get_static_model_path(model_checkpoint, StaticModelCategory.nlp)
            save_path = str(os.path.join(root_path, StaticModelCategory.torch))
            self.model = BertForTokenClassification.from_pretrained(
                save_path,
                num_labels=params.number_of_tags
            )
        else:
            # Define a transformer layer
            self.model = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=params.embedding_dim, nhead=params.nhead, batch_first=True),
                num_layers=params.num_layers
            ).to(params.device)
            logging.debug(f'transformer config: nhead={params.nhead}, num_layers={params.num_layers}')
            self.fc = nn.Linear(params.embedding_dim, params.number_of_tags).to(params.device)

    def forward(self, s):
        """
        This function defines how we use the components of our network to operate on an input batch.

        Args:
            s: (Variable) contains a batch of sentences, of dimension batch_size x seq_len, where seq_len is
               the length of the longest sentence in the batch. For sentences shorter than seq_len, the remaining
               tokens are PADding tokens. Each row is a sentence with each element corresponding to the index of
               the token in the vocab.

        Returns:
            out: (Variable) dimension batch_size*seq_len x num_tags with the log probabilities of tokens for each token
                 of each sentence.

        Note: the dimensions after each step are provided
        """
        #                                -> batch_size x seq_len
        # apply the embedding layer that maps each token to its embedding
        # dim: batch_size x seq_len x embedding_dim
        if self.params.model == "bert":
            return self.model(**s)

        s = self.embedding(s)
        # run the LSTM along the sentences of length seq_len
        # dim: batch_size x seq_len x lstm_hidden_dim
        if self.params.model == "lstm":
            s, _ = self.model(s)
        else:
            s = self.model(s)

        # make the Variable contiguous in memory (a PyTorch artefact)
        s = s.contiguous()

        # reshape the Variable so that each row contains one token
        # dim: batch_size*seq_len x lstm_hidden_dim
        s = s.view(-1, s.shape[2])

        # apply the fully connected layer and obtain the output (before softmax) for each token
        s = self.fc(s)  # dim: batch_size*seq_len x num_tags

        # apply log softmax on each token's output (this is recommended over applying softmax
        # since it is numerically more stable)
        return F.log_softmax(s, dim=1)  # dim: batch_size*seq_len x num_tags


def loss_fn(outputs, labels, idxes):
    """
    Compute the cross entropy loss given outputs from the model and labels for all tokens. Exclude loss terms
    for PADding tokens.

    Args:
        outputs: (Variable) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (Variable) dimension batch_size x seq_len where each element is either a label in [0, 1, ... num_tag-1],
                or -1 in case it is a PADding token.

    Returns:
        loss: (Variable) cross entropy loss for all tokens in the batch

    Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
          demonstrates how you can easily define a custom loss function.
    """

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.view(-1)

    # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0).float()

    # indexing with negative values is not supported. Since PADded tokens have label -1, we convert them to a positive
    # number. This does not affect training, since we ignore the PADded tokens with the mask.
    labels = labels % outputs.shape[1]

    num_tokens = int(torch.sum(mask))

    # compute cross entropy loss for all tokens (except PADding tokens), by multiplying with mask.
    return -torch.sum(outputs[range(outputs.shape[0]), labels] * mask) / num_tokens


def accuracy(outputs, labels, idxes) -> float:
    """
    Compute the accuracy, given the outputs and labels for all tokens. Exclude PADding terms.

    Args:
        outputs: (np.ndarray) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (np.ndarray) dimension batch_size x seq_len where each element is either a label in
                [0, 1, ... num_tag-1], or -1 in case it is a PADding token.

    Returns: (float) accuracy in [0,1]
    """
    entity_classification(outputs, labels, idxes)

    # reshape labels to give a flat vector of length batch_size*seq_len
    labels = labels.ravel()

    # since PADding tokens have label -1, we can generate a mask to exclude the loss from those terms
    mask = (labels >= 0)

    # np.argmax gives us the class predicted for each token by the model
    outputs = np.argmax(outputs, axis=1)

    # compare outputs with labels and divide by number of tokens (excluding PADding tokens)
    return np.sum(outputs == labels) / float(np.sum(mask))


def entity_classification(outputs, labels, idxes) -> dict:
    """
    Compute the accuracy, given the outputs and labels for all tokens. Exclude PADding terms.

    Args:
        outputs: (np.ndarray) dimension batch_size*seq_len x num_tags - log softmax output of the model
        labels: (np.ndarray) dimension batch_size x seq_len where each element is either a label in
                [0, 1, ... num_tag-1], or -1 in case it is a PADding token.

    Returns:
    """
    assert outputs.shape[-1] % 2 == 1
    num_tags_plus = (outputs.shape[-1] + 1) // 2
    batch_size, seq_len = labels.shape
    outputs = np.argmax(outputs, axis=1)
    outputs = outputs.reshape(batch_size, seq_len)

    overall = np.zeros((num_tags_plus, 4), dtype=int)
    mistakes_tag_pqs = [[[] for _ in range(3)] for _ in range(num_tags_plus)]

    for i in range(batch_size):  # for each sentence
        masks = labels[i] >= 0
        raw_sentence_idx = idxes[i]
        output = outputs[i][masks]
        label = labels[i][masks]
        pred_set = generate_ner_set(output)
        true_set = generate_ner_set(label)

        tp_set = pred_set & true_set
        fp_set = pred_set - true_set
        fn_set = true_set - pred_set

        delta = np.zeros((num_tags_plus, 4), dtype=int)
        for metric_idx, metric_set in enumerate([tp_set, fp_set, fn_set]):
            for t in metric_set:  # t=(tag, beg_idx, end_idx)
                delta[t[0] // 2 + 1][metric_idx] += 1  # each tag
                delta[0][metric_idx] += 1  # overall
        for t in true_set:  # support
            delta[t[0] // 2 + 1][-1] += 1  # each tag
            delta[0][-1] += 1  # overall
        overall += delta
        for tag_idx, (tp, fp, fn, support) in enumerate(delta):
            if tp + fp + fn == 0:
                continue  # all "O" case
            t = calculate_metrics(tp, fp, fn)
            pqs = mistakes_tag_pqs[tag_idx]
            for idx, val in enumerate(t):
                if val != 1:
                    pq = pqs[idx]
                    heappush(pq, (-val, -raw_sentence_idx, label.tolist(), output.tolist()))
                    if len(pq) > ReportConstants.mistakes_top_n:
                        heappop(pq)

    return {
        "overall": overall,
        "mistakes": mistakes_tag_pqs
    }


def agg_entity_classification(summ: List[dict]) -> dict:
    overall = np.zeros_like(summ[0]["entity_classification"]["overall"], dtype=int)
    num_tags_plus = len(summ[0]["entity_classification"]["mistakes"])
    agg_mistakes = [[[] for _ in range(3)] for _ in range(num_tags_plus)]
    for epoch in summ:
        overall += epoch["entity_classification"]["overall"]
        mistake = epoch["entity_classification"]["mistakes"]
        for i in range(num_tags_plus):
            for j in range(3):
                realtime_pq = agg_mistakes[i][j]
                for t in mistake[i][j]:
                    heappush(realtime_pq, t)
                    if len(realtime_pq) > ReportConstants.mistakes_top_n:
                        heappop(realtime_pq)

    calculated = calc_classification_metrics(overall)
    table = calculated.tolist()

    return {
        "overall_f1": table[0][-1],
        "overall": table,
        "mistakes": agg_mistakes
    }


def calc_classification_metrics(data: np.ndarray) -> np.ndarray:
    # Initialize arrays for precision, recall, and F1
    precision = np.zeros(data.shape[0])
    recall = np.zeros(data.shape[0])
    f1 = np.zeros(data.shape[0])

    # Compute precision, recall, and F1
    for i in range(data.shape[0]):
        TP = data[i, 0]
        FP = data[i, 1]
        FN = data[i, 2]

        precision[i], recall[i], f1[i] = calculate_metrics(TP, FP, FN)

    # Append results to the original matrix
    result = np.hstack((data, precision.reshape(-1, 1), recall.reshape(-1, 1), f1.reshape(-1, 1)))
    return result


def calculate_metrics(TP, FP, FN) -> Tuple[float, float, float]:
    if TP + FP > 0:
        precision = TP / (TP + FP)
    else:
        precision = 0  # Handle division by zero

    if TP + FN > 0:
        recall = TP / (TP + FN)
    else:
        recall = 0  # Handle division by zero

    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0  # Handle division by zero

    return precision, recall, f1


def generate_ner_set(labels: np.array) -> Set[tuple]:
    n = len(labels)
    cur = [0, -1, -1]
    cur_idx = 0
    gather = []
    while cur_idx < n:
        label = labels[cur_idx]
        if label != 0:
            if label % 2 == 1:
                cur = [label, cur_idx, cur_idx]
                gather.append(cur)
            elif label - 1 == cur[0]:
                cur[-1] = cur_idx
            else:
                cur = [0, -1, -1]
        else:
            cur = [0, -1, -1]
        cur_idx += 1
    return set(tuple(a) for a in gather)


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'accuracy': accuracy,
    'entity_classification': entity_classification
    # could add more metrics such as accuracy for each token type
}
