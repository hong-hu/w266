import os
import random
from typing import List

import numpy as np
import torch
from torch.autograd import Variable
from transformers import BatchEncoding

import utils


class DataLoader(object):
    """
    Handles all aspects of the data. Stores the dataset_params, vocabulary and tags with their mappings to indices.
    """

    def __init__(self, data_dir, params):
        """
        Loads dataset_params, vocabulary and tags. Ensure you have run `build_vocab.py` on data_dir before using this
        class.

        Args:
            data_dir: (string) directory containing the dataset
            params: (Params) hyperparameters of the training process. This function modifies params and appends
                    dataset_params (such as vocab size, num_of_tags etc.) to params.
        """

        # loading dataset_params
        json_path = os.path.join(data_dir, 'dataset_params.json')
        assert os.path.isfile(json_path), "No json file found at {}, run build_vocab.py".format(json_path)
        self.dataset_params = utils.Params(json_path)
        self.params = params
        # {
        #     "train_size": 10,
        #     "dev_size": 10,
        #     "test_size": 10,
        #     "vocab_size": 368,
        #     "number_of_tags": 9,
        #     "pad_word": "<pad>",
        #     "pad_tag": "O",
        #     "unk_word": "<unk>"
        # }

        # setting the indices for UNKnown words and PADding symbols
        self.bert_tokenizer = None
        if params.embedding == "fasttext":
            self.tokenize = None
            self.pad_ind = self.dataset_params.pad_word
            self.fasttext_model = utils.load_fasttext_model()
        elif params.embedding != "vocab":  # uncased
            assert params.model == "bert", f"{params.embedding} not supported for {params.model}"
            _, self.bert_tokenizer = utils.load_bert_torch_model(f"bert-base-{params.embedding}")

            def tokenize(sentence: str) -> List[int]:
                return self.bert_tokenizer(sentence, return_tensors='pt', is_split_into_words=True)

            self.tokenize = tokenize
            self.pad_ind = self.bert_tokenizer.pad_token_id
        else:
            # loading vocab (we require this to map words to their indices)
            vocab_path = os.path.join(data_dir, 'words.txt')
            self.vocab = {}
            with open(vocab_path) as f:
                for i, l in enumerate(f.read().splitlines()):
                    self.vocab[l] = i
            unk_ind = self.vocab[self.dataset_params.unk_word]
            self.pad_ind = self.vocab[self.dataset_params.pad_word]

            def tokenize(sentence: str) -> List[int]:
                return [self.vocab[token] if token in self.vocab else unk_ind for token in sentence.split(' ')]

            self.tokenize = tokenize

        # loading tags (we require this to map tags to their indices)
        tags_path = os.path.join(data_dir, 'tags.txt')
        self.tag_map = {}
        with open(tags_path) as f:
            for i, t in enumerate(f.read().splitlines()):
                self.tag_map[t] = i

        # adding dataset parameters to param (e.g. vocab size, )
        params.update(json_path)

    def load_sentences_labels(self, sentences_file, labels_file, d, is_over_sampling=False):
        """
        Loads sentences and labels from their corresponding files. Maps tokens and tags to their indices and stores
        them in the provided dict d.

        Args:
            sentences_file: (string) file with sentences with tokens space-separated
            labels_file: (string) file with NER tags for the sentences in labels_file
            d: (dict) a dictionary in which the loaded data is stored
        """

        token_ids_for_sentences = []
        labels = []

        if self.params.embedding == "fasttext":
            with open(sentences_file) as f:
                for sentence in f.read().splitlines():
                    token_ids_for_sentences.append(sentence.split(' '))
        elif self.params.model == "bert":
            with open(sentences_file) as f:
                for sentence in f.read().splitlines():
                    token_ids_for_sentences.append(sentence.split(' '))
        else:
            with open(sentences_file) as f:
                for sentence in f.read().splitlines():
                    # replace each token by its index if it is in vocab
                    # else use index of UNK_WORD
                    token_ids_for_sentences.append(self.tokenize(sentence))

        with open(labels_file) as f:
            for labels_per_sentence in f.read().splitlines():
                # replace each label by its index
                l = [self.tag_map[label] for label in labels_per_sentence.split(' ')]
                labels.append(l)

        line_nums = [i for i in range(len(labels))]
        if is_over_sampling:
            tags = len(self.tag_map) // 2
            cand_indexes = [[] for _ in range(tags)]
            counts = [0] * tags
            for idx, label_tag_ids in enumerate(labels):
                delta = [0] * tags
                for tag_id in label_tag_ids:
                    if tag_id and tag_id % 2 == 1:
                        delta[tag_id // 2] = 1
                if sum(delta) == 1:  # find idx has only one tag
                    cand_indexes[delta.index(1)].append(idx)
                for i in range(tags):
                    counts[i] += delta[i]
            max_freq = max(counts)
            new_indexes = []
            for tag_idx, freq in enumerate(counts):
                gap = max_freq - freq
                cands = cand_indexes[tag_idx]
                times = gap // len(cands)
                mod = gap % len(cands)
                new_indexes.extend(cands * times)
                new_indexes.extend(cands[0: mod])
            new_sentences = [token_ids_for_sentences[idx] for idx in new_indexes]
            new_labels = [labels[idx] for idx in new_indexes]
            token_ids_for_sentences.extend(new_sentences)
            labels.extend(new_labels)
            line_nums.extend(new_indexes)

        if self.params.model == "bert":
            token_ids_for_sentences, labels = utils.tokenize_and_align_labels(token_ids_for_sentences, labels, self.bert_tokenizer, True)
            for i in range(len(labels)):
                assert len(labels[i]) == len(token_ids_for_sentences["input_ids"][i])
            d['size'] = len(token_ids_for_sentences["input_ids"])
        else:
            # checks to ensure there is a tag for each token
            assert len(labels) == len(token_ids_for_sentences) == len(line_nums)
            for i in range(len(labels)):
                assert len(labels[i]) == len(token_ids_for_sentences[i])
            d['size'] = len(token_ids_for_sentences)

        # storing sentences and labels in dict d
        d['data'] = token_ids_for_sentences
        d['labels'] = labels
        d['line_num'] = line_nums

    def load_data(self, types, data_dir):
        """
        Loads the data for each type in types from data_dir.

        Args:
            types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
            data_dir: (string) directory containing the dataset

        Returns:
            data: (dict) contains the data with labels for each type in types

        """
        data = {}

        for split in ['train', 'val', 'test']:
            if split in types:
                sentences_file = os.path.join(data_dir, split, "sentences.txt")
                labels_file = os.path.join(data_dir, split, "labels.txt")
                data[split] = {}
                self.load_sentences_labels(sentences_file, labels_file, data[split], split == "train")

        return data

    def data_iterator(self, data, params, shuffle=False):
        """
        Returns a generator that yields batches data with labels. Batch size is params.batch_size. Expires after one
        pass over the data.

        Args:
            data: (dict) contains data which has keys 'data', 'labels' and 'size'
            params: (Params) hyperparameters of the training process.
            shuffle: (bool) whether the data should be shuffled

        Yields:
            batch_data: (Variable) dimension batch_size x seq_len with the sentence data
            batch_labels: (Variable) dimension batch_size x seq_len with the corresponding labels

        """

        # make a list that decides the order in which we go over the data- this avoids explicit shuffling of data
        order = list(range(data['size']))
        if shuffle:
            random.seed(230)
            random.shuffle(order)

        # one pass over data
        total_size = data['size']
        line_nums = data['line_num']
        rounds = total_size // params.batch_size + int(total_size % params.batch_size != 0)
        if params.embedding == "fasttext":
            for i in range(rounds):
                # fetch sentences and tags
                end_idx_ex = min(total_size, (i + 1) * params.batch_size)
                batch_sentences = [data['data'][idx] for idx in order[i * params.batch_size: end_idx_ex]]
                batch_tags = [data['labels'][idx] for idx in order[i * params.batch_size: end_idx_ex]]

                # compute length of longest sentence in batch
                batch_max_len = max([len(s) for s in batch_sentences])
                batch_max_len = min(512, batch_max_len)

                batch_data = [[self.pad_ind] * batch_max_len for _ in range(len(batch_sentences))]
                batch_labels = -1 * np.ones((len(batch_sentences), batch_max_len))

                # copy the data to the numpy array
                for j in range(len(batch_sentences)):
                    cur_len = len(batch_sentences[j])
                    batch_data[j][:cur_len] = batch_sentences[j]
                    batch_labels[j][:cur_len] = batch_tags[j]

                # since all data are indices, we convert them to torch LongTensors
                batch_data, batch_labels = torch.tensor(np.array(utils.get_fasttext_embeddings(self.fasttext_model, batch_data))), torch.LongTensor(batch_labels)

                # shift tensors to GPU if available
                batch_data, batch_labels = batch_data.to(device=params.device), batch_labels.to(device=params.device)

                # convert them to Variables to record operations in the computational graph
                batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)

                yield batch_data, batch_labels, [line_nums[idx] for idx in order[i * params.batch_size: end_idx_ex]]
        if params.model == "bert":
            for i in range(rounds):
                # fetch sentences and tags
                end_idx_ex = min(total_size, (i + 1) * params.batch_size)
                encoding = data['data']
                input_ids = encoding['input_ids']
                attention_mask = encoding['attention_mask']

                input_ids = [input_ids[idx] for idx in order[i * params.batch_size: end_idx_ex]]
                attention_mask = [attention_mask[idx] for idx in order[i * params.batch_size: end_idx_ex]]
                batch_tags = [data['labels'][idx] for idx in order[i * params.batch_size: end_idx_ex]]

                # compute length of longest sentence in batch
                batch_max_len = max([len(s) for s in data['labels']])
                batch_max_len = min(512, batch_max_len)
                batch_size = len(batch_tags)

                # Prepare labels
                batch_data = torch.full((batch_size, batch_max_len), fill_value=self.pad_ind)  # Use fill_value for padding
                batch_attention_mask = torch.zeros((batch_size, batch_max_len), dtype=torch.long)  # Create with zeros
                batch_labels = torch.full((batch_size, batch_max_len), fill_value=-100, dtype=torch.long)  # Create with -1 for labels
                for j in range(len(batch_tags)):
                    cur_len = len(batch_tags[j])
                    batch_labels[j][:cur_len] = torch.tensor(batch_tags[j], dtype=torch.long)
                    batch_attention_mask[j][:cur_len] = torch.tensor(attention_mask[j], dtype=torch.long)
                    batch_data[j][:cur_len] = torch.tensor(input_ids[j], dtype=torch.long)

                # Move tensors to GPU if available
                batch_data, batch_attention_mask, batch_labels = batch_data.to(device=params.device), batch_attention_mask.to(device=params.device), batch_labels.to(device=params.device)

                # Use Variables if needed (usually, autograd handles this now)
                batch_data, batch_attention_mask, batch_labels = Variable(batch_data), Variable(batch_attention_mask), Variable(batch_labels)

                yield batch_data, batch_attention_mask, batch_labels, [line_nums[idx] for idx in order[i * params.batch_size: end_idx_ex]]
        else:
            for i in range(rounds):
                # fetch sentences and tags
                end_idx_ex = min(total_size, (i + 1) * params.batch_size)
                batch_sentences = [data['data'][idx] for idx in order[i * params.batch_size: end_idx_ex]]
                batch_tags = [data['labels'][idx] for idx in order[i * params.batch_size: end_idx_ex]]

                # compute length of longest sentence in batch
                batch_max_len = max([len(s) for s in batch_sentences])
                batch_max_len = min(512, batch_max_len)

                # prepare a numpy array with the data, initialising the data with pad_ind and all labels with -1
                # initialising labels to -1 differentiates tokens with tags from PADding tokens
                batch_data = self.pad_ind * np.ones((len(batch_sentences), batch_max_len))
                batch_labels = -1 * np.ones((len(batch_sentences), batch_max_len))

                # copy the data to the numpy array
                for j in range(len(batch_sentences)):
                    cur_len = len(batch_sentences[j])
                    batch_data[j][:cur_len] = batch_sentences[j]
                    batch_labels[j][:cur_len] = batch_tags[j]

                # since all data are indices, we convert them to torch LongTensors
                batch_data, batch_labels = torch.LongTensor(batch_data), torch.LongTensor(batch_labels)

                # shift tensors to GPU if available
                batch_data, batch_labels = batch_data.to(device=params.device), batch_labels.to(device=params.device)

                # convert them to Variables to record operations in the computational graph
                batch_data, batch_labels = Variable(batch_data), Variable(batch_labels)

                yield batch_data, batch_labels, [line_nums[idx] for idx in order[i * params.batch_size: end_idx_ex]]
