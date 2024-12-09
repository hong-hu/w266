"""Train the model"""

import argparse
import json
import logging
import os
import shutil
from timeit import default_timer
from typing import List, Tuple

import numpy as np
import torch
import torch.optim as optim
from tqdm import trange

import model.net as net
import utils
from evaluate import evaluate
from model.data_loader import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', default='data/small',
                    help="Directory containing the dataset")

result_save_dir = utils.get_result_save_path()
DEBUG_DIR = f"{result_save_dir}/test"
os.makedirs(DEBUG_DIR, exist_ok=True)
shutil.copy2("params.json", DEBUG_DIR)

parser.add_argument('--model_dir', default=DEBUG_DIR,
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")  # 'test' or 'train'
parser.add_argument('--embedding', default="fasttext",
                    help="Optional, embedding equal to 'vocab' or 'uncased' or 'fasttext'")
parser.add_argument('--model', default="lstm",
                    help="Optional, model equal to lstm', 'bert' or 'transformer'")
parser.add_argument('--nhead', default=None,
                    help="Optional, number")
parser.add_argument('--num_layers', default=None,
                    help="Optional, number")
parser.add_argument('--learning_rate', default=None,
                    help="Optional, number")


def train(model, optimizer, loss_fn, data_iterator, metrics, params, num_steps) -> Tuple[dict, List[List[List[Tuple[float, int, List[int], List[int]]]]]]:
    """Train the model on `num_steps` batches

    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    t = trange(num_steps)
    t1 = default_timer()  # time() returns a float

    # set model to training mode
    model.train()

    for i in t:
        # fetch the next training batch
        if params.model == "bert":
            train_batch, attention_mask_batch, labels_batch, idxes = next(data_iterator)
            encodings = {
                'input_ids': train_batch,
                'attention_mask': attention_mask_batch,
                'labels': labels_batch
            }
            if torch.any(labels_batch == -1):
                logging.warning("Warning: Found -1 label in batch", labels_batch)
            outputs = model(encodings)
            loss, tr_logits = outputs.loss, outputs.logits
            output_batch = tr_logits.view(-1, params.number_of_tags)
        else:
            train_batch, labels_batch, idxes = next(data_iterator)
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch, idxes)

        # clear previous gradients, compute gradients of all variables wrt loss
        optimizer.zero_grad()
        loss.backward()

        # performs updates using calculated gradients
        optimizer.step()

        # Evaluate summaries only once in a while
        loss_val = loss.clone().detach().item() if params.model == "bert" else loss.item()
        if i % params.save_summary_steps == 0:
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            output_batch = output_batch.data.cpu().numpy()
            labels_batch = labels_batch.data.cpu().numpy()

            # compute all metrics on this batch
            summary_batch = {
                'loss': loss_val
            }
            summary_batch.update({
                metric: metrics[metric](output_batch, labels_batch, idxes)
                for metric in metrics
            })
            summ.append(summary_batch)

        # update the average loss
        loss_avg.update(loss_val)
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

    t2 = default_timer()
    total_time = t2 - t1
    logging.debug(f"Training one epoch took {total_time} seconds")

    # compute mean of all metrics in summary
    metrics_mean = {
        "epoch_time": total_time
    }
    metrics_mean.update({
        metric: np.mean([x[metric] for x in summ]) if metric != "entity_classification" else net.agg_entity_classification(summ)
        for metric in summ[0]
    })
    mistakes = metrics_mean["entity_classification"]["mistakes"]
    del metrics_mean["entity_classification"]["mistakes"]
    metrics_string = " ; ".join("{}: {}".format(k, v) for k, v in metrics_mean.items())
    logging.debug("- Training Eval complete ; " + metrics_string)
    return metrics_mean, mistakes


def train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, metrics, params, model_dir, restore_file=None):
    """Train the model and evaluate every epoch.

    Args:
        model: (torch.nn.Module) the neural network
        train_data: (dict) training data with keys 'data' and 'labels'
        val_data: (dict) validaion data with keys 'data' and 'labels'
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    logging.debug(f"params: {json.dumps(params.__dict__)}")
    # reload weights from restore_file if specified
    if params.pre_trained_tar is not None:
        utils.load_checkpoint(params.pre_trained_tar, model, optimizer)
    elif restore_file is not None:
        restore_path = os.path.join(args.restore_file + '.pth.tar')
        logging.debug("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer if params.is_opt_checkpoint else None)

    best_val_metric = 0.0
    history = []

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.debug("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        num_steps = (params.train_size + 1) // params.batch_size
        train_data_iterator = data_loader.data_iterator(
            train_data, params, shuffle=True)
        train_metrics, train_mistakes = train(model, optimizer, loss_fn, train_data_iterator,
                               metrics, params, num_steps)

        # Evaluate for one epoch on validation set
        num_steps = (params.val_size + 1) // params.batch_size
        val_data_iterator = data_loader.data_iterator(
            val_data, params, shuffle=False)
        val_metrics, val_mistakes = evaluate(
            model, loss_fn, val_data_iterator, metrics, params, num_steps, True)

        val_metric = val_metrics['entity_classification']['overall_f1']
        is_best = val_metric >= best_val_metric

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict()},
                              optimizer=optimizer if params.is_opt_checkpoint else None,
                              is_best=is_best,
                              checkpoint=model_dir)

        # If best_eval, best_save_path
        if is_best:
            logging.debug("- Found new best accuracy")
            best_val_metric = val_metric

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)
            utils.gen_ner_mistakes_html(args.model_dir, args.data_dir, utils.DataTypeConstant.train, train_mistakes)
            utils.gen_ner_mistakes_html(args.model_dir, args.data_dir, utils.DataTypeConstant.val, val_mistakes)

        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)
        history.append({
            "train": train_metrics,
            "val": val_metrics
        })
    history_json_path = os.path.join(
        model_dir, "train_metrics_history.json")
    utils.save_dict_to_json(history, history_json_path)


if __name__ == '__main__':

    # Load the parameters from json file
    args = parser.parse_args()
    # --data_dir = data/small
    # --model_dir = experiments/base_model

    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)
    # {
    #     "learning_rate": 1e-3,
    #     "batch_size": 5,
    #     "num_epochs": 10,
    #     "lstm_hidden_dim": 50,
    #     "embedding_dim": 50,
    #     "save_summary_steps": 100
    # }

    # use GPU if available
    params.device = utils.get_local_device()  # use GPU is available

    # reset embedding_dim for bert
    if args.embedding:
        params.embedding = args.embedding
    if args.model:
        params.model = args.model
    if params.embedding != "vocab":
        if params.embedding == "fasttext":
            params.embedding_dim = 300
        else:
            params.embedding_dim = 768
    if params.model == "bert":
        params.embedding = "uncased"
    if args.learning_rate:
        params.learning_rate = float(args.learning_rate)
    if args.nhead:
        params.nhead = int(args.nhead)
    if args.num_layers:
        params.num_layers = int(args.num_layers)

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.device == "cuda":
        torch.cuda.manual_seed(230)
    elif params.device == "mps":
        torch.mps.manual_seed(230)

    # Set the logger
    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.debug("Loading the datasets...")

    # load data
    data_loader = DataLoader(args.data_dir, params)
    data = data_loader.load_data(['train', 'val'], args.data_dir)
    train_data = data['train']
    val_data = data['val']

    # specify the train and val dataset sizes
    params.train_size = train_data['size']
    params.val_size = val_data['size']

    logging.debug("- done.")

    # Define the model and optimizer
    model = net.Net(params).to(device=params.device)
    optimizer = optim.AdamW(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = net.loss_fn
    metrics = net.metrics.copy()

    # Train the model
    logging.debug("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_data, val_data, optimizer, loss_fn, metrics, params, args.model_dir,
                       args.restore_file)
