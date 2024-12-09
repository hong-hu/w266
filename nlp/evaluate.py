"""Evaluates the model"""

import argparse
import json
import logging
import os
from timeit import default_timer
from typing import List, Tuple

import numpy as np
import torch

import model.net as net
import utils
from model.data_loader import DataLoader

os.environ["TOKENIZERS_PARALLELISM"] = "false"

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/flarener',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='/Users/honghu/Doc/w266/Project/data_backup/train/finerord/fasttext_transformer_10x2',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best',
                    help="name of the file in --model_dir containing weights to load")
parser.add_argument('--embedding', default=None,
                    help="Optional, embedding equal to 'torch' or 'uncased' or 'cased'")  # 'torch' or 'uncased' or 'cased'
parser.add_argument('--model', default=None,
                    help="Optional, model equal to lstm' or 'transformer'")  # 'lstm' or 'transformer'
parser.add_argument('--nhead', default=None,
                    help="Optional, number")
parser.add_argument('--num_layers', default=None,
                    help="Optional, number")
parser.add_argument('--learning_rate', default=None,
                    help="Optional, number")


def evaluate(model, loss_fn, data_iterator, metrics, params, num_steps, is_val) -> Tuple[dict, List[List[List[Tuple[float, int, List[int], List[int]]]]]]:
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        data_iterator: (generator) a generator that generates batches of data and labels
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # summary for current eval loop
    summ = []

    t1 = default_timer()  # time() returns a float

    # set model to evaluation mode
    model.eval()

    # compute metrics over the dataset
    for _ in range(num_steps):
        # fetch the next evaluation batch
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
            data_batch, labels_batch, idxes = next(data_iterator)
            # compute model output
            output_batch = model(data_batch)
            loss = loss_fn(output_batch, labels_batch, idxes)

        # extract data from torch Variable, move to cpu, convert to numpy arrays
        output_batch = output_batch.data.cpu().numpy()
        labels_batch = labels_batch.data.cpu().numpy()

        # compute all metrics on this batch
        loss_val = loss.clone().detach().item() if params.model == "bert" else loss.item()
        summary_batch = {
            'loss': loss_val
        }
        summary_batch.update({
            metric: metrics[metric](output_batch, labels_batch, idxes)
            for metric in metrics
        })
        summ.append(summary_batch)

    t2 = default_timer()
    total_time = t2 - t1
    logging.debug(f"{'Validation' if is_val else 'Test'} took {total_time} seconds")
    metrics_mean = {
        "process_time": total_time
    }

    # compute mean of all metrics in summary
    metrics_mean.update({
        metric: np.mean([x[metric] for x in summ]) if metric != "entity_classification" else net.agg_entity_classification(summ)
        for metric in summ[0]
    })
    mistakes = metrics_mean["entity_classification"]["mistakes"]
    del metrics_mean["entity_classification"]["mistakes"]
    metrics_string = " ; ".join("{}: {}".format(k, v) for k, v in metrics_mean.items())
    prefix = "- Validation Eval complete ; " if is_val else "- Test Eval complete ; "
    if is_val:
        logging.debug(prefix + metrics_string)
    else:
        logging.info(prefix + metrics_string)
    return metrics_mean, mistakes


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    if not os.path.exists(json_path):
        best_path = os.path.join(args.model_dir, 'results.json')
        assert os.path.isfile(best_path), "No results json file found at {}".format(best_path)
        with open(best_path, 'r') as f:
            best_info = json.load(f)
            logging.debug(f"Find best model based on {best_path}: {json.dumps(best_info)}")
            args.model_dir = best_info['model_dir']

    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

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

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.debug("Creating the dataset...")

    # load data
    data_loader = DataLoader(args.data_dir, params)
    data = data_loader.load_data(['test'], args.data_dir)
    test_data = data['test']

    # specify the test set size
    params.test_size = test_data['size']
    test_data_iterator = data_loader.data_iterator(test_data, params)

    logging.debug("- done.")

    # Define the model
    model = net.Net(params).to(device=params.device)

    loss_fn = net.loss_fn
    metrics = net.metrics.copy()

    logging.debug("Starting evaluation")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    num_steps = (params.test_size + 1) // params.batch_size
    test_metrics, test_mistakes = evaluate(model, loss_fn, test_data_iterator, metrics, params, num_steps, False)
    save_path = os.path.join(args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
    utils.gen_ner_mistakes_html(args.model_dir, args.data_dir, utils.DataTypeConstant.test, test_mistakes)
