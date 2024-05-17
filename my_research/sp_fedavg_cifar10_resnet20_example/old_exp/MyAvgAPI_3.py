import copy
from datetime import datetime
import logging
import os
import random
from typing import OrderedDict
import fedml
from fedml.ml.trainer.my_model_trainer_classification import ModelTrainerCLS
import torch.nn as nn

from fedml.core.alg_frame.client_trainer import ClientTrainer
import numpy as np
import torch
import wandb

from fedml import mlops
from fedml.ml.trainer.trainer_creator import create_model_trainer
# from fedml.simulation.sp.fedavg.client import Client
TIME = datetime.now().strftime('%Y%m%d%H%M')


def save_model_param(model_params,
                        round_idx,
                        path_tag,
                        pre_desc=None,
                        post_desc=None,
                        is_grad=True):
    # 保存全局权重，作为实验
    if pre_desc is None:
        pre_desc = path_tag

    save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'grad_lists' if is_grad else 'weight_lists', TIME, path_tag)
    os.makedirs(save_dir, exist_ok=True)
    if post_desc is None:
        path = os.path.join(save_dir, f'{pre_desc}_round_{round_idx}.pt')
    else:
        path = os.path.join(save_dir,
                            f'{pre_desc}_round_{round_idx}_{post_desc}.pt')

    logging.info(f"Save {path_tag} {round_idx} model params to '{path}'.")
    torch.save(model_params, path)


def weight_sub(weight_x, weight_y):
    """
    Calculate the difference between two model weights.

    Args:
        weight_x (dict): State dictionary of the first model.
        weight_y (dict): State dictionary of the second model.
    Returns:
        dict: weight_x - weight_y
    """
    device = next(iter(weight_x.values())).device
    # Create a new dictionary to store the weight differences
    weight_diff = {}
    # Iterate through the keys (parameter names) in weight_x
    for key in weight_x.keys():
        # Compute the difference between corresponding weight tensors
        diff = weight_x[key].to(device) - weight_y[key].to(device)
        # Store the difference in the weight_diff dictionary
        weight_diff[key] = diff
    return weight_diff


def weight_add(weight_x, weight_y):
    """
    Calculate the addition result between two model weights.

    Args:
        weight_x (dict): State dictionary of the first model.
        weight_y (dict): State dictionary of the second model.
    Returns:
        dict: weight_x + weight_y
    """
    # Create a new dictionary to store the weight differences
    device = next(iter(weight_x.values())).device
    weight_add = {}
    # Iterate through the keys (parameter names) in weight_x
    for key in weight_x.keys():
        # Compute the difference between corresponding weight tensors
        weight_add[key] = weight_x[key].to(device) + weight_y[key].to(device)
    return weight_add


def get_model_gradient(model):
    """
    Description:
        - get norm gradients from model, and store in a OrderDict
    
    Args:
        - model: (torch.nn.Module), torch model
    
    Returns:
        - grads in OrderDict
    """
    grads = OrderedDict()
    for name, params in model.named_parameters():
        grad = params.grad
        if grad is not None:
            grads[name] = grad
    return grads


def filter_param_by_keys(param_dict,
                         unselect_keys=None,
                         all_select_keys=None,
                         any_select_keys=None):
    # filtered_weight_list = weight_list
    return {
        layer_key: param
        for layer_key, param in param_dict.items()
        if (unselect_keys is None or len(unselect_keys) == 0 or all(
            key not in layer_key for key in unselect_keys)) and
        (all_select_keys is None or len(all_select_keys) == 0 or all(
            key in layer_key for key in all_select_keys)) and (
                any_select_keys is None or len(any_select_keys) == 0 or any(
                    key in layer_key for key in any_select_keys))
    }


class MyTrainer_3(ModelTrainerCLS):

    def __init__(self, model, args):
        super().__init__(model, args)

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters, strict=False)

    def train(self, train_data, device, args):
        model = self.model
        model.to(device)
        model.train()
        grad_list = []

        # train and update
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )
        criterion = nn.CrossEntropyLoss().to(device)  # pylint: disable=E1102

        epoch_loss = []
        old_model = copy.deepcopy(self.model.state_dict())
        for epoch in range(args.epochs):
            batch_loss = []

            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                labels = labels.long()
                loss = criterion(log_probs, labels)  # pylint: disable=E1102
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            if len(batch_loss) == 0:
                epoch_loss.append(0.0)
            else:
                epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info("Client Index = {}\tEpoch: {}\tLoss: {:.6f}".format(
                self.id, epoch,
                sum(epoch_loss) / len(epoch_loss)))

            grad_list.append(get_model_gradient(self.model))
        
        total_grad = weight_sub(self.model.state_dict(), old_model)
        return grad_list, total_grad


class Client:

    def __init__(
        self,
        client_idx,
        local_training_data,
        local_test_data,
        local_sample_number,
        args,
        device,
        model_trainer: MyTrainer_3,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

        self.args = args
        self.device = device
        self.model_trainer: MyTrainer_3 = model_trainer

    def update_local_dataset(self, client_idx, local_training_data,
                             local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.model_trainer.set_id(client_idx)

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global, round_idx):
        self.model_trainer.set_model_params(w_global)

        grad_list, total_grad = self.model_trainer.train(
            self.local_training_data, self.device, self.args)
        # for index, g in enumerate(grad_list):
        #     save_model_gradient(g, round_idx, f'c{self.client_idx}', post_desc=f'{index}')

        save_model_param(self.model_trainer.get_model_params(), round_idx, f'c{self.client_idx}', is_grad=False)
        save_model_param(total_grad, round_idx, f'c{self.client_idx}_total')

        return filter_param_by_keys(total_grad, self.args.agg_unselect_layer,
                                    self.args.agg_all_select_layer,
                                    self.args.agg_any_select_layer)

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics


class MyAvgAPI_3(object):

    def __init__(self, args: fedml.arguments.Arguments, device, dataset,
                 model):
        self.device = device
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            class_num,
        ] = dataset
        self.args = args
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list: list[Client] = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        logging.info("model = {}".format(model))
        self.model_trainer = MyTrainer_3(model, args)
        # self.model_trainer = create_model_trainer(model, args)

        self.model_trainer.get_model_params()
        filter_param = filter_param_by_keys(
            self.model_trainer.get_model_params(), args.agg_unselect_layer,
            args.agg_all_select_layer, args.agg_any_select_layer)
        logging.info(
            f"agg_unselect_layer={args.agg_unselect_layer}\tagg_all_select_layer={args.agg_all_select_layer}\tagg_any_select_layer={args.agg_any_select_layer}"
        )

        logging.info(f"selected_layer={filter_param.keys()}")

        self.model = model
        logging.info("self.model_trainer = {}".format(self.model_trainer))

        self._setup_clients(train_data_local_num_dict, train_data_local_dict,
                            test_data_local_dict, self.model, self.args)

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict,
                       test_data_local_dict, model, args):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx],
                       test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args,
                       self.device, 
                       MyTrainer_3(copy.deepcopy(model), args)
                    #    self.model_trainer,
                       )
            
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        w_global = self.model_trainer.get_model_params()
        w_global = filter_param_by_keys(w_global, self.args.agg_unselect_layer,
                                        self.args.agg_all_select_layer,
                                        self.args.agg_any_select_layer)
        mlops.log_training_status(
            mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
        mlops.log_aggregation_status(
            mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        mlops.log_round_info(self.args.comm_round, -1)
        for round_idx in range(self.args.comm_round):

            logging.info(
                "################Communication round : {}".format(round_idx))

            g_locals = []
            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self._client_sampling(
                round_idx, self.args.client_num_in_total,
                self.args.client_num_per_round)
            logging.info("client_indexes = " + str(client_indexes))
            for idx in client_indexes:
                # update dataset
                client: Client = self.client_list[idx]

                # train on new dataset
                mlops.event("train",
                            event_started=True,
                            event_value="{}_{}".format(str(round_idx),
                                                       str(idx)))
                g = client.train(w_global, round_idx=round_idx)

                mlops.event("train",
                            event_started=False,
                            event_value="{}_{}".format(str(round_idx),
                                                       str(idx)))
                # self.logging.info("local weights = " + str(w))
                g_locals.append((client.get_sample_number(), g))

            # update global weights
            mlops.event("agg", event_started=True, event_value=str(round_idx))
            g_global = self._aggregate(g_locals)
            save_model_param(g_global, round_idx, "server")

            w_global = weight_add(w_global, g_global)
            self.model_trainer.set_model_params(w_global)
            mlops.event("agg", event_started=False, event_value=str(round_idx))

            # test results
            # at last round
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    self._local_test_on_all_clients(round_idx)

            mlops.log_round_info(self.args.comm_round, round_idx)

        mlops.log_training_finished_status()
        mlops.log_aggregation_finished_status()

    def _client_sampling(self, round_idx, client_num_in_total,
                         client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [
                client_index for client_index in range(client_num_in_total)
            ]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(
                round_idx
            )  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total),
                                              num_clients,
                                              replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num),
                                       min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset,
                                         sample_indices)
        sample_testset = torch.utils.data.DataLoader(
            subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        averaged_params = copy.deepcopy(averaged_params)
        for k in averaged_params.keys():
            averaged_params[k] = averaged_params[k] * sample_num / training_num
            for i in range(1, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                averaged_params[k] += local_model_params[k] * w
        return averaged_params

    def _aggregate_noniid_avg(self, w_locals):
        """
        The old aggregate method will impact the model performance when it comes to Non-IID setting
        Args:
            w_locals:
        Returns:
        """
        (_, averaged_params) = w_locals[0]
        averaged_params = copy.deepcopy(averaged_params)
        for k in averaged_params.keys():
            temp_w = []
            for (_, local_w) in w_locals:
                temp_w.append(local_w[k])
            averaged_params[k] = sum(temp_w) / len(temp_w)
        return averaged_params

    def _test(self, test_data):
        device = self.device
        model = self.model
        model.to(self.device)
        model.eval()

        metrics = {
            "test_correct": 0,
            "test_loss": 0,
            "test_precision": 0,
            "test_recall": 0,
            "test_total": 0,
        }
        criterion = nn.CrossEntropyLoss().to(self.device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)  # pylint: disable=E1102

                _, predicted = torch.max(pred, 1)
                correct = predicted.eq(target).sum()

                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                if len(target.size()) == 1:  #
                    metrics["test_total"] += target.size(0)
                elif len(target.size()
                         ) == 2:  # for tasks of next word prediction
                    metrics["test_total"] += target.size(0) * target.size(1)
        return metrics

    def _local_test_on_all_clients(self, round_idx):
        logging.info(
            "################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        test_metrics = {"num_samples": [], "num_correct": [], "losses": []}

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue

            # train data
            train_local_metrics = self.client_list[client_idx].local_test(False)
            train_metrics["num_samples"].append(
                copy.deepcopy(train_local_metrics["test_total"]))
            train_metrics["num_correct"].append(
                copy.deepcopy(train_local_metrics["test_correct"]))
            train_metrics["losses"].append(
                copy.deepcopy(train_local_metrics["test_loss"]))

            # test data
            test_local_metrics = self.client_list[client_idx].local_test(True)
            test_metrics["num_samples"].append(
                copy.deepcopy(test_local_metrics["test_total"]))
            test_metrics["num_correct"].append(
                copy.deepcopy(test_local_metrics["test_correct"]))
            test_metrics["losses"].append(
                copy.deepcopy(test_local_metrics["test_loss"]))

            if self.args.enable_wandb:
                wandb.log(
                    {
                        f"Train/c{client_idx}-Acc":
                        train_metrics["num_correct"][-1],
                        f"Train/c{client_idx}-Loss":
                        train_metrics["losses"][-1],
                        f"Test/c{client_idx}-Acc":
                        test_metrics["num_correct"][-1],
                        f"Test/c{client_idx}-Loss":
                        test_metrics["losses"][-1],
                        # per acc
                        f"Train/c{client_idx}-perAcc":
                        (train_metrics["num_correct"][-1] / train_metrics["num_samples"][-1]),
                        f"Train/c{client_idx}-perLoss":
                        (train_metrics["losses"][-1] / train_metrics["num_samples"][-1]),
                        f"Test/c{client_idx}-perAcc":
                        (test_metrics["num_correct"][-1] / test_metrics["num_samples"][-1]),
                        f"Test/c{client_idx}-perLoss":
                        (test_metrics["losses"][-1] / test_metrics["num_samples"][-1]),
                        "round":
                        round_idx
                    },
                    step=round_idx)

        # test on training dataset
        train_acc = sum(train_metrics["num_correct"]) / sum(
            train_metrics["num_samples"])
        train_loss = sum(train_metrics["losses"]) / sum(
            train_metrics["num_samples"])

        # test on test dataset
        test_acc = sum(test_metrics["num_correct"]) / sum(
            test_metrics["num_samples"])
        test_loss = sum(test_metrics["losses"]) / sum(
            test_metrics["num_samples"])

        stats = {"training_acc": train_acc, "training_loss": train_loss}
        if self.args.enable_wandb:
            wandb.log({
                "Train/Acc": train_acc,
                "Train/Loss": train_loss,
                "round": round_idx
            },
            step=round_idx)

        mlops.log({"Train/Acc": train_acc, "round": round_idx})
        mlops.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {"test_acc": test_acc, "test_loss": test_loss}
        if self.args.enable_wandb:
            wandb.log(
                {
                    "Test/Acc": test_acc,
                    "Test/Loss": test_loss,
                    "round": round_idx
                },
                step=round_idx)

        mlops.log({"Test/Acc": test_acc, "round": round_idx})
        mlops.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)

    def _local_test_on_validation_set(self, round_idx):

        logging.info(
            "################local_test_on_validation_set : {}".format(
                round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {"test_acc": test_acc, "test_loss": test_loss}
            if self.args.enable_wandb:
                wandb.log(
                    {
                        "Test/Acc": test_acc,
                        "Test/Loss": test_loss,
                        "round": round_idx
                    },
                    step=round_idx)

            mlops.log({"Test/Acc": test_acc, "round": round_idx})
            mlops.log({"Test/Loss": test_loss, "round": round_idx})

        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_pre = test_metrics["test_precision"] / test_metrics[
                "test_total"]
            test_rec = test_metrics["test_recall"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {
                "test_acc": test_acc,
                "test_pre": test_pre,
                "test_rec": test_rec,
                "test_loss": test_loss,
            }
            if self.args.enable_wandb:
                wandb.log(
                    {
                        "Test/Acc": test_acc,
                        "Test/Pre": test_pre,
                        "Test/Rec": test_rec,
                        "Test/Loss": test_loss,
                        "round": round_idx
                    },
                    step=round_idx)

            mlops.log({"Test/Acc": test_acc, "round": round_idx})
            mlops.log({"Test/Pre": test_pre, "round": round_idx})
            mlops.log({"Test/Rec": test_rec, "round": round_idx})
            mlops.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!" %
                            self.args.dataset)

        logging.info(stats)
