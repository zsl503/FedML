# 分层ckA聚合，能够决定间隔多少轮聚合某一层，且这些层通过cka相似度选择top-k聚合，会导致训练集明显提升，测试集明显下降
import copy
import logging
import random
from typing import Dict, List, Tuple
import fedml
from fedml.ml.trainer.my_model_trainer_classification import ModelTrainerCLS
import torch.nn as nn
import numpy as np
import torch
import wandb
from my_research.sp_fedavg_cifar10_resnet20_example.my_utils import *
from fedml import mlops


class MyTrainer_6(ModelTrainerCLS):

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
        model_trainer: MyTrainer_6,
    ):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

        self.args = args
        self.device = device
        self.model_trainer: MyTrainer_6 = model_trainer

    def update_local_dataset(self, client_idx, local_training_data,
                             local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        self.model_trainer.set_id(client_idx)

    def get_sample_number(self):
        return self.local_sample_number

    def train(self,
              w_global,
              round_idx,
              post_layer_filter: LayerFilter = None,
              set_layer_filter: LayerFilter = None,
              set_param:bool = True):
        '''
        set_layer_filter: 如果set_layer_filter不为空，则参数本地化时，只取用set_layer_filter后的参数
        post_layer_filter: 如果post_layer_filter不为空，则提交参数时，只提交经过post_layer_filter的参数
        '''
        if set_param:
            self.model_trainer.set_model_params(
                w_global if set_layer_filter is
                None else set_layer_filter(w_global))

        _, total_grad = self.model_trainer.train(self.local_training_data,
                                                 self.device, self.args)
        # for index, g in enumerate(grad_list):
        #     save_model_gradient(g, round_idx, f'c{self.client_idx}', post_desc=f'{index}')
        save_model_param(self.model_trainer.get_model_params(),
                         round_idx,
                         f'c{self.client_idx}',
                         is_grad=False)
        save_model_param(total_grad, round_idx, f'c{self.client_idx}_total')

        if post_layer_filter is None:
            return weight_sub(self.model_trainer.get_model_params(), w_global)
        else:
            return weight_sub(
                post_layer_filter(self.model_trainer.get_model_params()),
                post_layer_filter(w_global))

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics


class MyAvgAPI_6(object):

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
        self.model_trainer = MyTrainer_6(model, args)
        # self.model_trainer = create_model_trainer(model, args)

        self.default_unselect_keys = args.agg_unselect_layer
        self.default_all_select_keys = args.agg_all_select_layer
        self.default_any_select_keys = args.agg_any_select_layer

        self.cka_layer_filter = LayerFilter(
            unselect_keys=args.cka_unselect_layer,
            all_select_keys=args.cka_all_select_layer,
            any_select_keys=args.cka_any_select_layer)

        self.cka_topk = args.cka_select_topk
        logging.info(f'-----------CKA Filter Setting-----------')
        logging.info(f'cka_unselect_layer:{args.cka_unselect_layer}')
        logging.info(f'cka_all_select_layer:{args.cka_all_select_layer}')
        logging.info(f'cka_any_select_layer:{args.cka_any_select_layer}')
        logging.info(f'CKA Top-k:{self.cka_topk}')

        self.filter_mod_list: List[int] = args.agg_mod_list
        self.filter_mod_dict: Dict[int, Dict[str,
                                             List[str]]] = args.agg_mod_dict
        logging.info(f'-----------Default Filter Setting-----------')
        logging.info(f'default_unselect_keys:{self.default_unselect_keys}')
        logging.info(f'default_all_select_keys:{self.default_all_select_keys}')
        logging.info(f'default_any_select_keys:{self.default_any_select_keys}')
        logging.info(f'-----------Mod Filter Setting-----------')
        logging.info(f'filter_mod_list:{self.filter_mod_list}')
        logging.info(f'filter_mod_dict:{self.filter_mod_dict}')

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
                       MyTrainer_6(copy.deepcopy(model), args)
                       #    self.model_trainer,
                       )

            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        mlops.log_training_status(
            mlops.ClientConstants.MSG_MLOPS_CLIENT_STATUS_TRAINING)
        mlops.log_aggregation_status(
            mlops.ServerConstants.MSG_MLOPS_SERVER_STATUS_RUNNING)
        mlops.log_round_info(self.args.comm_round, -1)
        
        post_layer_filter = LayerFilter(
            unselect_keys=self.default_unselect_keys,
            all_select_keys=self.default_all_select_keys,
            any_select_keys=self.default_any_select_keys,
        )
        for round_idx in range(self.args.comm_round):

            logging.info(
                "################Communication round : {}".format(round_idx))

            if round_idx != 0:
                mod = 0
                for i in self.filter_mod_list:
                    if round_idx % i == 0:
                        mod = i
                        break
                # 如果有满足mod条件的
                if mod != 0:
                    logging.info(f"Satisfy condition {round_idx}%{mod}==0")
                    post_layer_filter.update_filter(
                        unselect_keys=self.filter_mod_dict[mod].get(
                            'agg_unselect_layer', None),
                        all_select_keys=self.filter_mod_dict[mod].get(
                            'agg_all_select_layer', None),
                        any_select_keys=self.filter_mod_dict[mod].get(
                            'agg_any_select_layer', None))
                else:
                    post_layer_filter.update_filter(
                        unselect_keys=self.default_unselect_keys,
                        all_select_keys=self.default_all_select_keys,
                        any_select_keys=self.default_any_select_keys,
                    )

            w_global = self.model_trainer.get_model_params()

            logging.info(f"post_layer_filter: {post_layer_filter}")

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

                g = client.train(w_global,
                                 round_idx=round_idx,
                                 post_layer_filter=post_layer_filter,
                                 set_param=False)

                mlops.event("train",
                            event_started=False,
                            event_value="{}_{}".format(str(round_idx),
                                                       str(idx)))
                # self.logging.info("local weights = " + str(w))
                g_locals.append((client.get_sample_number(), g))


            # test results
            # at last round
            if round_idx == self.args.comm_round - 1:
                self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                self._local_test_on_all_clients(round_idx)

            # update global weights
            mlops.event("agg", event_started=True, event_value=str(round_idx))
            
            # 结果已经保存到g_locals
            thresh = (self.args.cka_low_thresh, self.args.cka_high_thresh)
            g_all_global, g_cka_global = self._aggregate_by_cka(g_locals, 
                                                  self.cka_layer_filter, 
                                                  self.cka_topk,
                                                  thresh)

            for i, idx in enumerate(client_indexes):
                client: Client = self.client_list[idx]
                g_global = weight_add(post_layer_filter(w_global), g_cka_global[i])
                client.model_trainer.set_model_params(g_global)

            w_global = weight_add(post_layer_filter(w_global), g_all_global)
            self.model_trainer.set_model_params(w_global)
            save_model_param(w_global, round_idx, "server", is_grad=False)

            mlops.event("agg", event_started=False, event_value=str(round_idx))
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

    def _aggregate_by_cka(self, w_locals, 
                          cka_layer_filter:LayerFilter,
                          cka_topk:int, 
                          thresh:Tuple[int, int] = None):
        # 结果保存到w_locals
        (sample_num, params) = w_locals[0]
        cka_param = cka_layer_filter(params)
        w_avg_global = {}
        w_cka_global = [{} for i in w_locals]
        for layer_name in (params.keys() - cka_param.keys()):
            averaged_layer = aggregate_layer(w_locals, layer_name=layer_name)
            w_avg_global[layer_name] = averaged_layer

            for idx, (_, w) in enumerate(w_locals):
                w_cka_global[idx][layer_name] = averaged_layer.clone()

        if len(cka_param.keys()) == 0:
            return w_avg_global, w_cka_global

        logging.info(f"Begin CKA aggregate, keys:{cka_param.keys()}")
        if thresh is None:
            thresh = (0,1)
        elif thresh[0] is None:
            thresh[0] = 0
        elif thresh[1] is None:
            thresh[1] = 1

        for layer_name in cka_param.keys():
            averaged_layer = aggregate_layer(w_locals, layer_name=layer_name)
            w_avg_global[layer_name] = averaged_layer

            cka_matrix = get_cka_matrix(w_locals, layer_name)
            # logging.info(f"CKA Matrix for {layer_name}:\n{cka_matrix}")
            for idx, (_, w) in enumerate(w_locals):
                topk_indices_i = topk_indices(cka_matrix[idx], cka_topk)
                use_indices_i = [i for i in topk_indices_i if thresh[0] <= cka_matrix[idx][i] <= thresh[1]]
                if idx not in use_indices_i:
                    use_indices_i.append(idx)
                    logging.warn(f'Layer {layer_name} Appended {idx} itself to selected indices which not in top-k:{topk_indices_i}. \n\
                                 Used indices: {use_indices_i} Threshold: {thresh} \nCKA matrix: {cka_matrix}')
                    
                averaged_layer = aggregate_layer(
                    [w_locals[i] for i in use_indices_i], 
                    layer_name=layer_name)
                w_cka_global[idx][layer_name] = averaged_layer.clone()
        return w_avg_global, w_cka_global

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
            train_local_metrics = self.client_list[client_idx].local_test(
                False)
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
                        (train_metrics["num_correct"][-1] /
                         train_metrics["num_samples"][-1]),
                        f"Train/c{client_idx}-perLoss":
                        (train_metrics["losses"][-1] /
                         train_metrics["num_samples"][-1]),
                        f"Test/c{client_idx}-perAcc":
                        (test_metrics["num_correct"][-1] /
                         test_metrics["num_samples"][-1]),
                        f"Test/c{client_idx}-perLoss":
                        (test_metrics["losses"][-1] /
                         test_metrics["num_samples"][-1]),
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
            wandb.log(
                {
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
