import os.path
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from torch.nn import CrossEntropyLoss
from torch.nn import CosineEmbeddingLoss
from sklearn.metrics import f1_score 
import numpy as np


from transformers import *


class multi_label_classifier(nn.Module):
    def __init__(self, args): 
        super(multi_label_classifier, self).__init__()
        self.args = args
        self.hidden_dim = args["hdd_size"]
        self.rnn_num_layers = args["num_rnn_layers"]
        self.num_labels = args["num_labels"]
        self.bce = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.n_gpu = args["n_gpu"]

        ### Utterance Encoder
        self.utterance_encoder = args["model_class"].from_pretrained(self.args["model_name_or_path"])

        self.bert_output_dim = args["config"].hidden_size
        
        if self.args["fix_encoder"]:
            print("[Info] fix_encoder")
            for p in self.utterance_encoder.parameters():
                p.requires_grad = False
        
        if self.args["more_linear_mapping"]:
            self.one_more_layer = nn.Linear(self.bert_output_dim, self.bert_output_dim)
        
        self.classifier = nn.Linear(self.bert_output_dim, self.num_labels)
        print("self.classifier", self.bert_output_dim, self.num_labels)

        ## Prepare Optimizer
        def get_optimizer_grouped_parameters(model):
            param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
                 'lr': args["learning_rate"]},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
                 'lr': args["learning_rate"]},
            ]
            return optimizer_grouped_parameters

        if self.n_gpu == 1:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(self)
        else:
            optimizer_grouped_parameters = get_optimizer_grouped_parameters(self.module)

        
        self.optimizer = AdamW(optimizer_grouped_parameters,
                                 lr=args["learning_rate"],)

    def optimize(self):
        self.loss_grad.backward()
        clip_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), self.args["grad_clip"])
        self.optimizer.step()
    
    def forward(self, data, ind_to_conf_map=None, evaluate_gradient=False, use_truth=True):

        if not evaluate_gradient:
            self.optimizer.zero_grad()

            inputs = {"input_ids": data[self.args["input_name"]], "attention_mask":(data[self.args["input_name"]] > 0).long()}

            if "gpt2" in self.args["model_type"]:
                hidden = self.utterance_encoder(**inputs)[0]
                hidden_head = hidden.mean(1)
            elif self.args["model_type"] == "dialogpt":
                transformer_outputs = self.utterance_encoder.transformer(
                    inputs["input_ids"],
                    attention_mask=(inputs["input_ids"] > 0).long())[0]
                hidden_head = transformer_outputs.mean(1)
            else:
                hidden = self.utterance_encoder(**inputs)[0]
                hidden_head = hidden[:, 0, :]

            # loss
            if self.args["more_linear_mapping"]:
                hidden_head = self.one_more_layer(hidden_head)

            logits = self.classifier(hidden_head)
            prob = self.sigmoid(logits)


            batch_size = len(data[self.args["task_name"]])
            noise_ratio = 1
            logits_noised = self.classifier(hidden_head.repeat(noise_ratio, 1) + 1.0 * torch.randn(
                hidden_head.repeat(noise_ratio, 1).size()).cuda())
            prob_noised = self.sigmoid(logits_noised)

            if self.args['embedding_noise']:  # add gaussian noise to embedding
                loss1 = self.bce(prob, data[self.args["task_name"]])
                loss2 = self.bce(prob_noised, data[self.args["task_name"]].repeat(noise_ratio, 1))
                loss = (loss1 * batch_size + loss2 * batch_size * noise_ratio) / (noise_ratio + 1)

            else:
                weights = []
                for idx in data['index']:
                    if ind_to_conf_map and idx in ind_to_conf_map:
                        if self.args['confidence_weighting']:
                            weights.append(self.args['lambda'] * ind_to_conf_map[idx]) # weight samples based on confidence
                        else:
                            weights.append(self.args['lambda']) # weight pseudo labeled samples by lambda
                    else:
                        weights.append(1)
                weights = torch.FloatTensor(weights)
                if torch.cuda.is_available(): weights = weights.cuda()

                weights = weights.view(prob.size(0), -1).repeat(1, prob.size(1))

                criterion = nn.BCELoss(reduction='none')
                losses = criterion(prob, data[self.args["task_name"]])

                weighted_loss = losses * weights
                loss = weighted_loss.mean()


            if self.training:
                self.loss_grad = loss
                self.optimize()

            predictions = (prob > 0.5)
            avg_pre_prob = []
            for i, pred in enumerate(predictions):
                if sum(pred) == 0:
                    avg_pre_prob.append(0)
                else:
                    avg_pre_prob.append(prob[i][pred].mean().item())

            outputs = {"loss":loss.item(),
                       "pred":predictions.detach().cpu().tolist(),
                       "label":data[self.args["task_name"]].detach().cpu().numpy(),
                       'confidence': avg_pre_prob}

            return outputs

        else:
            self.optimizer.zero_grad()

            inputs = {"input_ids": data[self.args["input_name"]],
                      "attention_mask": (data[self.args["input_name"]] > 0).long()}

            hidden = self.utterance_encoder(**inputs)[0]
            hidden_head = hidden[:, 0, :]

            # loss
            if self.args["more_linear_mapping"]:
                hidden_head = self.one_more_layer(hidden_head)

            logits = self.classifier(hidden_head)
            prob = self.sigmoid(logits)
            predictions = (prob > 0.5)

            criterion = nn.BCELoss()

            if use_truth:
                loss = criterion(prob, data[self.args["task_name"]])
            else:
                loss = criterion(prob, predictions.float())
            loss.backward()

            outputs = {"loss": loss.item(),
                       "pred": predictions.detach().cpu().tolist(),
                       "label": data[self.args["task_name"]].detach().cpu().numpy(),
                       "logits": logits
                       }

            return outputs
    
    def evaluation(self, preds, labels):
        preds = np.array(preds)
        labels = np.array(labels)
        results = {}
        for avg_name in ['micro', 'macro', 'weighted', 'samples']:
            my_f1_score = f1_score(y_true=labels, y_pred=preds, average=avg_name)
            results["f1_{}".format(avg_name)] = my_f1_score

        return results

