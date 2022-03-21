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


class multi_class_classifier(nn.Module):
    def __init__(self, args): #, num_labels, device):
        super(multi_class_classifier, self).__init__()
        self.args = args
        self.hidden_dim = args["hdd_size"]
        self.rnn_num_layers = args["num_rnn_layers"]
        self.num_labels = args["num_labels"]
        self.xeloss = nn.CrossEntropyLoss()
        self.n_gpu = args["n_gpu"]
        self.training = args["do_train"]

        ### Utterance Encoder
        self.utterance_encoder = args["model_class"].from_pretrained(self.args["model_name_or_path"])
        
        self.bert_output_dim = args["config"].hidden_size
        
        if self.args["fix_encoder"]:
            print("[Info] Fixing Encoder...")
            for p in self.utterance_encoder.parameters():
                p.requires_grad = False
        
        if self.args["more_linear_mapping"]:
            self.one_more_layer = nn.Linear(self.bert_output_dim, self.bert_output_dim)
        
        self.classifier = nn.Linear(self.bert_output_dim, self.num_labels)

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

            if self.args["fix_encoder"]:
                with torch.no_grad():
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
            else:
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

            batch_size = len(data[self.args["task_name"]])
            noise_ratio = 1
            hidden_norm = torch.norm(hidden_head, dim=1)

            hidden_norm = hidden_norm.repeat(hidden_head.size(1),1).transpose(0,1)
            if self.args['noise_weight_type'] == 'samplewise':
                logits_noised = self.classifier(hidden_head.repeat(noise_ratio,1) + self.args['lambda'] * hidden_norm * torch.randn(hidden_head.repeat(noise_ratio,1).size()).cuda())
            if self.args['noise_weight_type'] == 'elementwise':
                logits_noised = self.classifier(
                    hidden_head.repeat(noise_ratio, 1) + self.args['lambda'] * torch.abs(hidden_head) * torch.randn(
                        hidden_head.repeat(noise_ratio, 1).size()).cuda())


            if self.args['embedding_noise']: # add gaussian noise to embedding
                loss1 = self.xeloss(logits, data[self.args["task_name"]])
                loss2 = self.xeloss(logits_noised, data[self.args["task_name"]].repeat(noise_ratio))
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

                criterion = nn.CrossEntropyLoss(reduction='none')
                losses = criterion(logits, data[self.args["task_name"]])

                loss = losses * weights
                loss = loss.mean()

            if self.training:
                self.loss_grad = loss
                self.optimize()

            softmax = nn.Softmax(-1)
            prob = softmax(logits)
            confidence, predictions = torch.max(prob, dim=-1)

            outputs = {"loss": loss.item(),
                       "logits": logits,
                       "pred": predictions.detach().cpu().tolist(),
                       "label": data[self.args["task_name"]].detach().cpu().numpy(),
                       "prob": prob,
                       'confidence': confidence.detach().cpu().tolist()
            }
            return outputs

        # only evaluate the gradient wrt input
        else:
            self.optimizer.zero_grad()

            inputs = {"input_ids": data[self.args["input_name"]],
                      "attention_mask": (data[self.args["input_name"]] > 0).long()}

            if self.args["fix_encoder"]:
                with torch.no_grad():
                    hidden = self.utterance_encoder(**inputs)[0]
                    hidden_head = hidden[:, 0, :]
            else:
                hidden = self.utterance_encoder(**inputs)[0]
                hidden_head = hidden[:, 0, :]

            logits = self.classifier(hidden_head)

            criterion = nn.CrossEntropyLoss(reduction='mean')
            if use_truth:
                loss = criterion(logits, data[self.args["task_name"]])
            else:
                loss = criterion(logits, torch.argmax(logits, -1))

            loss.backward()

            outputs = {
                "logits": logits,
                "label": data[self.args["task_name"]]
            }

            return outputs
    
    def evaluation(self, preds, labels):
        preds = np.array(preds)
        labels = np.array(labels)
        
        if self.args["task_name"] == "intent":
            oos_idx = self.args["unified_meta"]["intent"]["oos"]
            acc = (preds == labels).mean()
            oos_labels, oos_preds = [], []
            ins_labels, ins_preds = [], []
            for i in range(len(preds)):
                if labels[i] != oos_idx:
                    ins_preds.append(preds[i])
                    ins_labels.append(labels[i])

                oos_labels.append(int(labels[i] == oos_idx))
                oos_preds.append(int(preds[i] == oos_idx))

            ins_preds = np.array(ins_preds)
            ins_labels = np.array(ins_labels)
            oos_preds = np.array(oos_preds)
            oos_labels = np.array(oos_labels)
            ins_acc = (ins_preds == ins_labels).mean()
            oos_acc = (oos_preds == oos_labels).mean()

            # for oos samples recall = tp / (tp + fn) 
            TP = (oos_labels & oos_preds).sum()
            FN = ((oos_labels - oos_preds) > 0).sum()
            recall = TP / (TP+FN)
            results = {"acc":acc, "ins_acc":ins_acc, "oos_acc":oos_acc, "oos_recall":recall}
        else:
            acc = (preds == labels).mean()
            results = {"acc":acc}

        return results