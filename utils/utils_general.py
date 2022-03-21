import torch
import torch.utils.data as data
import random
import logging
import math
import numpy as np
from tqdm import tqdm
from collections import Counter
import torch.nn.functional as F
import copy
import warnings

from transformers import AutoModelForMaskedLM, AutoTokenizer, BertConfig, BertTokenizer, BertModel

from .Interpret.smooth_gradient import SmoothGradient
from .Interpret.vanilla_gradient import VanillaGradient
from .dataloader_dst import *
from .dataloader_nlg import *
from .dataloader_nlu import *
from .dataloader_dm import *
from .dataloader_usdl import *


task_to_target = {
    'intent': 'intent',
    'dst': 'belief_ontology'
}


def get_loader(args, mode, tokenizer, datasets, unified_meta, shuffle=False, for_unlabeled=False):
    """
    Get the initial datasets
    :param args: global configration
    :param mode: train/dev/test
    :param tokenizer: bert tokenizer
    :param datasets: datasets to be used
    :param unified_meta: meta information for the dataset
    :param: shuffle: whether to shuffle the dataset or not
    :param: for_unlabeled: whether to get the unlabeled instances from the original training dataset
    :return: train dataset or dev dataset or test dataset or unlabeled dataset
    """
    task = args["task"]
    batch_size = args["batch_size"] if mode == "train" else args["eval_batch_size"]

    combined_ds = []
    for ds in datasets:
        combined_ds += datasets[ds][mode]

    # do not consider empty system responses
    if (args["task_name"] == "rs") or (args["task"] == "dm"):
        print("[Info] Remove turns with empty system response...")
        combined_ds = [d for d in combined_ds if d["turn_sys"] != ""]

    if (args["task_name"] == "rs"):
        print("[Info] Remove turn=0 system response...")
        combined_ds = [d for d in combined_ds if d["turn_id"] != 0]

    # control data ratio
    data_ratio = args["train_data_ratio"]
    if (data_ratio != 1 or args["nb_shots"] != -1) and (mode == "train"):
        original_len = len(combined_ds)

        if ("oos_intent" in args["dataset"]):
            nb_train_sample_per_class = int(100 * data_ratio)
            class_count = {k: 0 for k in unified_meta["intent"]}
            random.Random(args["rand_seed"]).shuffle(combined_ds)
            pair_trn_new = []
            for d in combined_ds:
                if for_unlabeled:  # take the first nb_train_sample_per_class data for this class
                    if class_count[d["intent"]] < nb_train_sample_per_class:  # skip the first nb_train_sample_per_class data for this class
                        class_count[d["intent"]] += 1
                    else:
                        pair_trn_new.append(d)
                else:
                    if class_count[d["intent"]] < nb_train_sample_per_class:
                        pair_trn_new.append(d)
                        class_count[d["intent"]] += 1

            combined_ds = pair_trn_new
        else:
            if data_ratio != 1:
                random.Random(args["rand_seed"]).shuffle(combined_ds)
                if for_unlabeled:
                    combined_ds = combined_ds[int(len(combined_ds) * data_ratio):]
                else:
                    combined_ds = combined_ds[:int(len(combined_ds) * data_ratio)]
            else:
                random.Random(args["rand_seed"]).shuffle(combined_ds)
                if not for_unlabeled:
                    combined_ds = combined_ds[:args["nb_shots"]]
                else:
                    combined_ds = combined_ds[args["nb_shots"]:]

        print("[INFO] Use Training Data: from {} to {}".format(original_len, len(combined_ds)))

    data_info = {k: [] for k in
                 combined_ds[0].keys()}  # from [{turn_usr:, intent:},.] pairs to {turn_usr:[], intent:[], ...}
    for d in combined_ds:
        for k in combined_ds[0].keys():
            data_info[k].append(d[k])

    dataset = globals()["Dataset_" + task](data_info, tokenizer, args, unified_meta, mode, args["max_seq_length"])

    bool_shuffle = (mode == "train" or shuffle)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=bool_shuffle,
        collate_fn=globals()[
            "collate_fn_{}_{}".format(task, args["example_type"])]
    )

    return data_loader


def get_remaining_loader(
        args,
        unlabeled_dataset,
        new_candidate_num,
        seen_ind,
        use_confidence
):
    """
    Get the get_remaining_loader dataloader
    :param args:
    :param unlabeled_dataset: the whole unlabeled dataset
    :param new_candidate_num: number of new samples to be considered in each iteration
    :param seen_ind: indexes of unlabeled samples seen so far
    :return: candidate_loader
    """
    if new_candidate_num > 0:
        remaining_ind = list(
            set([i for i in range(len(unlabeled_dataset))]) - set(seen_ind)
        )

        if args['task'] == 'dst':
            remaining_ind = random.Random(args["rand_seed"]).sample(remaining_ind, int(
                new_candidate_num / args['confidence_top_ratio']))

        assert new_candidate_num <= len(remaining_ind)
        if not use_confidence:
            remaining_ind = random.Random(args["rand_seed"]).sample(remaining_ind, new_candidate_num)

    dataset_to_label = torch.utils.data.Subset(unlabeled_dataset, remaining_ind)  # the subset we want to pesudo label

    candidate_loader = torch.utils.data.DataLoader(
        dataset=dataset_to_label,
        batch_size=args["eval_batch_size"],
        shuffle=True,
        collate_fn=globals()[
            "collate_fn_{}_{}".format(args["task"], args["example_type"])
        ]
    )

    return candidate_loader


def get_unified_meta(datasets):
    """
    A helper function to get the meta info for a given dataset
    :param datasets: list of datasets to be processed
    :return the a dictionary of meta information
    """
    unified_meta = {"others": None}
    for ds in datasets:
        for key, value in datasets[ds]["meta"].items():
            if key not in unified_meta.keys():
                unified_meta[key] = {}
            if type(value) == list:
                for v in value:
                    if v not in unified_meta[key].keys():
                        unified_meta[key][v] = len(unified_meta[key])
            else:
                unified_meta[key] = value

    return unified_meta


def merge_loaders(task, dataloaders, seen_ind):
    """
    A helper function to merge the original train dataset and pseudo-labeled dataset
    :param task: task name
    :param dataloaders: list of dataloaders
    :param seen_ind: a list of indices in the unlabeled dataset that are already pseudo-labeled
    :return a merged dataset L (contains the original train dataset and pseudo-labeled dataset)
    """
    merged_dataset = torch.utils.data.ConcatDataset(
        [
            dataloaders[0].dataset,
            torch.utils.data.Subset(dataloaders[1].dataset.dataset, seen_ind)
        ]
    )

    return torch.utils.data.DataLoader(
        dataset=merged_dataset,
        batch_size=dataloaders[0].batch_size,
        shuffle=True,
        collate_fn=dataloaders[0].collate_fn
    )


def get_pseudo_label_loader(
        args,
        model,
        unlabeled_dataset,
        seen_ind,
        ind_to_conf_map
):
    """
    :param args: global config
    :param model: teacher model F^T
    :param unlabeled_dataset: the global unlabeled dataset U
    :param seen_ind: list of indices in U that are already pseudo-labeled
    :return: a loader for the new subset to label, all pesudo labeled indexes so far, and updated index-to-confidence map
    """

    # candidate loader with remaining unlabeled data
    candidate_loader = get_remaining_loader(
        args=args,
        unlabeled_dataset=unlabeled_dataset,
        new_candidate_num=args['new_candidate_num'],
        seen_ind=seen_ind,
        use_confidence=args['confidence_selection']
    )

    candidate_pbar = tqdm(candidate_loader)

    # evaluation on all unlabeled data using the current model
    unlabeled_data_info_for_classes = {}
    # pred_raw is only useful for RS: because we don't have mapping from label index to text
    unlabeled_data_info = {'data_index': [], 'confidence': [], 'pred': [], 'pred_raw': [], 'ground_truth': []}
    model.eval()
    for i, d in enumerate(candidate_pbar):
        with torch.no_grad():
            outputs = model(d)

        unlabeled_data_info['data_index'].extend(d['index'])
        unlabeled_data_info['confidence'].extend(outputs['confidence'])
        unlabeled_data_info['pred'].extend(outputs['pred'])

        if args['task'] == 'dst':
            unlabeled_data_info['ground_truth'].extend(d[task_to_target[args['task_name']]].detach().cpu().tolist())
        if args['task_name'] == 'intent':
            unlabeled_data_info['ground_truth'].extend(d['intent'].detach().cpu().tolist())
        if args['task_name'] == 'rs':
            unlabeled_data_info['ground_truth'].extend(d['response_plain'])
        if args['task_name'] == 'sysact':
            unlabeled_data_info['ground_truth'].extend(d['sysact'].bool().detach().cpu().tolist())

        if args['task_name'] == 'rs':
            label_name = 'response_plain'
            text_labels = []
            for pred in outputs['pred']:
                batch_size = len(d[label_name])
                if batch_size < args['eval_batch_size']:
                    for ind in range(batch_size):
                        valid_pred_ind = pred[-1 * (ind + 1)]
                        if valid_pred_ind < batch_size:
                            text_labels.append(d[label_name][valid_pred_ind].replace("{} ".format(args['sys_token']),
                                                                                     ""))  # append the text of top prediction
                            break
                else:
                    text_labels.append(d[label_name][pred[-1]].replace("{} ".format(args['sys_token']),
                                                                       ""))  # append the text of top prediction
            unlabeled_data_info['pred_raw'].extend(text_labels)

        if ("oos_intent" in args["dataset"]) and args['confidence_selection']:
            for j, label in enumerate(outputs['pred']):
                if label not in unlabeled_data_info_for_classes.keys():
                    unlabeled_data_info_for_classes[label] = {'data_index': [], 'confidence': [], 'pred': [],
                                                              'ground_truth': []}

                unlabeled_data_info_for_classes[label]['data_index'].append(d['index'][j])
                unlabeled_data_info_for_classes[label]['confidence'].append(outputs['confidence'][j])
                unlabeled_data_info_for_classes[label]['pred'].append(outputs['pred'][j])
                unlabeled_data_info_for_classes[label]['ground_truth'].append(d['intent'].detach().cpu().tolist()[j])

        candidate_pbar.set_description("Update pesudo labels on remaining dataset")

    # selection based on confidence
    if args['confidence_selection']:
        if ("oos_intent" in args["dataset"]):
            nb_train_sample_per_class = int(args['new_candidate_num'] /
                                            len(list(unlabeled_data_info_for_classes.keys()))
                                            )
            selected_info = {'index': [], 'confidence': [], 'pred': [], 'correct_prediction': []}
            for label in list(unlabeled_data_info_for_classes.keys()):
                selected_args = np.argsort(unlabeled_data_info_for_classes[label]['confidence'])[
                                -nb_train_sample_per_class:]
                selected_info['index'].extend(
                    [unlabeled_data_info_for_classes[label]['data_index'][arg] for arg in selected_args])
                selected_info['pred'].extend(
                    [unlabeled_data_info_for_classes[label]['pred'][arg] for arg in selected_args])
                selected_info['confidence'].extend(
                    [unlabeled_data_info_for_classes[label]['confidence'][arg] for arg in selected_args])
                selected_info['correct_prediction'].extend(
                    [unlabeled_data_info_for_classes[label]['pred'][arg] ==
                     unlabeled_data_info_for_classes[label]['ground_truth'][arg] for arg in selected_args])

        else:
            unlabeled_numb = int(args['new_candidate_num'])
            selected_args = np.argsort(unlabeled_data_info['confidence'])[-unlabeled_numb:]

            if args['task_name'] == 'rs':
                selected_info = {
                    'index': [unlabeled_data_info['data_index'][arg] for arg in selected_args],
                    'pred': [unlabeled_data_info['pred'][arg] for arg in selected_args],
                    'pred_raw': [unlabeled_data_info['pred_raw'][arg] for arg in selected_args],
                    'confidence': [unlabeled_data_info['confidence'][arg] for arg in selected_args],
                    'correct_prediction': [
                        unlabeled_data_info['pred_raw'][arg] == unlabeled_data_info['ground_truth'][arg].replace(
                            "{} ".format(args['sys_token']), "") for
                        arg in selected_args]
                }
            elif args['task'] == 'dst':
                selected_info = {
                    'index': [unlabeled_data_info['data_index'][arg] for arg in selected_args],
                    'pred': [unlabeled_data_info['pred'][arg] for arg in selected_args],
                    'confidence': [unlabeled_data_info['confidence'][arg] for arg in selected_args],
                    'correct_prediction': np.mean(
                        [
                            np.sum(
                                np.array(unlabeled_data_info['pred'][arg]) == np.array(
                                    unlabeled_data_info['ground_truth'][arg])
                            ).astype(float) / len(unlabeled_data_info['pred'][arg])
                            for arg in selected_args
                        ]
                    )
                }
            else:
                selected_info = {
                    'index': [unlabeled_data_info['data_index'][arg] for arg in selected_args],
                    'pred': [unlabeled_data_info['pred'][arg] for arg in selected_args],
                    'confidence': [unlabeled_data_info['confidence'][arg] for arg in selected_args],
                    'correct_prediction': [unlabeled_data_info['pred'][arg] == unlabeled_data_info['ground_truth'][arg]
                                           for arg in selected_args]
                }

    del unlabeled_data_info, unlabeled_data_info_for_classes

    if args["task_name"] == "dst":
        logging.info('Correct Prediction on Pesudo Label: {}'.format(selected_info['correct_prediction']))
    else:
        logging.info('Correct Prediction on Pesudo Label: {}'.format(
            sum(selected_info['correct_prediction']) / float(args['new_candidate_num'])))

    if args['task_name'] == 'rs':
        unlabeled_dataset.set_pesudo_label(selected_info['index'], selected_info['pred_raw'])
    else:
        unlabeled_dataset.set_pesudo_label(selected_info['index'], selected_info['pred'])

    # get the new subset to label in an iteration
    dataset_to_label = torch.utils.data.Subset(unlabeled_dataset,
                                               selected_info['index'])  # the subset we want to pesudo label

    # update the ind_to_conf_map
    for i, ind in enumerate(selected_info['index']):
        ind_to_conf_map[ind] = selected_info['confidence'][i]

    pseudo_label_loader = torch.utils.data.DataLoader(
        dataset=dataset_to_label,
        batch_size=args["batch_size"],
        shuffle=True,
        collate_fn=globals()[
            "collate_fn_{}_{}".format(args["task"], args["example_type"])
        ]
    )

    return pseudo_label_loader, seen_ind + selected_info['index'], ind_to_conf_map


def build_gradient_map(args,
                       model: torch.nn.Module,
                       tokenizer,
                       batch_sents,
                       labels_tensor):
    """
    The function the builds up a saliency map for input instances conditioned on the downstream task
    :param args: global configs
    :param model: current task model for evaluating gradients, i.e. F^T
    :param tokenizer: tokenizer for string input
    :param batch_sents: batch of sentences
    :param labels_tensor: tensor of labels y
    :return: list of gradient map for each sentence. each gradient map is a dictionary
    """
    if args["task_name"] == "intent":
        tokens = [torch.tensor(tokenizer.encode(sample, add_special_tokens=True)) for sample in batch_sents]
    else:
        tokens = [torch.tensor(sentence_encode(tokenizer, sample, max_length=args["max_seq_length"])) for sample in batch_sents]

    # pad
    max_len = max([len(tok) for tok in tokens])
    tokens = [F.pad(tok, [0, max_len - len(tok)], 'constant', tokenizer.pad_token_id) for tok in tokens]
    tokens = torch.stack(tokens)

    gradient_calc_input = None

    if args["task_name"] == "dst":
        gradient_calc_input = {"context": tokens.cuda(), "belief_ontology": labels_tensor}
    elif args["task_name"] == "sysact":
        gradient_calc_input = {args["input_name"]: tokens.cuda(), args["task_name"]: labels_tensor}
    elif args["task_name"] == "rs":
        gradient_calc_input = {"context": tokens.cuda(), "response": labels_tensor}
    elif args["task_name"] == "intent":
        gradient_calc_input = {args["input_name"]: tokens.cuda(), args["task_name"]: labels_tensor}
    else:
        raise NotImplementedError

    # use smooth saliency map to calculate gradient of each token
    grad_calc = SmoothGradient(
        model=model,
        tokenizer=tokenizer,
        show_progress=True,
    )

    gradient_map = grad_calc.saliency_interpret(gradient_calc_input, use_truth=True)
    return gradient_map


def mlm_augment_data(args,
                     model: torch.nn.Module,
                     original_dataset: torch.utils.data.DataLoader,
                     is_subset: bool,
                     new_subset_indices: set = None,
                     prev_augmented_data: torch.utils.data.ConcatDataset = None,
                     use_gradient: bool = False,
                     augmentation_factor: int = 3) -> torch.utils.data.ConcatDataset:
    """
    Augment the original dataset to the augmented dataset
    :param args: global configs
    :param model: teacher model F^T
    :param original_dataset: the dataset to be augmented
    :param is_subset: whether the incoming dataset is a subset or not.
                        if the incoming dataset is a labeled dataset, then is_subset is False.
                        if the incoming dataset is pseudo-labeled, then it is a subset.
    :param new_subset_indices: the indices of a given full dataset (to extract the desired subset)
    :param prev_augmented_data: the augmented data from previous iteration.
                                in this iteration, based on the previous data, augment based on `new_subset_indices`
    :param use_gradient:
    :param augmentation_factor:
    :return: a ConcatDataset which consists of a concatenation of `augmentation_factor` pieces of the modified dataset.
    """

    # prepare models and configs
    model_class, tokenizer_class, config_class = BertModel, BertTokenizer, BertConfig
    encoder = AutoModelForMaskedLM.from_pretrained(args["model_name_or_path"])
    tokenizer = tokenizer_class.from_pretrained(args["model_name_or_path"], cache_dir=args["cache_dir"])

    if args["model_name_or_path"] == "bert-base-uncased":
        softmax_mask = np.full(len(tokenizer.vocab), False)
        softmax_mask[tokenizer.all_special_ids] = True
        for k, v in tokenizer.vocab.items():
            if '[unused' in k:
                softmax_mask[v] = True

    else:
        # remove unused vocab and special ids from sampling, add two additional tokens [usr] [sys]
        softmax_mask = np.full(len(tokenizer.vocab) + 2, False)
        softmax_mask[tokenizer.all_special_ids] = True
        for k, v in tokenizer.vocab.items():
            if '[unused' in k:
                softmax_mask[v] = True
        softmax_mask[-2:] = True

    encoder.eval()

    if torch.cuda.is_available():
        encoder.cuda()

    augmented_datasets = []
    prev_subset_indices = []

    if is_subset and prev_augmented_data:
        # called until the 2nd iteration of augmenting unlabeled data
        prev_subset_indices = prev_augmented_data.datasets[0].indices
        for it in range(augmentation_factor):
            # get the full dataset (some of indices correspond to the augmented data)
            dataset_copy = copy.deepcopy(prev_augmented_data.datasets[it].dataset)
            dataloader_copy = torch.utils.data.DataLoader(
                dataset=dataset_copy,
                batch_size=original_dataset.batch_size,
                shuffle=True,
                collate_fn=globals()["collate_fn_{}_{}".format(args["task"], args["example_type"])]
            )
            augmented_datasets.append(dataloader_copy)
    else:
        # incoming data is the whole train loader with labels
        # OR
        # incoming data is the unlabeled data, and during 1st iteration of augmenting unlabeled data
        for it in range(augmentation_factor):
            dataset_copy = copy.deepcopy(original_dataset.dataset)
            dataloader_copy = torch.utils.data.DataLoader(
                dataset=dataset_copy,
                batch_size=original_dataset.batch_size,
                shuffle=True,
                collate_fn=globals()["collate_fn_{}_{}".format(args["task"], args["example_type"])]
            )
            augmented_datasets.append(dataloader_copy)

    if is_subset:
        # only iterate over `new_subset_indices`, i.e., previously augmented data are ignored
        original_dataset = torch.utils.data.DataLoader(
                dataset=torch.utils.data.Subset(original_dataset.dataset, list(new_subset_indices)),
                batch_size=original_dataset.batch_size,
                shuffle=True,
                collate_fn=globals()["collate_fn_{}_{}".format(args["task"], args["example_type"])]
        )

    original_dataset_bar = tqdm(original_dataset)

    for ind, d in enumerate(original_dataset_bar):

        num_lines, index, lines = len(d["context_plain"]),\
                                  d["index"],\
                                  [tuple(s.strip().split('\t')) for s in d["context_plain"]]

        labels = None

        if args["task_name"] == "dst":
            labels = d["belief_ontology"]
        if args["task_name"] == "intent":
            labels = d["intent_plain"]
        if args["task_name"] == "sysact":
            labels = d["sysact_plain"]
        if args["task_name"] == "rs":
            labels = d["response_plain"]

        # max_len_in_d = max(d['context_len'])
        lines = [[[s] for s in s_list] for s_list in list(zip(*lines))]

        # sentences and labels to process
        sents = []
        l = []
        # number sentences generated
        num_gen = []
        # sentence index to noise from
        gen_index = []
        # number of tries generating a new sentence
        num_tries = []
        # next sentence index to draw from
        next_sent = 0

        sents, l, next_sent, num_gen, num_tries, gen_index = \
            fill_batch(tokenizer, sents, l, lines, labels, next_sent, num_gen, num_tries, gen_index)

        gradient_map = None
        all_candidate_sents = [sent[0][0] for sent in sents]

        if use_gradient:
            if args["task_name"] == "rs":
                gradient_map = build_gradient_map(args, model, tokenizer, all_candidate_sents, d["response"])
            else:
                gradient_map = build_gradient_map(args, model, tokenizer, all_candidate_sents, d[args["task_name"]])

        while sents:
            # remove any sentences that are done generating and dump to file
            for i in range(len(num_gen))[::-1]:
                if num_gen[i] >= augmentation_factor or num_tries[i] > 5:
                    # get sent info
                    gen_sents = sents.pop(i)
                    num_gen.pop(i)
                    gen_index.pop(i)
                    label = l.pop(i)

                    # write generated sentences
                    for sample_i, sg in enumerate(gen_sents[1:]):
                        if args["task_name"] == "sysact":
                            new_sent = " ".join([repr(val)[1:-1] for val in sg])
                            augmented_datasets[sample_i].dataset.reset_dm_context(index[i], new_sent)
                        elif args["task_name"] == "dst":
                            new_sent = " ".join([repr(val)[1:-1] for val in sg])
                            augmented_datasets[sample_i].dataset.reset_dst_context(index[i], new_sent)
                        elif args["task_name"] == "intent":
                            new_sent = " ".join([repr(val)[1:-1] for val in sg])
                            # skip [cls] [sys] [usr]
                            if args["model_name_or_path"] == "bert-base-uncased":
                                augmented_datasets[sample_i].dataset.reset_nlu_context(index[i], new_sent)
                            else:
                                new_sent = " ".join(new_sent.split(" ")[3:])
                                augmented_datasets[sample_i].dataset.data["turn_usr"][index[i]] = new_sent
                        elif args["task_name"] == "rs":
                            new_sent = " ".join([repr(val)[1:-1] for val in sg])
                            augmented_datasets[sample_i].dataset.reset_nlg_context(index[i], new_sent)
                        else:
                            raise NotImplementedError


            # fill batch
            sents, l, next_sent, num_gen, num_tries, gen_index = \
                fill_batch(tokenizer, sents, l, lines, labels, next_sent, num_gen, num_tries, gen_index)

            # break if done dumping
            if len(sents) == 0:
                break

            # build batch
            toks = []
            masks = []

            for i in range(len(gen_index)):
                s = sents[i][gen_index[i]]
                tok, mask = hf_masked_encode(
                    args,
                    tokenizer,
                    *s,
                    noise_prob=0.15,
                    random_token_prob=0.1,
                    leave_unmasked_prob=0.1,
                    use_gradient=use_gradient,
                    gradient_map=gradient_map,
                    ind_of_gradient_map=i
                )
                toks.append(tok)
                masks.append(mask)

            # pad up to max len input
            max_len = max([len(tok) for tok in toks])
            pad_tok = tokenizer.pad_token_id

            toks = [F.pad(tok, [0, max_len - len(tok)], 'constant', pad_tok) for tok in toks]
            masks = [F.pad(mask, [0, max_len - len(mask)], 'constant', pad_tok) for mask in masks]
            toks = torch.stack(toks)
            masks = torch.stack(masks)

            # load to GPU if available
            if torch.cuda.is_available():
                toks = toks.cuda()
                masks = masks.cuda()

            # predict reconstruction
            rec, rec_masks = hf_reconstruction_prob_tok(toks, masks, tokenizer, encoder, softmax_mask,
                                                        reconstruct=True, topk=10)

            # decode reconstructions and append to lists
            for i in range(len(rec)):
                rec_work = rec[i].cpu().tolist()
                if args["task_name"] == "intent":
                    s_rec = [s.strip() for s in
                             tokenizer.decode([val for val in rec_work if val != tokenizer.pad_token_id][1:-1]).split(
                                 tokenizer.sep_token)]
                    s_rec = tuple(s_rec)
                else:
                    s_rec = [s.strip() for s in
                             tokenizer.decode([val for val in rec_work if (val != tokenizer.pad_token_id
                                                                           and val != tokenizer.cls_token_id)]).split(
                                 tokenizer.sep_token)]

                # rejoining sentences that are separated by [sep]
                if args["task_name"] == "dst":
                    s_rec = tuple([" [sep] ".join(s_rec)])
                elif args["task_name"] == "sysact":
                    s_rec = tuple([" [sep] ".join(s_rec)])
                elif args["task_name"] == "rs":
                    s_rec = tuple(s_rec)

                # check if identical reconstruction or empty
                # s_rec not in sents[i] and
                if '' not in s_rec:
                    sents[i].append(s_rec)
                    num_gen[i] += 1
                    num_tries[i] = 0
                    gen_index[i] = 0

                # otherwise try next sentence
                else:
                    num_tries[i] += 1
                    gen_index[i] += 1
                    if gen_index[i] == len(sents[i]):
                        gen_index[i] = 0

            # clean up tensors
            del toks
            del masks

        original_dataset_bar.set_description("augmenting training data, using gradient: " + str(use_gradient))

    if is_subset:
        all_indices = prev_subset_indices + list(new_subset_indices)
        return torch.utils.data.ConcatDataset([torch.utils.data.Subset(augmented_datasets[i].dataset, all_indices)
                                               for i in range(len(augmented_datasets))])
    else:
        return torch.utils.data.ConcatDataset([augmented_datasets[i].dataset for i in range(len(augmented_datasets))])


def fill_batch(tokenizer,
               sents,
               l,
               lines,
               labels,
               next_sent,
               num_gen,
               num_tries,
               gen_index):
    # search for the next valid sentence
    while True:
        while True:
            if next_sent >= len(lines[0]):
                break

            next_sents = [s_list[next_sent][0] for s_list in lines]
            next_len = len(tokenizer.encode(*next_sents))

            # skip input if too short or long
            if 2 < next_len < 1024:
                break
            next_sent += 1

        # add it to our lists
        if next_sent < len(lines[0]):
            next_sent_lists = [s_list[next_sent] for s_list in lines]
            sents.append(list(zip(*next_sent_lists)))
            l.append(labels[next_sent])

            num_gen.append(0)
            num_tries.append(0)
            gen_index.append(0)
            next_sent += 1
        else:
            break

    return sents, l, next_sent, num_gen, num_tries, gen_index


def sentence_encode(tokenizer, sentence, *addl_sentences, max_length):
    """
    Encode a sentence of tokens to token ids 
    """
    tokens = tokenizer.tokenize(tokenizer.cls_token) + tokenizer.tokenize(sentence, *addl_sentences)[-max_length+1:]
    story = tokenizer.convert_tokens_to_ids(tokens)
    return story


def hf_masked_encode(args,
                     tokenizer,
                     sentence: str,
                     *addl_sentences,
                     noise_prob=0.0,
                     random_token_prob=0.0,
                     leave_unmasked_prob=0.0,
                     use_gradient: bool = False,
                     gradient_map: list = None,
                     ind_of_gradient_map: int):
    if random_token_prob > 0.0:
        # add two special token for [usr] and [sys]
        if args["model_name_or_path"] == "bert-base-uncased":
            weights = np.ones(len(tokenizer.vocab))
            weights[tokenizer.all_special_ids] = 0
            for k, v in tokenizer.vocab.items():
                if '[unused' in k:
                    weights[v] = 0
            weights = weights / weights.sum()
        else:
            weights = np.ones(len(tokenizer.vocab) + 2)
            weights[tokenizer.all_special_ids] = 0
            weights[-2:] = 0
            for k, v in tokenizer.vocab.items():
                if '[unused' in k:
                    weights[v] = 0
            weights = weights / weights.sum()

    if args["task_name"] == "intent":
        tokens = np.asarray(tokenizer.encode(sentence, *addl_sentences, add_special_tokens=True))
    else:
        tokens = np.asarray(sentence_encode(tokenizer, sentence, *addl_sentences, max_length=args["max_seq_length"]))

    if noise_prob == 0.0:
        return tokens

    sz = len(tokens)
    mask = np.full(sz, False)
    num_mask = int(noise_prob * sz + np.random.rand())

    mask_choice_p = np.ones(sz)
    for i in range(sz):
        if tokens[i] in [tokenizer.sep_token_id, tokenizer.cls_token_id, tokenizer.pad_token_id]:
            mask_choice_p[i] = 0
        # add two special token for [usr] and [sys]
        if args["model_name_or_path"] != "bert-base-uncased" and tokens[i] in list(tokenizer.added_tokens_encoder.values()):
            mask_choice_p[i] = 0
    mask_choice_p = mask_choice_p / mask_choice_p.sum()

    if use_gradient:
        # dict_keys(['tokens', 'grad', 'label', 'prob'])
        curr_grad = np.array(gradient_map[ind_of_gradient_map]['grad'])
        if mask_choice_p.shape == curr_grad.shape:
            eps = np.random.rand() / 20
            # scale the gradient to [0, 1]
            curr_grad = (curr_grad - np.min(curr_grad) + eps) / (np.ptp(curr_grad) + eps)
            curr_grad = 1 / curr_grad
            mask_choice_p = mask_choice_p * curr_grad

    mask_choice_p = mask_choice_p / mask_choice_p.sum()
    mask[np.random.choice(sz, num_mask, replace=False, p=mask_choice_p)] = True

    mask_targets = np.full(len(mask), tokenizer.pad_token_id)
    mask_targets[mask] = tokens[mask == 1]

    # decide unmasking and random replacement
    rand_or_unmask_prob = random_token_prob + leave_unmasked_prob
    if rand_or_unmask_prob > 0.0:
        rand_or_unmask = mask & (np.random.rand(sz) < rand_or_unmask_prob)
        if random_token_prob == 0.0:
            unmask = rand_or_unmask
            rand_mask = None
        elif leave_unmasked_prob == 0.0:
            unmask = None
            rand_mask = rand_or_unmask
        else:
            unmask_prob = leave_unmasked_prob / rand_or_unmask_prob
            decision = np.random.rand(sz) < unmask_prob
            unmask = rand_or_unmask & decision
            rand_mask = rand_or_unmask & (~decision)
    else:
        unmask = rand_mask = None

    if unmask is not None:
        mask = mask ^ unmask

    tokens[mask] = tokenizer.mask_token_id
    if rand_mask is not None:
        num_rand = rand_mask.sum()
        if num_rand > 0:
            if args["model_name_or_path"] == "bert-base-uncased":
                tokens[rand_mask] = np.random.choice(
                    len(tokenizer.vocab),
                    num_rand,
                    p=weights,
                )
            else:
                tokens[rand_mask] = np.random.choice(
                    len(tokenizer.vocab) + 2,
                    num_rand,
                    p=weights,
                )

    return torch.tensor(tokens).long(), torch.tensor(mask).long()


def hf_reconstruction_prob_tok(masked_tokens,
                               target_tokens,
                               tokenizer,
                               model,
                               softmax_mask,
                               reconstruct=False,
                               topk=1):

    model.eval()

    if torch.cuda.is_available():
        model.cuda()

    single = False

    # expand batch size 1
    if masked_tokens.dim() == 1:
        single = True
        masked_tokens = masked_tokens.unsqueeze(0)
        target_tokens = target_tokens.unsqueeze(0)

    masked_fill = torch.ones_like(masked_tokens)

    masked_index = (target_tokens != tokenizer.pad_token_id).nonzero(as_tuple=True)
    masked_orig_index = target_tokens[masked_index]

    # edge case of no masked tokens
    if len(masked_orig_index) == 0:
        if reconstruct:
            return masked_tokens, masked_fill
        else:
            return 1.0

    masked_orig_enum = [list(range(len(masked_orig_index))), masked_orig_index]

    outputs = model(
        masked_tokens.long().to(device=next(model.parameters()).device),
        masked_lm_labels=target_tokens
    )

    # b * max_len * vocab_size (30524)
    features = outputs[1]

    logits = features[masked_index]
    for l in logits:
        l[softmax_mask] = float('-inf')
    probs = logits.softmax(dim=-1)

    if reconstruct:
        # sample from topk
        if topk != -1:
            values, indices = probs.topk(k=topk, dim=-1)
            kprobs = values.softmax(dim=-1)
            if len(masked_index) > 1:
                samples = torch.cat([idx[torch.multinomial(kprob, 1)] for kprob, idx in zip(kprobs, indices)])
            else:
                samples = indices[torch.multinomial(kprobs, 1)]

        # unrestricted sampling
        else:
            if len(masked_index) > 1:
                samples = torch.cat([torch.multinomial(prob, 1) for prob in probs])
            else:
                samples = torch.multinomial(probs, 1)

        # set samples
        masked_tokens[masked_index] = samples
        masked_fill[masked_index] = samples

        if single:
            return masked_tokens[0], masked_fill[0]
        else:
            return masked_tokens, masked_fill

    return torch.sum(torch.log(probs[masked_orig_enum])).item()

