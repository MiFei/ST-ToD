from tqdm import tqdm
import torch.nn as nn
import logging
import ast
import glob
import numpy as np
import copy
import torch
import torch.utils.data as data
import datetime

# utils 
from utils.config import *
from utils.utils_general import *
from utils.utils_multiwoz import *
from utils.utils_oos_intent import *
from utils.utils_universal_act import *

# models
from models.multi_label_classifier import *
from models.multi_class_classifier import *
from models.BERT_DST_Picklist import *
from models.dual_encoder_ranking import *

# hugging face models
from transformers import *

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

## model selection
MODELS = {"bert": (BertModel,       BertTokenizer,       BertConfig),
          "todbert": (BertModel,       BertTokenizer,       BertConfig)
          }

start_time = datetime.datetime.now()

# Fix torch random seed #! not necessarily consistent with random seeds in multiple runs
torch.manual_seed(SEEDS[0])
args['n_gpu'] = len(args['gpu'].split(','))

# Reading data and create data loaders
datasets = {}
ds_name = args["dataset"]
data_trn, data_dev, data_tst, data_meta = globals()["prepare_data_{}".format(ds_name)](args)
datasets[ds_name] = {"train": data_trn, "dev":data_dev, "test": data_tst, "meta":data_meta}
unified_meta = get_unified_meta(datasets)  
if "resp_cand_trn" not in unified_meta.keys(): 
    unified_meta["resp_cand_trn"] = {}
args["unified_meta"] = unified_meta 

# Create vocab and model class
args["model_type"] = args["model_type"].lower()
model_class, tokenizer_class, config_class = MODELS[args["model_type"]]
tokenizer = tokenizer_class.from_pretrained(args["model_name_or_path"], cache_dir=args["cache_dir"])
args["model_class"] = model_class
args["tokenizer"] = tokenizer
if args["model_name_or_path"]:
    config = config_class.from_pretrained(args["model_name_or_path"], cache_dir=args["cache_dir"]) 
else:
    config = config_class()
args["config"] = config
args["num_labels"] = unified_meta["num_labels"]

# Training and Testing Loop
if args["do_train"]:
    result_runs = []
    output_dir_origin = str(args["output_dir"])
    
    # Setup logger
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=os.path.join(args["output_dir"], "train.log"),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    # training loop
    for run in range(args["nb_runs"]):
         
        # Setup random seed and output dir
        rand_seed = SEEDS[run]
        args["rand_seed"] = rand_seed
        np.random.seed(rand_seed)

        torch.manual_seed(rand_seed)
        torch.cuda.manual_seed(rand_seed)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True 

        args["output_dir"] = os.path.join(output_dir_origin, "run{}".format(run))
        os.makedirs(args["output_dir"], exist_ok=False)
        logging.info("Running Random Seed: {}".format(rand_seed))
        
        # Loading model
        model = globals()[args['my_model']](args)
        if torch.cuda.is_available():
            model = model.cuda()
            if args['n_gpu'] > 1:
                model = nn.DataParallel(model)
        
        # Create Dataloader
        trn_loader = get_loader(args, "train", tokenizer, datasets, unified_meta)
        dev_loader = get_loader(args, "dev", tokenizer, datasets, unified_meta, shuffle=args["task_name"]=="rs")
        tst_loader = get_loader(args, "test", tokenizer, datasets, unified_meta, shuffle=args["task_name"]=="rs")
        unlabeled_dataset = get_loader(args, "train", tokenizer, datasets, unified_meta, for_unlabeled=True).dataset

        augmented_trn = None
        augmented_pseudo = None

        # Create TF Writer
        tb_writer = SummaryWriter(comment=args["output_dir"].replace("/", "-"))

        # Start training process with early stopping
        loss_best, acc_best, cnt, train_step = 1e10, -1, 0, 0
        global_loss_best, global_acc_best, global_cnt, global_train_step = 1e10, -1, 0, 0

        seen_ind = []
        ind_to_conf_map = {}
        for itr in range(args['iterations']):

            logging.info("Iteration:{}".format(itr))
            current_time = datetime.datetime.now()
            logging.info("Accumulated time cost: {}".format(str(current_time - start_time)))
            
            # reset for each iteration
            loss_best, acc_best, cnt, train_step = 1e10, -1, 0, 0

            if itr == 0:
                self_train_loader = trn_loader

            else:
                # augment the original labeled data
                if itr == 1:
                    augmented_trn = mlm_augment_data(args=args,
                                                     model=model,
                                                     original_dataset=trn_loader,
                                                     is_subset=False,
                                                     use_gradient=True)

                # using the best model in last iteration to pseudo-label the unlabeled data
                prev_seen_ind = set(copy.deepcopy(seen_ind))
                pseudo_label_loader, seen_ind, ind_to_conf_map = get_pseudo_label_loader(
                    args=args,
                    model=model,
                    unlabeled_dataset=unlabeled_dataset,
                    seen_ind=seen_ind,
                    ind_to_conf_map=ind_to_conf_map
                )
                new_seen_ind = set(copy.deepcopy(seen_ind)).difference(prev_seen_ind)

                unlabeled_dataset_loader = torch.utils.data.DataLoader(
                    dataset=unlabeled_dataset,
                    batch_size=trn_loader.batch_size,
                    shuffle=True,
                    collate_fn=trn_loader.collate_fn
                )

                # get the augmented pseudo-labeled data
                augmented_pseudo = mlm_augment_data(args=args,
                                                    model=model,
                                                    original_dataset=unlabeled_dataset_loader,
                                                    is_subset=True,
                                                    new_subset_indices=new_seen_ind,
                                                    prev_augmented_data=augmented_pseudo,
                                                    use_gradient=True)

                # current labeled dataset: trn_loader + unlabeled dataset with pseudo labels till current iteration
                current_labeled_dataset = merge_loaders(
                    task=args['task'],
                    dataloaders=[trn_loader, pseudo_label_loader],
                    seen_ind=seen_ind
                )

                # consist of four parts, (trn_loader, pseudo_labeled_data), augmented_trn, augmented_pseudo
                self_train_loader = torch.utils.data.DataLoader(
                    dataset=torch.utils.data.ConcatDataset([current_labeled_dataset.dataset,
                                                            augmented_trn,
                                                            augmented_pseudo]),
                    batch_size=trn_loader.batch_size,
                    shuffle=True,
                    collate_fn=trn_loader.collate_fn
                )

            model = globals()[args['my_model']](args)
            if torch.cuda.is_available():
                model = model.cuda()

            try:
                for epoch in range(args["epoch"]):
                    logging.info("Iteration:{}, Epoch:{}".format(itr, epoch))
                    self_train_pbar = tqdm(self_train_loader)
                    train_loss = 0

                    model.train()

                    for i, d in enumerate(self_train_pbar):

                        if itr > 0 and args['confidence_weighting']:
                            outputs = model(d, ind_to_conf_map)
                        else:
                            outputs = model(d)
                        train_loss += outputs["loss"]
                        train_step += 1
                        self_train_pbar.set_description("Training Loss: {:.4f}".format(train_loss / (i + 1)))

                    # evaluation
                    if (epoch + 1) % args["eval_by_epoch"] == 0:
                        model.eval()
                        dev_loss = 0
                        preds, labels = [], []
                        ppbar = tqdm(dev_loader)
                        for d in ppbar:
                            with torch.no_grad():
                                outputs = model(d)
                            dev_loss += outputs["loss"]
                            preds += [item for item in outputs["pred"]]
                            labels += [item for item in outputs["label"]]
                            ppbar.set_description("Evaluation on validation set: ")

                        dev_loss = dev_loss / len(dev_loader)
                        results = model.evaluation(preds, labels)
                        dev_acc = results[args["earlystop"]] if args["earlystop"] != "loss" else dev_loss

                        # write to tensorboard
                        tb_writer.add_scalar("train_loss", train_loss / (i + 1), train_step)
                        tb_writer.add_scalar("eval_loss", dev_loss, train_step)
                        tb_writer.add_scalar("eval_{}".format(args["earlystop"]), dev_acc, train_step)

                        if (dev_loss < loss_best and args["earlystop"] == "loss") or \
                                (dev_acc > acc_best and args["earlystop"] != "loss"):
                            loss_best = dev_loss
                            acc_best = dev_acc
                            cnt = 0  # reset

                            if args["not_save_model"]:
                                model_clone = globals()[args['my_model']](args)
                                model_clone.load_state_dict(copy.deepcopy(model.state_dict()))
                            else:
                                output_model_file = os.path.join(args["output_dir"], "pytorch_model.bin")
                                print(output_model_file)
                                if args["n_gpu"] == 1:
                                    torch.save(model.state_dict(), output_model_file)
                                else:
                                    torch.save(model.module.state_dict(), output_model_file)
                                logging.info("[Info] Model saved at epoch {} step {}".format(epoch, train_step))

                        else:
                            cnt += 1
                            logging.info("[Info] Early stop count: {}/{}...".format(cnt, args["patience"]))

                        logging.info("Trn loss {:.4f}, Dev loss {:.4f}, Dev {} {:.4f}".format(train_loss / (i + 1),
                                                                                              dev_loss,
                                                                                              args["earlystop"],
                                                                                              dev_acc))

                    if cnt > args["patience"]:
                        tb_writer.close()
                        logging.info("Ran out of patient, early stop...")
                        break

            except KeyboardInterrupt:
                logging.info("[Warning] Earlystop by KeyboardInterrupt")

            # test after each iteration
            # Load the best model
            if args["not_save_model"]:
                model.load_state_dict(copy.deepcopy(model_clone.state_dict()))
            else:
                # Start evaluating on the test set
                if torch.cuda.is_available():
                    model.load_state_dict(torch.load(output_model_file))
                else:
                    model.load_state_dict(torch.load(output_model_file, lambda storage, loc: storage))

            # Run test set evaluation
            pbar = tqdm(tst_loader)
            for nb_eval in range(args["nb_evals"]):
                test_loss = 0
                preds, labels = [], []
                model.eval()
                for d in pbar:
                    with torch.no_grad():
                        outputs = model(d)
                    test_loss += outputs["loss"]
                    preds += [item for item in outputs["pred"]]
                    labels += [item for item in outputs["label"]]
                    pbar.set_description("Evaluation on test set: ")

                test_loss = test_loss / len(tst_loader)
                results = model.evaluation(preds, labels)
                result_runs.append(results)
                logging.info("Test Results at Iteration {}: {} ".format(itr, str(results)))
            
            if global_acc_best < acc_best
                global_acc_best = acc_best
                logging.info("[Info] Global best result {} at iteration {}".format(global_acc_best, itr))
                global_cnt = 0
            else:
                logging.info("[Info] Global early stop count: {}".format(global_cnt))
                global_cnt += 1
            
            if global_cnt > args["patience"]:
                logging.info("Ran out of patient, early stop on global iteration...")
                break
    
    # Average results over runs
    if args["nb_runs"] > 1:
        f_out = open(os.path.join(output_dir_origin, "eval_results_multi-runs.txt"), "w")
        f_out.write("Average over {} runs and {} evals \n".format(args["nb_runs"], args["nb_evals"]))
        for key in results.keys():
            mean = np.mean([r[key] for r in result_runs])
            std  = np.std([r[key] for r in result_runs])
            f_out.write("{}: mean {} std {} \n".format(key, mean, std))
        f_out.close()

else:
    # Load Model
    print("[Info] Loading model from {}".format(args['my_model']))
    model = globals()[args['my_model']](args)    
    if args["load_path"]:
        print("MODEL {} LOADED".format(args["load_path"]))
        if torch.cuda.is_available(): 
            model.load_state_dict(torch.load(args["load_path"]))
        else:
            model.load_state_dict(torch.load(args["load_path"], lambda storage, loc: storage))
    if torch.cuda.is_available():
        model = model.cuda()
    
    print("[Info] Start Evaluation on dev and test set...")
    dev_loader = get_loader(args, "dev"  , tokenizer, datasets, unified_meta)
    tst_loader = get_loader(args, "test" , tokenizer, datasets, unified_meta, shuffle=args["task_name"]=="rs")
    model.eval()
    
    for d_eval in ["tst"]: #["dev", "tst"]:
        f_w = open(os.path.join(args["output_dir"], "{}_results.txt".format(d_eval)), "w")

        # Start evaluating on the test set
        test_loss = 0
        preds, labels = [], []
        pbar = tqdm(locals()["{}_loader".format(d_eval)])
        for d in pbar:
            with torch.no_grad():
                outputs = model(d)
            test_loss += outputs["loss"]
            preds += [item for item in outputs["pred"]]
            labels += [item for item in outputs["label"]] 

        test_loss = test_loss / len(tst_loader)
        results = model.evaluation(preds, labels)
        print("{} Results: {}".format(d_eval, str(results)))
        f_w.write(str(results))
        f_w.close()
