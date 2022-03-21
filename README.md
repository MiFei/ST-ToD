# Self-training Improves Pre-training for Few-shot Learning in Task-oriented Dialog Systems


## Introduction

This code repo is for the paper called ["Self-training Improves Pre-training for Few-shot Learning in Task-oriented Dialog Systems"](https://aclanthology.org/2021.emnlp-main.142/) presented at EMNLP 2021 (Oral)

As the labeling cost for different modules in task-oriented dialog (ToD) systems is expensive, a major challenge is to train different modules with the least amount of labeled data. Recently, large-scale pre-trained language models, such as BERT and GPT-2, have shown promising results for few-shot learning in ToD. In this paper, we devise a self-training approach to utilize the abundant unlabeled dialog data to further improve state-of-the-art pre-trained models in few-shot learning scenarios.
Specifically, we propose a self-training approach which iteratively labels the most confident unlabeled data to train a stronger ***Student*** model. Moreover, a new text augmentation technique (GradAug) is proposed to better train the ***Student*** by replacing non-crucial tokens using a masked language model.
We conduct extensive experiments and present analysis on four downstream tasks in ToD,  including intent classification, dialog state tracking, dialog act prediction, and response selection. Empirical results demonstrate that the proposed self-training approach consistently improves state-of-the-art pre-trained models (BERT, ToD-BERT) when only a small number of labeled data are available.



## File Orgnanization

```
.
└── models
    └── multi_class_classifier.py
    └── multi_label_classifier.py
    └── BERT_DST_Picklist.py
    └── dual_encoder_ranking.py
└── utils.py
    └── utils_general.py
    └── Interpret
        └── saliency_interpreter.py
        └── smooth_gradient.py
        └── vanilla_gradient.py
    └── multiwoz
        └── ...
    └── metrics
        └── ...
    └── loss_function
        └── ...
    └── dataloader_nlu.py
    └── dataloader_dst.py
    └── dataloader_dm.py
    └── dataloader_nlg.py
    └── dataloader_usdl.py
    └── ...
└── README.md
└── requirements.txt
└── evaluation_ratio_pipeline.sh
└── main_st.py
```

Some key files that are relevant to our Self-Training algorithm:

- ```main_st.py```: the main loop that is used to execute our Algorithm 1.
- ```utils/utils_general.py```: contains helper functions that pseudo-label the data and augment text
- ```Interpret```: contains helper functions that  calculate the vanilla saliency map and smooth saliency map



## Environment

We use implement the algorithm and test using python 3.6.12. Dependencies are given in ```requirements.txt```.



## Dataset

Datasets for downstream tasks can be retrieved [here](https://drive.google.com/file/d/1EnGX0UF4KW6rVBKMF3fL-9Q2ZyFKNOIy/view?usp=sharing).



## Running Four Downstream Tasks

The detailed script of all experiments in Tables 1, 2, 3 and 4 with pre-configured hyper-parameters are given in the script: ```evaluation_ratio_pipeline.sh```.

For example:

```bash
./evaluation_ratio_pipeline.sh 0 bert bert-base-uncased save/BERT --nb_runs=3
./evaluation_ratio_pipeline.sh 0 todbert TODBERT/TOD-BERT-JNT-V1 save/TOD-BERT-JNT-V1 --nb_runs=3

```

Two types of Bert are tested:

- bert-base-uncased
- TODBERT/TOD-BERT-JNT-V1

To run only a part of the experiments, comment out irrelevant experiments in ```evaluation_ratio_pipeline.sh```.


## Citation
```[]
@inproceedings{mi2021self,
  title={Self-training Improves Pre-training for Few-shot Learning in Task-oriented Dialog Systems},
  author={Mi, Fei and Zhou, Wanhao and Kong, Lingjing and Cai, Fengyu and Huang, Minlie and Faltings, Boi},
  booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing},
  pages={1887--1898},
  year={2021}
}
```

## Credit

This code repository is based on:

- Chien-Sheng Wu's ToD-Bert implementation: https://github.com/jasonwu0731/ToD-BERT
- The Interpret module that calculates the saliency map: https://github.com/koren-v/Interpret
- SSMBA, a masking-reconstruction framework to augment text data: https://github.com/nng555/ssmba