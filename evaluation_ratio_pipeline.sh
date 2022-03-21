gpu=$1
model=$2
bert_dir=$3
output_dir=$4
iterations=50
epoch=300
lambda=1
add1=$5
add2=$6
add3=$7

# example use:
# ./evaluation_ratio_pipeline.sh 0 bert bert-base-uncased save/BERT --nb_runs=3
# ./evaluation_ratio_pipeline.sh 0 todbert TODBERT/TOD-BERT-JNT-V1 save/TOD-BERT-JNT-V1 --nb_runs=3

#==============Intent Classification ratio=0.01===============
ratio=0.01
CUDA_VISIBLE_DEVICES=$gpu python main_st.py \
   --my_model=multi_class_classifier \
   --dataset=oos_intent \
   --task_name="intent" \
   --earlystop="acc" \
   --output_dir=${output_dir}/Intent/OOS-Ratio/R${ratio} \
   --do_train \
   --task=nlu \
   --example_type=turn \
   --model_type=${model} \
   --model_name_or_path=${bert_dir} \
   --batch_size=16 \
   --usr_token=[USR] --sys_token=[SYS] \
   --epoch=${epoch} \
   --train_data_ratio=${ratio} \
   --new_candidate_num=500 \
   --iterations=${iterations} \
   --eval_by_epoch=2
   --patience=10
   $add1 $add2 $add3

#==============Intent Classification ratio=0.1===============
ratio=0.1
CUDA_VISIBLE_DEVICES=$gpu python main_st.py \
   --my_model=multi_class_classifier \
   --dataset=oos_intent \
   --task_name="intent" \
   --earlystop="acc" \
   --output_dir=${output_dir}/Intent/OOS-Ratio/R${ratio} \
   --do_train \
   --task=nlu \
   --example_type=turn \
   --model_type=${model} \
   --model_name_or_path=${bert_dir} \
   --batch_size=16 \
   --usr_token=[USR] --sys_token=[SYS] \
   --epoch=${epoch} \
   --train_data_ratio=${ratio} \
   --new_candidate_num=1510 \
   --iterations=${iterations} \
   --eval_by_epoch=2 \
   --patience=10 \
   $add1 $add2 $add3

#==============Dialog State Tracking ratio=0.01===============
ratio=0.01
CUDA_VISIBLE_DEVICES=$gpu python main_st.py \
   --my_model=BeliefTracker \
   --model_type=${model} \
   --dataset=multiwoz \
   --task_name="dst" \
   --earlystop="joint_acc" \
   --output_dir=${output_dir}/DST/MWOZ-Ratio/R${ratio} \
   --do_train \
   --task=dst \
   --example_type=turn \
   --model_name_or_path=${bert_dir} \
   --batch_size=8 --eval_batch_size=8 \
   --usr_token=[USR] --sys_token=[SYS] \
   --train_data_ratio=${ratio} \
   --new_candidate_num=200 \
   --eval_by_epoch=3 \
   --patience=10 \
   $add1 $add2 $add3

#==============Dialog State Tracking ratio=0.1===============
ratio=0.1
CUDA_VISIBLE_DEVICES=$gpu python main_st.py \
   --my_model=BeliefTracker \
   --model_type=${model} \
   --dataset=multiwoz \
   --task_name="dst" \
   --earlystop="joint_acc" \
   --output_dir=${output_dir}/DST/MWOZ-Ratio/R${ratio} \
   --do_train \
   --task=dst \
   --example_type=turn \
   --model_name_or_path=${bert_dir} \
   --batch_size=8 --eval_batch_size=8 \
   --usr_token=[USR] --sys_token=[SYS] \
   --train_data_ratio=${ratio} \
   --new_candidate_num=2000 \
   --eval_by_epoch=3 \
   --patience=10 \
   $add1 $add2 $add3

#==============Dialog Act Prediction MWOZ ratio=0.01===============
ratio=0.01
CUDA_VISIBLE_DEVICES=$gpu python main_st.py \
   --my_model=multi_label_classifier \
   --do_train --dataset=multiwoz \
   --task=dm --task_name=sysact --example_type=turn \
   --model_type=${model} \
   --model_name_or_path=${bert_dir} \
   --output_dir=${output_dir}/DA/MWOZ-Ratio/R${ratio} \
   --batch_size=8 \
   --eval_batch_size=8 \
   --learning_rate=5e-5 \
   --usr_token=[USR] --sys_token=[SYS] \
   --earlystop=f1_weighted \
   --train_data_ratio=${ratio} \
   --epoch=${epoch} \
   --new_candidate_num=500 \
   --iterations=${iterations} \
   --eval_by_epoch=2 \
   --patience=10 \
   $add1 $add2 $add3

#==============Dialog Act Prediction MWOZ ratio=0.1===============
ratio=0.1
CUDA_VISIBLE_DEVICES=$gpu python main_st.py \
   --my_model=multi_label_classifier \
   --do_train --dataset=multiwoz \
   --task=dm --task_name=sysact --example_type=turn \
   --model_type=${model} \
   --model_name_or_path=${bert_dir} \
   --output_dir=${output_dir}/DA/MWOZ-Ratio/R${ratio} \
   --batch_size=8 \
   --eval_batch_size=8 \
   --learning_rate=5e-5 \
   --usr_token=[USR] --sys_token=[SYS] \
   --earlystop=f1_weighted \
   --train_data_ratio=${ratio} \
   --epoch=${epoch} \
   --new_candidate_num=2000 \
   --iterations=${iterations} \
   --eval_by_epoch=2 \
   --patience=10 \
   $add1 $add2 $add3

#==============Dialog Act Prediction DSTC2 ratio=0.01===============
ratio=0.01
CUDA_VISIBLE_DEVICES=$gpu python main_st.py \
   --my_model=multi_label_classifier \
   --do_train \
   --dataset=universal_act_dstc2 \
   --task=dm --task_name=sysact --example_type=turn \
   --model_type=${model} --model_name_or_path=${bert_dir} \
   --output_dir=${output_dir}/DA/DSTC2-Ratio/R${ratio} \
   --batch_size=8 \
   --eval_batch_size=8 \
   --learning_rate=5e-5 \
   --usr_token=[USR] --sys_token=[SYS] \
   --earlystop=f1_weighted \
   --train_data_ratio=${ratio} \
   --epoch=${epoch} \
   --new_candidate_num=200 \
   --iterations=${iterations} \
   --eval_by_epoch=2 \
   --patience=10 \
   $add1 $add2 $add3


#==============Dialog Act Prediction DSTC2 ratio=0.1===============
ratio=0.1
CUDA_VISIBLE_DEVICES=$gpu python main_st.py \
   --my_model=multi_label_classifier \
   --do_train \
   --dataset=universal_act_dstc2 \
   --task=dm --task_name=sysact --example_type=turn \
   --model_type=${model} --model_name_or_path=${bert_dir} \
   --output_dir=${output_dir}/DA/DSTC2-Ratio/R${ratio} \
   --batch_size=8 \
   --eval_batch_size=8 \
   --learning_rate=5e-5 \
   --usr_token=[USR] --sys_token=[SYS] \
   --earlystop=f1_weighted \
   --train_data_ratio=${ratio} \
   --epoch=${epoch} \
   --new_candidate_num=500 \
   --iterations=${iterations} \
   --eval_by_epoch=2 \
   --patience=10 \
   $add1 $add2 $add3


#==============Dialog Act Prediction GSIM ratio=0.01===============
ratio=0.01
CUDA_VISIBLE_DEVICES=$gpu python main_st.py \
   --my_model=multi_label_classifier \
   --do_train \
   --dataset=universal_act_sim_joint \
   --task=dm --task_name=sysact --example_type=turn \
   --model_type=${model} --model_name_or_path=${bert_dir} \
   --output_dir=${output_dir}/DA/SIM_JOINT-Ratio/R${ratio} \
   --batch_size=8 \
   --eval_batch_size=8 \
   --learning_rate=5e-5 \
   --usr_token=[USR] --sys_token=[SYS] \
   --earlystop=f1_weighted \
   --train_data_ratio=${ratio} \
   --epoch=${epoch} \
   --new_candidate_num=100 \
   --iterations=${iterations} \
   --eval_by_epoch=2 \
   --patience=10 \
   $add1 $add2 $add3


#==============Dialog Act Prediction GSIM ratio=0.1===============
ratio=0.1
CUDA_VISIBLE_DEVICES=$gpu python main_st.py \
   --my_model=multi_label_classifier \
   --do_train \
   --dataset=universal_act_sim_joint \
   --task=dm --task_name=sysact --example_type=turn \
   --model_type=${model} --model_name_or_path=${bert_dir} \
   --output_dir=${output_dir}/DA/SIM_JOINT-Ratio/R${ratio} \
   --batch_size=8 \
   --eval_batch_size=8 \
   --learning_rate=5e-5 \
   --usr_token=[USR] --sys_token=[SYS] \
   --earlystop=f1_weighted \
   --train_data_ratio=${ratio} \
   --epoch=${epoch} \
   --new_candidate_num=500 \
   --iterations=${iterations} \
   --eval_by_epoch=2 \
   --patience=10 \
   $add1 $add2 $add3


#==============Response Selection MWOZ ratio=0.01 ===============
ratio=0.01
CUDA_VISIBLE_DEVICES=$gpu python main_st.py \
   --my_model=dual_encoder_ranking \
   --do_train \
   --dataset=multiwoz  --task=nlg \
   --task_name=rs \
   --example_type=turn \
   --model_type=${model} \
   --model_name_or_path=${bert_dir} \
   --output_dir=${output_dir}/RS/MWOZ-Ratio/R${ratio} \
   --batch_size=20 --eval_batch_size=100 \
   --usr_token=[USR] --sys_token=[SYS] \
   --train_data_ratio=${ratio} \
   --max_seq_length=256 \
   --epoch=${epoch} \
   --new_candidate_num=500 \
   --iterations=${iterations} \
   --earlystop='top-1' \
   $add1 $add2 $add3

#==============Response Selection MWOZ ratio=0.1 ===============
ratio=0.1
CUDA_VISIBLE_DEVICES=$gpu python main_st.py \
   --my_model=dual_encoder_ranking \
   --do_train \
   --dataset=multiwoz  --task=nlg \
   --task_name=rs \
   --example_type=turn \
   --model_type=${model} \
   --model_name_or_path=${bert_dir} \
   --output_dir=${output_dir}/RS/MWOZ-Ratio/R${ratio} \
   --batch_size=20 --eval_batch_size=100 \
   --usr_token=[USR] --sys_token=[SYS] \
   --train_data_ratio=${ratio} \
   --max_seq_length=256 \
   --epoch=${epoch} \
   --new_candidate_num=4500 \
   --iterations=${iterations} \
   --earlystop='top-1' --not_save_model \
   $add1 $add2 $add3

#==============Response Selection DSTC2 ratio=0.01 ===============
ratio=0.01
CUDA_VISIBLE_DEVICES=$gpu python main_st.py \
   --my_model=dual_encoder_ranking \
   --do_train \
   --dataset=universal_act_dstc2 \
   --task=nlg --task_name=rs \
   --example_type=turn \
   --model_type=${model} \
   --model_name_or_path=${bert_dir} \
   --output_dir=${output_dir}/RS/DSTC2-Ratio/R${ratio} \
   --batch_size=20 --eval_batch_size=100 \
   --train_data_ratio=${ratio} \
   --max_seq_length=256 \
   --epoch=${epoch} \
   --new_candidate_num=100 \
   --iterations=${iterations} \
   --earlystop='top-1'\
   $add1 $add2 $add3

#==============Response Selection DSTC2 ratio=0.1 ===============
ratio=0.1
CUDA_VISIBLE_DEVICES=$gpu python main_st.py \
   --my_model=dual_encoder_ranking \
   --do_train \
   --dataset=universal_act_dstc2 \
   --task=nlg --task_name=rs \
   --example_type=turn \
   --model_type=${model} \
   --model_name_or_path=${bert_dir} \
   --output_dir=${output_dir}/RS/DSTC2-Ratio/R${ratio} \
   --batch_size=20 --eval_batch_size=100 \
   --train_data_ratio=${ratio} \
   --max_seq_length=256 \
   --epoch=${epoch} \
   --new_candidate_num=700 \
   --iterations=${iterations} \
   --earlystop='top-1' --not_save_model \
   $add1 $add2 $add3

#==============Response Selection GSIM ratio=0.01 ===============
ratio=0.01
CUDA_VISIBLE_DEVICES=$gpu python main_st.py \
   --my_model=dual_encoder_ranking \
   --do_train \
   --dataset=universal_act_sim_joint \
   --task=nlg --task_name=rs \
   --example_type=turn \
   --model_type=${model} \
   --model_name_or_path=${bert_dir} \
   --output_dir=${output_dir}/RS/SIM_JOINT-Ratio/R${ratio} \
   --batch_size=20 --eval_batch_size=100 \
   --train_data_ratio=${ratio} \
   --max_seq_length=256 \
   --epoch=${epoch} \
   --new_candidate_num=80 \
   --iterations=${iterations} \
   --earlystop='top-1'\
   $add1 $add2 $add3

#==============Response Selection GSIM ratio=0.1 ===============
ratio=0.1
CUDA_VISIBLE_DEVICES=$gpu python main_st.py \
   --my_model=dual_encoder_ranking \
   --do_train \
   --dataset=universal_act_sim_joint \
   --task=nlg --task_name=rs \
   --example_type=turn \
   --model_type=${model} \
   --model_name_or_path=${bert_dir} \
   --output_dir=${output_dir}/RS/SIM_JOINT-Ratio/R${ratio} \
   --batch_size=20 --eval_batch_size=100 \
   --train_data_ratio=${ratio} \
   --max_seq_length=256 \
   --epoch=${epoch} \
   --new_candidate_num=500 \
   --iterations=${iterations} \
   --earlystop='top-1' --not_save_model \
   $add1 $add2 $add3

