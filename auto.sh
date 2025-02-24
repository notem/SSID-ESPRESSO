#
DATA_FILE=../data/processed-socat.pkl
DATA_FILE=../../SSID/datasets/processed_May17.pkl
DATA_FILE=../data/ssh/processed_nov17_fixtime.pkl
DATA_FILE=../data/processed-dns.pkl
DATA_FILE=../data/processed-icmp.pkl
RES_DIR=./res
#EXP_NAME=Espresso_brok10
#EXP_NAME=Espresso_brok8
#EXP_NAME=Espresso_mixed3
#EXP_NAME=Espresso_ssh_host
EXP_NAME=Espresso_icmp_host
#EXP_NAME=Espresso_ssh_chain_01
#EXP_NAME=Espresso_ssh_host
#EXP_NAME=Espresso_dns_host
CKPT_NAME=final

MODE=network-ends #{same-host,network-ends,network-all}


#python chain-joint-new.py \
#    --data $DATA_FILE \
#    --mode $MODE \
#    --exp_name $EXP_NAME \
#    --ckpt_dir $RES_DIR/$EXP_NAME/ckpt \
#    --results_dir $RES_DIR/$EXP_NAME/logs \
#    --cache_dir ./cache \
#    --ckpt_epoch 40 \
#    --config ./configs/espresso.json \
#    --margin 0.5 \
#    --w 0.0 \
#    --temporal_alignment \
#    --online \
#    --host \
#    --single_fen
#    #--hard \
#    #--config ./configs/espresso-tweak.json \


python calc_sims.py \
    --mode $MODE \
    --ckpt ${RES_DIR}/${EXP_NAME}/ckpt/${EXP_NAME}/${CKPT_NAME}.pth \
    --sims_file ${RES_DIR}/${EXP_NAME}/sims2.pkl \
    --host \
    --data $DATA_FILE
    #--ckpt ${RES_DIR}/${EXP_NAME}/ckpt/${CKPT_NAME}.pth \
    #--temporal_alignment \
#
python benchmark-corr.py \
    --sims_file ${RES_DIR}/${EXP_NAME}/sims2.pkl \
    --results_file ${RES_DIR}/${EXP_NAME}/res.pkl \
    --dropout 0.3

#python benchmark-tmp.py \
#    --sims_file ${RES_DIR}/${EXP_NAME}/sims2.pkl \
#    --results_file ${RES_DIR}/${EXP_NAME}/res.pkl \
#    --dropout 0.1


#
#DATA_FILE=../../SSID/datasets/processed_May17.pkl
#EXP_NAME=Espresso_mixed_host
#
#python calc_sims.py \
#    --mode $MODE \
#    --ckpt ${RES_DIR}/${EXP_NAME}/ckpt/${EXP_NAME}/${CKPT_NAME}.pth \
#    --sims_file ${RES_DIR}/${EXP_NAME}/sims2.pkl \
#    --host \
#    --data $DATA_FILE
#    #--ckpt ${RES_DIR}/${EXP_NAME}/ckpt/${CKPT_NAME}.pth \
#    #--temporal_alignment \
#
#python benchmark-corr.py \
#    --sims_file ${RES_DIR}/${EXP_NAME}/sims2.pkl \
#    --results_file ${RES_DIR}/${EXP_NAME}/res.pkl \
#    --dropout 0.2
#
#
#DATA_FILE=../data/processed-icmp.pkl
#EXP_NAME=Espresso_icmp_host
#
#python calc_sims.py \
#    --mode $MODE \
#    --ckpt ${RES_DIR}/${EXP_NAME}/ckpt/${EXP_NAME}/${CKPT_NAME}.pth \
#    --sims_file ${RES_DIR}/${EXP_NAME}/sims2.pkl \
#    --host \
#    --data $DATA_FILE
#    #--ckpt ${RES_DIR}/${EXP_NAME}/ckpt/${CKPT_NAME}.pth \
#    #--temporal_alignment \
#
#python benchmark-corr.py \
#    --sims_file ${RES_DIR}/${EXP_NAME}/sims2.pkl \
#    --results_file ${RES_DIR}/${EXP_NAME}/res.pkl \
#    --dropout 0.2
#
#
#DATA_FILE=../data/processed-socat.pkl
#EXP_NAME=Espresso_socat_host
#
#python calc_sims.py \
#    --mode $MODE \
#    --ckpt ${RES_DIR}/${EXP_NAME}/ckpt/${EXP_NAME}/${CKPT_NAME}.pth \
#    --sims_file ${RES_DIR}/${EXP_NAME}/sims2.pkl \
#    --host \
#    --data $DATA_FILE
#    #--ckpt ${RES_DIR}/${EXP_NAME}/ckpt/${CKPT_NAME}.pth \
#    #--temporal_alignment \
#python benchmark-corr.py \
#    --sims_file ${RES_DIR}/${EXP_NAME}/sims2.pkl \
#    --results_file ${RES_DIR}/${EXP_NAME}/res.pkl \
#    --dropout 0.2
