#!/usr/bin/env bash
#sleep 30
#fi_info -p efa -t FI_EP_RDMH=`hostname`

echo myuser=`whoami`
echo LD_LIBRARY_PATH = $LD_LIBRARY_PATH
echo PATH = $PATH
echo which mpicc `which mpicc`
echo HOSTNAMES = $SLURM_JOB_NODELIST
echo hostname = `hostname`
echo MASTER_ADDR= $SLURM_LAUNCH_NODE_IPADDR
echo MASTER_PORT= $MASTER_PORT
echo SLURM_JOB_NODELIST= $SLURM_JOB_NODELIST
echo SLURM_JOB_NUM_NODES= $SLURM_JOB_NUM_NODES
echo NCCL_ASYNC_ERROR_HANDLING=$NCCL_ASYNC_ERROR_HANDLING

export HF_TOKEN="" #INSERT

export H=`hostname`
export THEID=$SLURM_PROCID
echo THEID=$THEID

cd ~/hf_fsdp

accelerate launch \
--num_processes=$(( 8 * $SLURM_JOB_NUM_NODES )) \
--num_machines $SLURM_JOB_NUM_NODES \
--machine_rank $THEID \
--main_process_ip $SLURM_LAUNCH_NODE_IPADDR \
--main_process_port $MASTER_PORT \
--mixed_precision=bf16  \
--config_file accelerate_config.yaml \
train.py \
--batch_size 8 \
--gradient_accumulate_every 1 \
--wandb_entity "tmabraham" \
--wandb_project "pubmed-llama-2" \
--wandb_name "pubmed-llama-2-7b-full-epoch-accelerate-test" \
--output_dir "/fsx/home-tmabraham/ckpts/pubmed-llama-2-7b/pubmed-llama-2-7b-full-epoch-accelerate-test" \
--dataset_name "tmabraham/pubmed-enrico-tokenized"
