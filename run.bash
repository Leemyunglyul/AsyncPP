###########


 #gpipe
rt GLOO_SOCKET_IFNAME=lo
export NCCL_SOCKET_IFNAME=lo

n_processes=8
n_physical_gpus=2

batch="-b 2 --eval-batch-size 2"
epochs="--epochs 2"
minibatches="--num_minibatches 1000 --num_eval_minibatches 25"
model="--module models.gptn.gpus=$n_processes --block_size 128 --n_embd 128 --n_head 4 --n_layer 8 --config_path models/gptn/gpus=$n_processes/mp_conf.json"
d="wikitext-103-v1"
outdir="${d}_gptn_512_384_12_8_b8"
lr="--lr 3e-4 --lr_warmup --optimizer nadamw"
logtb="--log_tb --tb_dir ./runs"
cg="--clip_grad 10"

basecmdstr="python -u main_with_runtime.py $model $batch -d $d --master_addr localhost --master_port 29500 --distributed_backend gloo $lr $epochs $minibatches $cg $logtb --recompute --lr_policy cosine"

method="--momentum 0.99 --optimizer nadamw"
expname="$outdir/gpus=$n_processes/ours/"
ckptdir="$d/$expname"
mkdir -p $ckptdir

cmdstr="$basecmdstr $method --exp_name $expname --checkpoint_dir $ckptdir"

for rank in $(seq 0 $(($n_processes-1)));
do
    local_rank=$(($rank / 4))
    cmd="$cmdstr --rank $rank --local_rank $local_rank > log_rank${rank}.txt 2>&1 &"
    echo $cmd
    eval $cmd
done

wait

# basecmdstr="python sync_main.py $model $batch -d $d $dd --master_addr localhost --distributed_backend nccl 
# $lr $epochs $minibatches $cg $logtb --recompute --lr_policy cosine --optimizer adamw"

# gpipe
# method="--sync_schedule gpipe --num_microbatches 4"
# expname="$outdir/gpus=$ngpus/gpipe/"
# ckptdir="$d/$expname"    
# mkdir -p $ckptdir   # create checkpoint directory if not exists
# cmdstr="$basecmdstr $method --exp_name $expname --checkpoint_dir $ckptdir"
# for rank in $(seq 0 $(($ngpus-1))); do
#    cmd="$cmdstr --rank $rank --local_rank $(($rank % $nnodes)) &"
#    echo $cmd
#    eval $cmd
# done
# wait
