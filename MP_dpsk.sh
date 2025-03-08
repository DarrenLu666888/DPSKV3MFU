# torchrun --nnodes 2 --nproc-per-node 8 --node-rank $RANK --master-addr $ADDR dpskv3_params.py
torchrun --nproc-per-node 8   dpskv3_params.py > params.log