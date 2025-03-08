import sys
sys.path.append('DeepSeek-V3/inference')
from model import Transformer, ModelArgs
import torch
import torch.distributed as dist

import json


'''方法1，自定义函数 参考自 https://blog.csdn.net/qq_33757398/article/details/109210240'''
def model_structure(model):
    blank = ' '
    print('-' * 120)
    print('|' + ' ' * 15 + 'weight name' + ' ' * 16 + '|' \
            + ' ' * 10 + 'weight shape' + ' ' * 10 + '|' \
            + ' ' * 3 + 'number' + ' ' * 3 + '|' \
            + ' ' * 6 + 'dtype' + ' ' * 7 + '|' \
            + ' ' * 1 + 'space(Bytes)' + ' ' * 1 + '|') #\
            # + ' ' * 1 + 'rank' + ' ' * 1 + '|')
    print('-' * 120)
    num_para_y = 0 # number of parameters in all layers splited by world_size
    num_para_n = 0 # number of parameters in all layers not splited by world_size
    total_space_y = 0 # space of parameters in all layers splited by world_size
    total_space_n = 0 # space of parameters in all layers not splited by world_size

    for index, (key, w_variable) in enumerate(model.named_parameters()):
        # print(w_variable.dtype)
        if len(key) <= 40:
            key = key + (40 - len(key)) * blank
        shape = str(w_variable.shape)
        if len(shape) <= 30:
            shape = shape + (30 - len(shape)) * blank
        each_para = 1
        for k in w_variable.shape:
            each_para *= k
        # num_para += each_para
        if ('gate' in key or 'norm' in key or 'wq_a' in key or 'wkv_a' in key): # or 'shared' in key):
            num_para_n += each_para
        else:
            num_para_y += each_para
        str_num = str(each_para)
        if len(str_num) <= 10:
            str_num = str_num + (10 - len(str_num)) * blank
        str_dtype = str(w_variable.dtype)
        if len(str_dtype) <= 16:
            str_dtype = str_dtype + (16 - len(str_dtype)) * blank
        str_space = each_para * w_variable.itemsize
        # total_space += str_space
        if ('gate' in key or 'norm' in key or 'wq_a' in key or 'wkv_a' in key): # or 'shared' in key):
            total_space_n += str_space
        else:
            total_space_y += str_space
        if len(str(str_space)) <= 12:
            str_space = str(str_space) + (12 - len(str(str_space))) * blank
        # now_rank = rank
        # if (len(str(now_rank)) <= 4):
        #     now_rank = str(now_rank) + (4 - len(str(now_rank))) * blank
        if (index < 2):
            print('| {} | {} | {} | {} | {} |'.format(key, shape, str_num, str_dtype, str_space))
        
        
    print('-' * 90)
    print('The total number of parameters splited by world_size: ' + str(num_para_y) + '(%.2fB)' % (num_para_y / ((1000**3))))
    print('The total number of parameters not splited by world_size: ' + str(num_para_n) + '(%.2fB)' % (num_para_n / ((1000**3))))
    total_para = num_para_y + num_para_n
    print('The total number of parameters: ' + str(total_para) + '(%.2fB)' % (total_para / ((1000**3))))
    print('-' * 90)
    print('The space of parameters splited by world_size: ' + str(total_space_y) + '(%.2fB)' % (total_space_y / ((1024**3))))
    print('The space of parameters not splited by world_size: ' + str(total_space_n) + '(%.2fB)' % (total_space_n / ((1024**3))))
    total_space = total_space_y + total_space_n
    print('The space of parameters: ' + str(total_space) + '(%.2fGB)' % (total_space / ((1024**3))))
    print('-' * 90)

# model_structure(net)
if (__name__ == '__main__'):
    # model_structure(model)
    # 查看可用 GPU 数量
    # num_gpus = torch.cuda.device_count()
    # print(f"可用 GPU 数量: {num_gpus}")
    # # 遍历所有可见的 GPU 并打印信息
    # for gpu_id in range(num_gpus):
    #     print(f"GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    #     total_memory = torch.cuda.get_device_properties(gpu_id).total_memory  # 总显存
    #     allocated_memory = torch.cuda.memory_allocated(gpu_id)  # 已使用显存
    #     reserved_memory = torch.cuda.memory_reserved(gpu_id)  # 预留显存
    #     free_memory = total_memory - allocated_memory  # 空闲显存（大致估算）
    #     print(f"总显存: {total_memory / 1024**2:.2f} MB")
    #     print(f"已使用显存: {allocated_memory / 1024**2:.2f} MB")
    #     print(f"预留显存: {reserved_memory / 1024**2:.2f} MB")
    #     print(f"空闲显存（估算）: {free_memory / 1024**2:.2f} MB")
    # torch.set_default_dtype(torch.bfloat16)
    # torch.set_default_device("cuda")
    # torch.manual_seed(0)
    # args = ModelArgs()
    # with open('DeepSeek-V3/inference/configs/config_671B.json') as f:
    #     args = ModelArgs(**json.load(f))
    # torch.cuda.empty_cache()
    # x = torch.randint(0, args.vocab_size, (1, 128))
    # model = Transformer(args)
    # model_structure(model)
    
    
    # model-parrel inference following "DeepSeek-V3/inference/generate.py"
    config = 'DeepSeek-V3/inference/configs/config_671B.json'
    import os
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    
    global print
    if rank != 0:
        print = lambda *_, **__: None
    print("world_size: ", world_size)
    # print("rank: ", rank)
    # print("local_rank: ", local_rank)
    torch.cuda.set_device(local_rank)
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(965)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
    print(args)
    with torch.device("cuda"):
        model = Transformer(args)
        model_structure(model)

    model(torch.randint(0, args.vocab_size, (1, 128)).cuda())


        # print(torch.cuda.memory_summary())
    if world_size > 1:
        dist.destroy_process_group()