import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
import os 
import shutil
import os
import argparse



def generate_one_case(broadcase_p, data_type, tensor_size):
    a_b = random.randint(tensor_size['C'][0], tensor_size['C'][1])
    a_c = random.randint(tensor_size['B'][0], tensor_size['B'][1])
    a_h = random.randint(tensor_size['H'][0], tensor_size['H'][1])
    a_w = random.randint(tensor_size['W'][0], tensor_size['W'][1])

    b_b = a_b
    b_c = a_c
    b_h = (a_h + 1) // 2
    b_w = (a_w + 1) // 2
    
    #for triggering broadcast 
    if random.random() < broadcase_p: a_b = 1 
    if random.random() < broadcase_p: a_c = 1
    
    if random.random() < broadcase_p: b_b = 1
    if random.random() < broadcase_p: b_c = 1
    if random.random() < broadcase_p: b_h = 1
    if random.random() < broadcase_p: b_w = 1
    
      
    if data_type == "int32":
        a = torch.randint(low = 0, high= 1000000, size = (a_b, a_c, a_h, a_w)).double()
        b = torch.randint(low = 0, high= 1000000, size = (b_b, b_c, b_h, b_w)).double()
        
    elif data_type == 'float32':
        a = torch.rand((a_b, a_c, a_h, a_w)).float() * 1000000
        b = torch.rand((b_b, b_c, b_h, b_w)).float() * 1000000
        
    elif data_type == 'double':
        a = torch.rand((a_b, a_c, a_h, a_w)).double() * 1000000
        b = torch.rand((b_b, b_c, b_h, b_w)).double() * 1000000
    
    return a,b


maxpool = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))
data_type_list = ['int32', 'float32', 'double']

def generate_cases(test_case_dir, case_nums, tensor_size, broadcase_p):
    if os.path.exists(test_case_dir):
        shutil.rmtree(test_case_dir)
    os.mkdir(test_case_dir)
    for i in tqdm(range(case_nums)):
        data_type = random.sample(data_type_list, k =1)[0]
        a, b = generate_one_case(broadcase_p, data_type, tensor_size)
        c = (maxpool(a)) + b 

        # append meta info, i.e, shape to the start of the tensor
        a = torch.cat((torch.tensor(a.shape),  a.flatten()), dim = 0)
        b = torch.cat((torch.tensor(b.shape),  b.flatten()), dim = 0)
        c = torch.cat((torch.tensor(c.shape),  c.flatten()), dim = 0)
        
        ## serialize a, b, c to binary files
        a.numpy().astype(data_type).tofile('./{}/case_{}_{}_a.bin'.format(test_case_dir,i,data_type))
        b.numpy().astype(data_type).tofile('./{}/case_{}_{}_b.bin'.format(test_case_dir,i,data_type))
        c.numpy().astype(data_type).tofile('./{}/case_{}_{}_c.bin'.format(test_case_dir,i,data_type))




if __name__ == "__main__":
    size_dict = {
        'small': { "C": [ 1,  4], "B": [ 1,  4], "H": [100, 200], "W":[100, 200] },
        'mid'  : { "C": [ 4,  8], "B": [ 4,  8], "H": [200, 400], "W":[200, 400]},
        'large': { "C": [ 8, 16], "B": [ 8, 16], "H": [400, 800], "W":[400, 800]},
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--tensor_size', choices = ['small', 'mid', 'large'], default= 'large', help = 'Operated tensor size')
    parser.add_argument('-o','--out_directory', type = str, default='./test_case', help='Directory to save test cases')
    parser.add_argument('-n','--case_nums', type = int, default = 100, help='Number of test cases')
    parser.add_argument('-p','--broadcast_probality', type = float, default= 0.1,help = 'Probability of performing broadcast')
    parser.add_argument('-r','--random_seed', type = int, default= 42, help = 'Random seed')
    args = parser.parse_args()
    
        
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    generate_cases(args.out_directory, args.case_nums, size_dict[args.tensor_size],args.broadcast_probality)

