# 该脚本用于设置输入到网络的参数形状及格式

import torch
import numpy as np
import pandas as pd
import einops as ep
from random import randint


def dataprocess(filename1,filename2,filename3,filename4,simple_length=1000, num_split=1, data_length=100,
                   num_simple_input=False):
    data_np1 = np.array(pd.read_csv(filename1, header=None))
    data_np2 = np.array(pd.read_csv(filename2, header=None))
    data_np3 = np.array(pd.read_csv(filename3, header=None))
    data_np4 = np.array(pd.read_csv(filename4, header=None))
    data_mix1 = ep.rearrange(data_np1, '(h w) b -> b h w', w=simple_length)
    data_mix2 = ep.rearrange(data_np2, '(h w) b -> b h w', w=simple_length)
    data_mix3 = ep.rearrange(data_np3, '(h w) b -> b h w', w=simple_length)
    data_mix4 = ep.rearrange(data_np4, '(h w) b -> b h w', w=simple_length)

        # 数据形状设置
        # simple_length：原始数据切分时每一条样本的长度
        # data_length：每一条样本转换为二维矩阵时每一行的样本点数
    fe_tr_temp1 = data_mix1[:, 0: num_split, :]
    fe_tr_temp2 = data_mix2[:, 0: num_split, :]
    fe_tr_temp3 = data_mix3[:, 0: num_split, :]
    fe_va_temp  = data_mix4[:, 0: num_split, :]
        #fe_va_temp = data_mix[:, num_split: num_batchsize, :]
    fe_tr1 = ep.rearrange(fe_tr_temp1, 'b h (h1 w) -> (b h) h1 w', w=data_length)
    fe_tr2 = ep.rearrange(fe_tr_temp2, 'b h (h1 w) -> (b h) h1 w', w=data_length)
    fe_tr3 = ep.rearrange(fe_tr_temp3, 'b h (h1 w) -> (b h) h1 w', w=data_length)
    fe_va = ep.rearrange(fe_va_temp, 'b h (h1 w) -> (b h) h1 w', w=data_length)
    fe_tr=np.vstack((fe_tr1,fe_tr2,fe_tr3,fe_va))
    num1=2803
    num2=871
    num3=2375
    num4=1428
    la_tr1 = np.zeros(((num1-9) * num_split, 10))
    la_tr2 = np.zeros(((num2-9) * num_split, 10))
    la_tr3 = np.zeros(((num3-9) * num_split, 10))
    la_va = np.zeros(((num4-9) * num_split, 10))
        #la_va = np.zeros((2794 * (num_batchsize-num_split), 10))
        
    label1=np.zeros((num1 * num_split,1))
    i=0
    while i<num1 * num_split:
        label1[i]=(num1 * num_split-i)/(num1 * num_split)
        i+=1
    for i in range(num1 * num_split-10):
        la_tr1[i,:]=label1[i:i+10].transpose()

    label2=np.zeros((num2 * num_split,1))
    i=0
    while i<num2 * num_split:
        label2[i]=(num2 * num_split-i)/(num2 * num_split)
        i+=1
    for i in range(num2 * num_split-10):
        la_tr2[i,:]=label2[i:i+10].transpose()

    label3=np.zeros((num3 * num_split,1))
    i=0
    while i<num3 * num_split:
        label3[i]=(num3 * num_split-i)/(num3 * num_split)
        i+=1
    for i in range(num3 * num_split-10):
        la_tr3[i,:]=label3[i:i+10].transpose()            
        
    label4=np.zeros((num4*num_split,1))
    i=0
    while i<num4*num_split:
        label4[i]=(num4*num_split-i)/(num4*num_split)
        i+=1
    for i in range(num4*num_split-10):
        la_va[i,:]=label4[i:i+10].transpose()
    la_tr=np.vstack((la_tr1,la_tr2,la_tr3,la_va))
      
        

        # 数据格式设置
    fe_tr = torch.tensor(fe_tr)
    fe_tr = fe_tr.to(torch.float32)
    fe_va = torch.tensor(fe_va)
    fe_va = fe_va.to(torch.float32)

    la_tr = torch.tensor(la_tr)
    la_tr = la_tr.squeeze(-1)
    la_tr = la_tr.float()
    la_va = torch.tensor(la_va)
    la_va = la_va.squeeze(-1)
    la_va = la_va.float()

    return fe_tr, fe_va, la_tr, la_va
