import torch
import torch.nn.modules as nn
import torch.optim as optim
import torch.utils.data as data
from model_tft import TFT
from dataprocess import dataprocess
from tft_train import func_train





device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batchsize = 32
epoch =33
lr = 0.005

simple_length_ = 1000
data_length_ = 100
num_split_ = 1



feature_tr, feature_te, label_tr, label_te = dataprocess_sq(
    'C:/Users/Administrator/Desktop/PHM/bearing1-1.csv','C:/Users/Administrator/Desktop/PHM/bearing1-2.csv',
    'C:/Users/Administrator/Desktop/PHM/bearing1-3.csv','C:/Users/Administrator/Desktop/PHM/bearing1-4.csv',
    simple_length=simple_length_,
    num_split=num_split_,
    data_length=data_length_
)


ds_tr = data.TensorDataset(feature_tr, label_tr)
iter_tr = data.DataLoader(ds_tr, batchsize, shuffle=True)
ds_va = data.TensorDataset(feature_te, label_te)
iter_va = data.DataLoader(ds_va, batchsize, shuffle=False)  # 打乱的时候特征向量和标签向量会同时打乱

net_set = SiT_sq(num_classes=10, depth=5, heads=4, dim=64, dim_head=32, mlp_dim=256,
                 simple_length=simple_length_, data_length=data_length_)

optimizer_set = optim.Adamax(net_set.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
scheduler_set = optim.lr_scheduler.CosineAnnealingLR(optimizer_set, T_max=5, eta_min=0.00003)

loss_set = nn.L1Loss(reduction='sum')

func_train(net_set, iter_tr,iter_va, loss_set, optimizer_set, scheduler_set, epoch, device)

