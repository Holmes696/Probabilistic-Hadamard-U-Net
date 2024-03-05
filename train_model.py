import torch
import numpy as np
from torch.utils.data import DataLoader
#from torch.utils.data.sampler import SubsetRandomSampler
from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularisation
import pdb
import matplotlib.pyplot as plt
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     #torch.cuda.manual_seed(seed)
     np.random.seed(seed)

setup_seed(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Load training dataset
train_f_np = np.load('../Data_preprocess/x_train.npy')
train_g_np = np.load('../Data_preprocess/y_train.npy')

#Normalize to 0~1
train_f_np=(train_f_np-np.min(train_f_np))
train_f_np=train_f_np/np.max(abs(train_f_np))
train_g_np=(train_g_np-np.min(train_g_np))
train_g_np=train_g_np/np.max(abs(train_g_np))

#To pytorch
train_f = torch.from_numpy(train_f_np).float()
train_g = torch.from_numpy(train_g_np).float()

train_f = torch.unsqueeze(train_f,1)
train_g = torch.unsqueeze(train_g,1)


train_dataset = torch.utils.data.TensorDataset(train_f,train_g)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=5,shuffle=True,drop_last=True)

#Training the model
net = ProbabilisticUnet(input_channels=1, num_classes=1, num_filters=[32,64], latent_dim=2, no_convs_fcomb=4, beta=10)
net.to(device)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4, weight_decay=0)
epochs = 100
train_loss_log=[]
epochs_list=[]
for epoch in range(epochs):
    print(epoch)
    cv_sum=0
    train_loss = 0#初始化
    num_batches = len(train_loader)#batch的数量
    for step, (patch, mask) in enumerate(train_loader): 
        patch = patch.to(device)
        mask = mask.to(device)
        #mask = torch.unsqueeze(mask,1)
        net.forward(patch, mask, training=True)
        elbo = net.elbo(mask)
        reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
        loss = -elbo*1e+1 + 0 * reg_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()#求损失
    train_loss /= num_batches#求平均训练损失
    epochs_list.append(epoch)
    print(f"Train Avg loss: {train_loss:>8f}")#输出训练损失
    train_loss_log.append(train_loss)
    if train_loss<=min(train_loss_log):
        torch.save(net.state_dict(), './weight.pt')



        