import torch
from torch import nn
from autoencoder1 import ResNet_autoencoder, Bottleneck, DeconvBottleneck
from coco import load_dataset
import matplotlib.pyplot as plt
from torch.autograd import Variable

EPOCH = 1000

if __name__ == "__main__":
    model = ResNet_autoencoder(Bottleneck, DeconvBottleneck, [
        3, 4, 6, 3], 3).cuda()

    # load data
    print("start loading.")
    dataloader = load_dataset('/home/achhe_ucdavis_edu/resnet-autoencoder/data')
    print("load data success.")
    '''
    load pre_trained_model
    '''
    pretrained_dict = torch.load('./resnet50-19c8e357.pth')
    print("load pretrained model success")

    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k,
                       v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    # fix encoder
    fix_length = len(pretrained_dict.keys())
    all_length = len(model.state_dict().keys())
    for idx, k in enumerate(model_dict.keys()):
        if idx < fix_length:
            model.state_dict()[k].requires_grad = False

    params = filter(lambda p: p.requires_grad, model.parameters())

    # Loss and Optimizer
    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(params, lr=1e-4)

    model.train()

    loss_list=[]

    print("start training.")
    
    for epoch in range(EPOCH):
        for batch_idx, (image, target) in enumerate(dataloader):
            image = Variable(image.cuda())

            # Forward + Backward + Optimize

            optimizer.zero_grad()

            tmp1, tmp2, tmp3 = model(image)

            loss1 = criterion(tmp2,image.detach())
            loss2 = criterion(tmp3,tmp1.detach())
            loss = loss1 + loss2

            loss.backward()

            optimizer.step()

            if (batch_idx+1) % 2 == 0:
                print ("Epoch [%d/%d], Iter [%d] Loss: %.4f" % (epoch+1, EPOCH, batch_idx+1, loss.data[0]))

        loss_list.append(loss)
        plt.plot(loss_list)
        plt.ylable('loss')
        plt.show()
        
        if((epoch+1)%100==0):
            torch.save(resnet.state_dict(), './save/resnet'+str(epoch+1)+'pkl')
