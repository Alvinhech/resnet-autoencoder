import torch
from torch import nn
import torchvision.transforms as transforms
from autoencoder4 import ResNet_autoencoder, Bottleneck, DeconvBottleneck
from coco import load_dataset
import matplotlib.pyplot as plt
from torch.autograd import Variable
from Loader import Dataset
import torchvision.transforms.functional as F
from PIL import Image


def wct(c, s, alpha):
    '''
    Apply whitening and coloring transforms on content and style images.

    :param c: matrix of content image
    :param s: matrix of style image
    :param alpha: coefficient to control the strength of transformation
    :return: transformed matrix
    '''

    # content image whitening
    c = c.double()
    c_channels, c_width, c_height = c.size(0), c.size(1), c.size(2)
    c_new = c.view(c_channels, -1)  # c * (h * w)
    c_mean = torch.mean(c_new, 1)  # c
    c_mean = c_mean.unsqueeze(1).expand_as(c_new)  # c * 1 -> c * (h * w)
    c_new = c_new - c_mean  # subtract mean

    c_cov = torch.mm(c_new, c_new.t()).div(
        c_width * c_height - 1)  # covariance matrix of c*cT
    # singular value decomposition, c_e is a diagonal matrix
    c_u, c_e, c_v = torch.svd(c_cov, some=False)
    # find all positive eigenvalues of c*cT at the beginning
    c_pos_eigens_idx = c_channels
    for i in range(c_channels):
        if c_e[i] < 0.00001:
            c_pos_eigens_idx = i
            break
    # list of all positive eigenvalues of c*cT at the beginning
    c_pos_eigens = c_e[0: c_pos_eigens_idx]
    c_o = c_v[:, 0:c_pos_eigens_idx]  # orthogonal matrix of eigenvalues, Ec

    whitened = torch.mm(c_o, torch.diag(
        c_pos_eigens.pow(-0.5)))  # Ec * Dc^(-1/2)
    whitened = torch.mm(whitened, c_o.t())  # Ec * Dc^(-1/2) * EcT
    whitened = torch.mm(whitened, c_new)  # Ec * Dc^(-1/2) * EcT * c

    # style image coloring
    s = s.double()
    _, s_width, s_height = s.size(0), s.size(1), s.size(2)
    s_new = s.view(c_channels, -1)  # c * (h * w)
    s_mean = torch.mean(s_new, 1)  # c
    s_mean = s_mean.unsqueeze(1).expand_as(s_new)  # c * 1 -> c * (h * w)
    s_new = s_new - s_mean

    s_cov = torch.mm(s_new, s_new.t()).div(
        s_width * s_height - 1)  # covariance matrix of s*sT
    # singular value decomposition, s_e is a diagonal matrix
    s_u, s_e, s_v = torch.svd(s_cov, some=False)
    # find all positive eigenvalues of s*sT at the beginning
    s_pos_eigens_idx = c_channels
    for i in range(c_channels):
        if s_e[i] < 0.00001:
            s_pos_eigens_idx = i
            break
    # list of all positive eigenvalues of s*sT at the beginning
    s_pos_eigens = s_e[0: s_pos_eigens_idx]
    s_o = s_v[:, 0:s_pos_eigens_idx]  # orthogonal matrix of eigenvalues, Es

    colored = torch.mm(s_o, torch.diag(s_pos_eigens.pow(0.5)))  # Es * Ds^(1/2)
    colored = torch.mm(colored, s_o.t())  # Es * Ds^(1/2) * EsT
    colored = torch.mm(colored, whitened)  # Es * Ds^(1/2) * EsT * whitened
    # re-center the colored matrix with the mean
    colored = colored + s_mean.resize_(colored.size())
    colored = colored.view_as(c)

    # use alpha to control the strength of transformation
    transformed = alpha * colored + (1.0 - alpha) * c
    return transformed.float().unsqueeze(0)


PATH = "./save4/resnet3_2000.pkl"


def default_loader(path):
    return Image.open(path).convert('RGB')


if __name__ == "__main__":
    model = ResNet_autoencoder(Bottleneck, DeconvBottleneck, [
        3, 4, 6, 3], 3).cuda()

    model.load_state_dict(torch.load(PATH))
    model.eval()

    # Data loading code
    dataset = Dataset("./content", "./style", )
    loader = torch.utils.data.DataLoader(dataset=dataset,
                                         batch_size=1,
                                         shuffle=False)
    alpha_list = [0.4, 0.6, 0.7, 0.8, 1.0]
    for alpha in alpha_list:
        for i, (contentImg, styleImg, imname) in enumerate(loader):
            imname = imname[0]
            print('Transferring ' + imname)
            contentImg = contentImg.cuda()
            styleImg = styleImg.cuda()
            cImg = Variable(contentImg, volatile=True)
            sImg = Variable(styleImg, volatile=True)

            c, s = model.encoder(cImg), model.encoder(sImg)
            c = c.data.cpu().squeeze(0)
            s = s.data.cpu().squeeze(0)
            o = wct(c, s, alpha)
            o = Variable(o.cuda(), volatile=True)
            out = model.decoder(o, cImg.size())
            out = out.data.cpu().squeeze(0)
            output = F.to_pil_image(out)
            output.save("./output4/" + str(alpha) + "/" + imname)
