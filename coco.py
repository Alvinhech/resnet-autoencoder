import torchvision.datasets as dset
import torchvision.transforms as transforms
cap = dset.CocoCaptions(root='dir where images are',
                        annFile='json annotation file',
                        transform=transforms.ToTensor())

print('Number of samples: ', len(cap))
img, target = cap[3]  # load 4th sample

print("Image Size: ", img.size())
print(target)
