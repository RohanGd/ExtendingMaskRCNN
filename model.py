import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn

import tifffile as tiff
model = maskrcnn_resnet50_fpn(weights=None)
# model = model.backbone
# print(model)

img = tiff.imread("Fluo-N3DH-SIM+/01/t008.tif") 
img = img[40:43]
img = torch.Tensor(img)
img = img.unsqueeze(0)
print(img.shape)
model.eval()
print(model(img))
# print(model(img)[0]["masks"].shape)


'''
Passing a 3 channel image directly to the model yields an output with masks and boxes for the 2d image.
So passing 3 consecutive 2d slices as inputs would yield only masks and boxes for one of 2d image.
The idea is to modify the maskrcnn architecture so that 3 inputs are maintained until the RPN, and only then some fusion 
'''