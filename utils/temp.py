import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(sys.path)

from emrmodel.extended_mask_rcnn import ExtendedMaskRCNN
import torch


model = ExtendedMaskRCNN(1, min_size=642, max_size=652)

# print(model.backbone.out_channels)

temp_img = torch.load("datasets/Fluo-N3DH-SIM+/train/imgs/0000_052.pt") # torch.Size([642, 652])
temp_target = torch.load("datasets/Fluo-N3DH-SIM+/train/masks/0000_052.pt")
print(temp_img.shape, type(temp_target)) 
temp_img = torch.stack([temp_img] * 5) # torch.Size([3, 642, 652])
temp_target = [temp_target, temp_target] # batch size 2
print(temp_img.shape)
temp_img = torch.stack([temp_img, temp_img])
print(temp_img.shape)

# features = model.backbone(temp_img)

# for i in ['0', '1', '2', '3']:
#     print(features[i].shape)

# features_backbone_body = model.backbone.body(temp_img)

# for i in ['0', '1', '2', '3']:
#     print(features_backbone_body[i].shape)

# print(model.backbone.body.conv1.in_channels)

model.train()
loss = model(temp_img, temp_target)

print('output')
print(loss)
