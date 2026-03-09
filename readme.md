# Extending Mask-RCNN
Dataset Link - [Simulated nuclei of HL60 cells stained with Hoechst](https://celltrackingchallenge.net/3d-datasets/).

All class names start with _"emr"_ to avoid clashes with pytorch class name clashes.


## Reading
- [Point-Based Weakly Supervised 2.5D Cell Segmentation](https://link.springer.com/chapter/10.1007/978-3-031-72353-7_25)
- [HowtoBoxYourCells: An Introduction to Box Supervision for 2.5D Cell Instance Segmentation and a Study of Applications](https://www.scitepress.org/Papers/2025/131898/131898.pdf)
- [A Flexible 2.5D Medical Image Segmentation Approach with In-Slice and Cross-Slice Attention](https://arxiv.org/pdf/2405.00130)
- [Beyond mAP: Towards better evaluation of instance segmentation](https://arxiv.org/pdf/2207.01614)
- [Video Instance Segmentation](https://arxiv.org/pdf/1905.04804) -- see evaluatrion metrics - IOU for multiple frames. This one also adds a new head parallel to bounding box regression, classification and masking head. THis 4th head assigns an instance label to each bounding box.

## Things to think about:
- Varying dataset size - H, W will be handled by the transforms module. But num_slices? Currently designing so that the fusion layer will handle that.

The following content has not been updated since 06.01.2026
## ** TODOs ** Fixed Window Fusion:
Cool. Here is what I am going to do now: 
1. Resize to 512 x 512 and recreate dataset. (DONE)
2. Test loop with only SEG metric from official source: [What is SEG](https://public.celltrackingchallenge.net/documents/SEG.pdf), [Instructions](https://public.celltrackingchallenge.net/documents/Evaluation%20software.pdf) and [CLI tool for SEG](http://public.celltrackingchallenge.net/software/EvaluationSoftware.zip) (DONE)
2. Rerun simple convolutions early early fusion for n=3 or 5, for 5 epochs for new 512 x 512 images
3. Init MLP per slice weights as (0.33, 0.34, 0.33) and once as (0,1,0) . (DONE)
4. Fix windowed MLP early fusion. (DONE)
5. make windowed MLP early fusion v2, this time with k*k window size, instead of h/k, w/k window size. (FIXED)
6. Rerun
    - Simple Bacbone no MLP
    - MLP simple slice fusion (zero init_bias)
    - MLP windowed slice fusion (zero init_bias)
    - Compare best MLP model and retrain with init_bias = gaussian or only_center
6. See best MLP model, and implement per feature level seperate MLP. 
7. Implement convolution instead of MLP (early late convolutional fusion) 
8. Pixel wise attention.

## TODO 28.01.2026
1. Log the weights , logits_dynamics, and static logits for slice sE, windowed slice SE, pixel SE.
2. Formulate why you changed your approach in pixelattn
3. Create visualization for each of these methods, gaussians for pixel, alphavalue for slice
4. Remove static logits


## SEG score CLI tool
Stack the 2d masks volume wise and save as man_seg001.tif file in a folder called 01_GT/SEG
Similarly stack the per slice preds volume wise and save as mask001.tif file in a folder called 01_RES


The following content has not been updated since DECEMBER 2025.
## Observations and suggestions 

Model cannot load 59 slices at a time even sequentially because the compute graph will store the gradients for each slice.
Instead maybe return some n=10 no of slides at a time

ALso, should I pretrain a 2d model for the dataset and then load it into the extended class.

It was observed using a software like napier, slice of real image vs slice of man_seg image that adjusting contrast limits, gamma correction.
See napierView.py
Possibility of gamma and contrast limiting as a learnable parameter

22.01.2026
Perhaps the above things are auto learned by the backbone feature maps.

Experiment Observations:
- Per FPN MLP performed worse than global slice MLP. The reasoning for this that I understand is:
The instances in a slice shift by some distance or change size as you  move through the slices. Now we are introducing a different weighting across FPN levels. Same spatial location corresponds to the dame object instance, but viewed at different scales. '0'(x, y) ~ '1'(x/2, y/2) ~ '2'(x/4, y/4) ~ '3'(x/8, y/8). But each of these are feature maps of the same instance. 
If we have different weights w1=(w11, w12, w13), w2=(w21, w22, w23), w3=(w31, w32, w33) and so on for different fpn levels, we will have each level influencing which slice to choose.
For eg. for '0', say w12 > w11 and w13, but for '2' w31 > w32 and w33.
Thus when the slices are weighted by these weights we have center slice at level 0 but -1th slice at level 2. But we know that the instances shift or change in size across slices, so this causes inconsistent spatial representation especially for box prediction.

In the case of simple Slice SE mlp fusion, each fpn level affects the same slice weight, so across all fpn levels slice weighting is same. Nevertheless, it hasn't given a significant increase in score.

In teh case of Pixel wise Slice SE mlp fusion, we are weighting for each pixel, which slice to choose. This converged much sooner than globalSE but did not surpass it eventually. This is because ...
/22.01.2026

28.01.2026
Printed out the weights and saw that the logits dynamics are much smaller than the static_logits. THis is the case even for zero_bias. Thus the softmay is dominated by the static logits and my hypothesis is by doing so the model is not learning to use these logit_dynamics at all. So as pyrt of A/B testing, try removing the static logits.
/28.01.2026


# DataFlow and Architecture So Far

### emrDataset() _class
This class is defined under the emrDataset.py file and inherits from torch.utils.data.dataset to load the dataset
THe load_from_cache parameter if set to False, will recreate 2d slices from the 3d volume, making the algorithm slow. THis can be used once and then the 2d slices can directly be referenced for future time by setting load_from_cache =True for subsequent runs.

**TODO: Put this in the dataSplitter.py instead and remove this complication from the dataset. rename datasplitter to preprocess_dataset or smthng.**

### emrDataLoader - DataloaderBuilder _class
Returns dataset and dataloader based on the config file. expects emrConfigManager object.

### emrConfigManager() _class
Reads the config file and contains some helper functions for reading fromt he file
### createExperimentFolder _func
Creates a folder for the experiment where the log file, and the config file will be stored.

**TO DO: save the results after testing in this folder.**
Save vaildation results also in this folder

### emrMetrics _class

### emrModelBuilder
reads from the cfg (config file- emrConfigManager object) to build a model using parametrs in the config file. CAn also load the model from a checkpoint path if specified in the config file. 

**TODO: remove start_epochs from config_file.**

### COnfig file template
**TODO: CHANGE THIS. TOO BAD: CONFUSING!!!**
```
[EXPERIMENT]
exp_root = Experiments/
exp_name = EarlyFusion_N5_${LOOP:start_epochs}_to_15_epochs

[DATASET]
imgs_dir = Fluo-N3DH-SIM+_joined/train/imgs
masks_dir = Fluo-N3DH-SIM+_joined/train/masks
load_from_cache = 1

[LOOP]
start_epochs = 10
num_epochs = 5
learning_rate = 1e-5
weight_decay = 1e-7
batch_size = 4
save_path = saved_models/${EXPERIMENT:exp_name}
ckpt_path = saved_models/EarlyFusion_N5_10_epochs(10_outof_10).pt

[MODEL]
num_slices_per_batch = 5
rpn_positive_fraction=0.5

```


### training_loop.py
**TODO: add validation, compute loss on validation set, and metrics. Think about how to add it in the config file**
Maybe just add all train_dirs, test_dirs, val_dirs in the file, and let _"mode"_ decide .


## Mask-RCNN Theory
RCNN -  Region based Convolutional Neural Network / regions with CNN features - 

Uses selective search to extrsct 2000 regions from the image (regions proposals), then uses greedy Non maximum supressions to combine similar regions into larger ones. Then the regions ar earped to a standard size and sent to Convolutional neural nets (feature extractor), then classify with SVM, adn bounding box regressor.
Faster RCNN : Regions Proposal Network

Image -> Convolutions -> feature maps -> Regions Proposal Network  -> proposals on the feature maps  ->  ROI pooling -> classifier
                                    ------------------------------------------------------------------------^

Mask RCNN

after RPN ( using ROI align) we have Regions of Interests each of a fixed size. For an input slice  


## Understanding Instance Segmentation
so we have a 2D image - (H, W) and a corresponding mask (H, W) with 5 unique greyscale values applied to each instance.
Now our masks are transformed to be (5,H,W) each layer corresponding to the unique instance - binary masks

Our model recieves the image and predicts - 5, 1, H, W masks because the RPN prooposes say 5 regions for which score > threshold(0.5)

What if our model predicts along with this - an instance_id - 1, 2, 3, 4, 5 for slice 0

and say the next slice only has 4 instances - so now the model output masks is say correct - 4, 1, H, W
and the gt instance_id is 1, 2, 4, 5, i.e. instance 3 no longer exists in this slice

the model should somehow use previous result of 1,2,3,4,5 and previous predicted boxes and current predicted boxes to give out 1, 2, 4, 5
Some sort of regression on past predicted boxes, current predicted boxes and past predicted instance_ids?

## WHat is dice loss


## Architecture Understanding

Okay so [MaskRCNN](/home/rohan/.pyenv/versions/venv/lib/python3.11/site-packages/torchvision/models/detection/mask_rcnn.py) init function initializes 3 main things:
- mask_roi_pool = MultiScaleRoIAlign
- mask_head = MaskRCNNHeads
- mask_predictor = MaskRCNNPredictor
These are the ROIheads. The roi heads is initialized in FasterRCNN Class.


The backbone is fixed to resnet50 with IMAGENET1K_V1 weights.
in the init function of Slice_MaskRCNN in model.py, check if modifying the learnable params from 5 to larger is necessary here.
``` python
    trainable_backbone_layers = _validate_trainable_layers(is_trained, trainable_backbone_layers, 5, 3)
```

The MaskRCNN inherits from FasterRCNN which inherits from GeneralizedRCNN which inherits from the base nn.Module. The forward function is defined in the GeneralizedRCNN.


### Backbone
####  [FPN](/home/rohan/.pyenv/versions/venv/lib/python3.11/site-packages/torchvision/ops/feature_pyramid_network.py)

### [GeneralizedRCNN](/home/rohan/.pyenv/versions/venv/lib/python3.11/site-packages/torchvision/models/detection/generalized_rcnn.py)

init recieves
- backbone (nn.Module)
- rpn (nn.Module)
- roi_heads (nn.Module): takes the features + the proposals from the RPN and computes detections / masks from it.
- transform (nn.Module)

The forward function does
1. Extract features by passing the image through the backbone: features = self.backbone(images.tensors)
2. Get proposals (bounding boxes) from the Region Proposal Network and corresponding losses: proposals, proposal_losses = self.rpn(images, features, targets)
3. Get detections: self.roi_heads(features, proposals, images.image_sizes, targets)
```json
[
  {
    "boxes": [[12.0, 34.0, 56.0, 78.0], [90.0, 45.0, 120.0, 160.0]],
    "labels": [1, 2],
    "scores": [0.98, 0.87],
    "masks": "Tensor of shape [2, 1, 28, 28]"
  },
  {
    "boxes": [[15.0, 25.0, 60.0, 80.0]],
    "labels": [3],
    "scores": [0.92],
    "masks": "Tensor of shape [1, 1, 28, 28]"
  }
]
```

i.e., for each input ``(B, H, W)`` (C = 1 and unsqueezed as we are working with greyscale; change the GeneralizedRCNNTransform), we will get some ``m1, m2, ..mb`` detections, i.e. boxes, masks, corresponding labels, scores of some fixed size, i.e. 28*28 (unless changed in GeneralizedRCNNTransform.postprocess)


### [RPNHead](/home/rohan/.pyenv/versions/venv/lib/python3.11/site-packages/torchvision/models/detection/rpn.py)
RPN head recieves feature maps from the anchor generator from the backbone, passes them through a series of conv layers to get the objectness score (cls_logits) and bounding box regression values for each anchor box.

``` python
self.conv = nn.Sequential(*convs)
self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)
```

 
### [FasterRCNN](/home/rohan/.pyenv/versions/venv/lib/python3.11/site-packages/torchvision/models/detection/faster_rcnn.py)

init recieves:
- backbone (nn.Module)
- rpn params like - rpn_pre/post_nms_top_n_train, rpn_fg_iou_thresh, rpn_positive_fraction etc etc
- box_nms_thresh, box_score_thresh, box_fg_iou_thresh etc etc

The backbone gives a feature map out of which the rpn_anchor_generator creates anchor boxes at different scales and aspect ratios. Default anchors are created with 5 sizes (32, 64, 128, 256, 512) and 3 aspect ratios (0.5, 1.0, 2.0). 
The RPN uses RPN heads to classify each anchor box as object or not object (objectness score) and regresses the bounding box coordinates and uses NMS to filter proposals. The proposals are sent to the roi_heads for final classification and mask prediction.

The box_roi_pool is initialized as MultiScaleRoIAlign with output_size=7 (7x7 feature map for each proposal) and sampling_ratio=2.
The box_head is initialized as TwoMLPHead with representation_size=1024 (2 fully connected layers of size 1024 each).
The box_predictor is initialized as FastRCNNPredictor. 


### [RoIHeads](/home/rohan/.pyenv/versions/venv/lib/python3.11/site-packages/torchvision/models/detection/roi_heads.py)
The RoIHead takes 
- box_roi_pool,
- box_head,
- box_predictor,



### [MaskRCNN](/home/rohan/.pyenv/versions/venv/lib/python3.11/site-packages/torchvision/models/detection/mask_rcnn.py)
init receives:
- backbone (nn.Module)
- mask_roi_pool = MultiScaleRoIAlign
- mask_head = MaskRCNNHeads
- mask_predictor = MaskRCNNPredictor
- fasterRCNN and generalizedRCNN init params - rpn_head, rpn_anchor_generator, rpn_pre_nms_top_n_train, rpn_post_nms_top_n_train, box_roi_pool, box_head, box_predictor.


So to combine, we can say: 
GeneralizedRCNN init only takes the nn.Module - backbone, rpn, roi_heads, transform
THe FasterRCNN init takes backbone, transform parameters, RPN parameters and box parameters
The maskRCNN init takes backbone, transform parameters, RPN parameters, box parameters and Mask Parameters


NOTE: you could also try conv3dNormActivation for early fusion.

# Post RPN Fusion
See mask_head in roi_heads.py, or class MaskRCNNHeads in mask_rcnn.py.
If we pass per slice list of features and list of proposals instead of fused, we could implement some dynamic fusion here.

### Option 1 [MaskRCNNHeads](emrmodel/mask_rcnn.py)


        if self.mask_roi_pool is not None:
            mask_features = self.mask_roi_pool(features, mask_proposals, image_shapes) # MultiScaleROIAlign - This takes the mask proposals and samples a 7*7 output from each feature map.
            mask_features = self.mask_head(mask_features) # does 256 in_channel to 256 out_channel Sequential Convs on the 256 channel 7*7 features
            mask_logits = self.mask_predictor(mask_features) # funnels 256 channels into num_classes
            
expects, in_channels = backbone.out_channels = 256
nn.Sequential with 4 blocks, each a conv2d with in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, dilation=1

o = ((i + 2p - d(k-1) - 1) / s) + 1
o = (256 +2 - 2 - 1) / 1 +1 = 256

so i = o

we could here use: [conv3dNormActivation](/home/rohan/.pyenv/versions/venv/lib/python3.11/site-packages/torchvision/ops/misc.py)
Conv3d, remember that your input tensor must have 5 dimensions:$$(\text{Batch}, \text{Channels}, \text{Depth/Time}, \text{Height}, \text{Width})$$


matched_idxs is a list of tensors. Each tensor corresponds to one image in a batch and contains integer indices that map each proposal (region of interest) to the best-matching ground-truth bounding box from the dataset targets.
FOR NOW just use only the center slice matched_idxs

IN option 1, I am computing the box_regression for each slice, then box_loss_regression is computed by comparing with target["boxes"] of the center slice only. So losses are also svaled based on how far the slice is from the center slice, the boxes predicted in the neighbouring slices are not 

### Option 2
Do we need to pass all slices through rpn to get proposals for each slice? Or proposals only from the center slice can be used?


### Option 3
For proposals from each slice we can compute a pairwise IoU to have a one to -one matching with neighbouring proposals. Then we do a fusion.

## THings to do after PostRPNFusion:
1. Try other datasets.

## Things to think about
1. #### targets
During forward pass you are always passing the center slice as the target. Now consider the RPN, where anchors are matched to the target. The loss is calculated by comparing the target with the proposals which are created by looking at the feature maps of the slice.
This begs the question, what when the cells shift. Then you are trying to compare with the wrong target boxes. 
This leaves us with two options
  a. Pass the correct target for each slice. This means rewrite the emrDataLoader, datasetGenerator, emrDataset, and the forward pass the generalized_rcnn. Too much work ,maybe for later.
  b. Assume that the boxes are not shifting that much and use the fused feature maps from early fusion. But that means u would be using the same feature maps, it doesnt make sense to pass it 3 times then. Okay, if a proposal belongs to a neighbour slice and does not belong in the target, then it will give a high loss and confuse the model, because in the center slice the same behaviour gave a lower loss. But since there are multiple neighbouring slices, atleast for this dataset where the anisotropy is low, the target should belong in majority of the slices. 
  Would it benefit to scale the losses then? i.e., an incorrect classification in the center slice should be penalized, but in the neighbouring slices the loss could be scaled down by a factor of how far it is from the center slice. 
  THis might be a lazy fix , if at all it works, as it may not help when anisotropy is high.
  Can this be interpreted as a case of noisy datalabels. If yes then how could I modify my training to handle this?

2. #### Masked training?
  SINce we already have the targets for each slice, What if we included the target of the neighbour slices as well? THne instead of just predicting the center slice masks, we could mask random targets, and use transformer like training to reproduce the masked target. THen at inference some way of doing this without having the target masks at all.?

3. #### earlyMLPFusion slice weight patterns - possible bugs:
For the Slice Squeeze and Excite Experiments I am observing a pattern. IN this experiment I use global slice fusion, i.e. take mean across H, W out of [B, S, C, H, W] to get [B, S, C] and pass that to an MLP to get logits and then add static_bias  which look like a gaussian, assigning highest weight to the center slice, and then taking softmax to get per slice_weights. I observe that always, the last slice is weighted highest. For eg., in the 3 slice case I observed the mean of the final slice weights when recorded for the test set, as [0.15, 0.38, 0.47]. FOr the 7 slice case I observe:
0.12048158 0.011455724

0.10890814 0.007848853

0.09320539 0.00485457

0.16590582 0.0061298353

0.101940885 0.003744808

0.15724547 0.0074364166

0.25231272 0.016438028

This seems like a pattern now. What could be the reason the last slice is always weighted higher.

These are observations for when the static_bias is added after the logits as a free variable. However when it is not added the each of slice_weights just take 1/num_slices value, causing slice averaging as the fusion strategy.

An point of failure is seen from the observation that, when I printed out these weights, these are hardly dynamic. In the sense it is learning it as a rule these slice weights. When I am passing the first slice of the volume, I have to pad it with a zero slice. HOwever, even in this case are the slice weights not drasatically different than when the slice is available. ALso when observing the slice weights during the pixelwiseSliceSEFusion if I plot the slice weights when target is the first slice, then it slice -1 is a 0 slice but it is not seen so in when the slice_wights are plotted (here the slice weights are (S,H,W)), infact the cells acan be seen in the features maps of the slice -1. - What is causing this..??
