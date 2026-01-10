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

Okay so [MaskRCNN](venv/Lib/site-packages/torchvision/models/detection/mask_rcnn.py) init function initializes 3 main things:
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
####  [FPN](venv/Lib/site-packages/torchvision/ops/feature_pyramid_network.py)

### [GeneralizedRCNN](venv/Lib/site-packages/torchvision/models/detection/generalized_rcnn.py)

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


### [RPNHead](venv/Lib/site-packages/torchvision/models/detection/rpn.py)
RPN head recieves feature maps from the anchor generator from the backbone, passes them through a series of conv layers to get the objectness score (cls_logits) and bounding box regression values for each anchor box.

``` python
self.conv = nn.Sequential(*convs)
self.cls_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=1, stride=1)
self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=1, stride=1)
```

 
### [FasterRCNN](venv/Lib/site-packages/torchvision/models/detection/faster_rcnn.py)

init recieves:
- backbone (nn.Module)
- rpn params like - rpn_pre/post_nms_top_n_train, rpn_fg_iou_thresh, rpn_positive_fraction etc etc
- box_nms_thresh, box_score_thresh, box_fg_iou_thresh etc etc

The backbone gives a feature map out of which the rpn_anchor_generator creates anchor boxes at different scales and aspect ratios. Default anchors are created with 5 sizes (32, 64, 128, 256, 512) and 3 aspect ratios (0.5, 1.0, 2.0). 
The RPN uses RPN heads to classify each anchor box as object or not object (objectness score) and regresses the bounding box coordinates and uses NMS to filter proposals. The proposals are sent to the roi_heads for final classification and mask prediction.

The box_roi_pool is initialized as MultiScaleRoIAlign with output_size=7 (7x7 feature map for each proposal) and sampling_ratio=2.
The box_head is initialized as TwoMLPHead with representation_size=1024 (2 fully connected layers of size 1024 each).
The box_predictor is initialized as FastRCNNPredictor. 


### [RoIHeads](venv/Lib/site-packages/torchvision/models/detection/roi_heads.py)
The RoIHead takes 
- box_roi_pool,
- box_head,
- box_predictor,



### [MaskRCNN](venv/Lib/site-packages/torchvision/models/detection/mask_rcnn.py)
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


