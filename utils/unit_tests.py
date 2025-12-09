def check_dataset_output():
    from emrDataset import emrDataset, emrCollate_fn
    from torch.utils.data import DataLoader
    import torch, torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn

    imgs_dir = "datasets/Fluo-N3DH-CHO/train/imgs"
    masks_dir = "datasets/Fluo-N3DH-CHO/train/masks"

    num_slices = 3
    dataset = emrDataset(imgs_dir, masks_dir, num_slices) # num_slice, n=5

    from torch.utils.data import DataLoader
    generator = torch.manual_seed(42)

    emrdataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=False, collate_fn=emrCollate_fn)

    i = 2
    for imgs, targets in emrdataloader:
        if i <= 0:
            break
        else:
            i-=1
        print(imgs.shape)
        print([t['image_id'].item() for t in targets]) 

    model = maskrcnn_resnet50_fpn(weights=None)
    model.backbone.body.conv1 = torch.nn.Conv2d(num_slices, 64, kernel_size=7, stride=2, padding=3, bias=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, 2) # binary mask -  2classes
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        model.roi_heads.mask_predictor.conv5_mask.in_channels, 256, 2
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    imgs = imgs.to(device)
    targets = [ {k:v.to(device) for k, v in t.items()} for t in targets]
    print(model(imgs, targets))
    model.eval()
    preds = model(imgs)
    for pred in preds:
        print(pred['labels'].unique(return_counts=True))

def check_model_forward_pass():
    from emrmodel.extended_mask_rcnn import ExtendedMaskRCNN
    from emrDataset import emrDataset, emrCollate_fn
    import torch
    from torch.utils.data import DataLoader


    imgs_dir = "datasets/Fluo-N3DH-CHO/train/imgs"
    masks_dir = "datasets/Fluo-N3DH-CHO/train/masks"

    num_slices = 3
    dataset = emrDataset(imgs_dir, masks_dir, num_slices , load_from_cache=True) # num_slice, n=5

    generator = torch.manual_seed(42)
    emrdataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=False, collate_fn=emrCollate_fn)

    model = ExtendedMaskRCNN(3)

    imgs, targets = next(iter(emrdataloader))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()
    imgs = imgs.to(device)
    targets = [ {k:v.to(device) for k, v in t.items()} for t in targets]
    print(model(imgs, targets))
    model.eval()
    preds = model(imgs)
    for pred in preds:
        print(pred['labels'].unique(return_counts=True))



def check_targets():
    from emrDataset import emrDataset, emrCollate_fn
    import torch
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    
    imgs_dir = "datasets/Fluo-N3DH-CHO/train/imgs"
    masks_dir = "datasets/Fluo-N3DH-CHO/train/masks"

    num_slices = 3
    dataset = emrDataset(imgs_dir, masks_dir, num_slices , load_from_cache=True) # num_slice, n=5
    emrdataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, collate_fn=emrCollate_fn)

    images, targets = next(iter(emrdataloader))
    for target in targets:
        print(target['masks'].shape)
        for i in range(target['masks'].shape[0]):
            t = target['masks'][i]
            plt.figure(figsize=(10,10), cmap='grey')
            plt.imshow(t)
            plt.show()   

def check_consecutive_targets_maintain_spatial_consistency():
    from emrDataset import emrDataset, emrCollate_fn
    import torch
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt

    imgs_dir = "datasets/Fluo-N3DH-CHO/train/imgs"
    masks_dir = "datasets/Fluo-N3DH-CHO/train/masks"

    num_slices = 3
    dataset = emrDataset(imgs_dir, masks_dir, num_slices , load_from_cache=True) # num_slice, n=5
    emrdataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, collate_fn=emrCollate_fn)

    stop_iter = dataset.v_size
    volume_0_masks = list() # [(k1, H, W), (k2, H, W), (k3, H, W)]
    volume_0_ids = list()
    for images, targets in emrdataloader:
        stop_iter -= 1
        target = targets[0] # bcz batch_size = 1
        volume_0_masks.append(target['masks']) # k1, H, W
        volume_0_ids.append(target['image_id'])
        if stop_iter < 0:
            break
    
    # eg. for each slice print the 3rd mask. I printed the 
    for i in range(0, 5):
        mask_no_3 = volume_0_masks[i][3]
        plt.imshow(mask_no_3)
        plt.show()

    
    # volume_0_masks = torch.stack(volume_0_masks)
    # v0_m_shape = volume_0_masks.shape
    # volume_0_masks = volume_0_masks.view((v0_m_shape[1], v0_m_shape[0], v0_m_shape[2], v0_m_shape[3]))

    # for i in range(v0_m_shape[1]):
    #     curr_mask = volume_0_masks[i]
    #     plt.figure(figsize=(50, 50))
    #     plt.subplot(1, v0_m_shape[0])
    #     for j in range(v0_m_shape[1]):
    #         plt.axes(1, j, 3, 3)
    #         plt.imshow(volume_0_masks[i][j])
    #     plt.show()
    # pass


if __name__ == "__main__":
    check_dataset_output()
    check_targets()
    check_consecutive_targets_maintain_spatial_consistency()
    check_model_forward_pass()
