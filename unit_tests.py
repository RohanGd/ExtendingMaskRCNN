def check_dataset_output():
    from emrDataset import emrDataset, emrCollate_fn
    from torch.utils.data import DataLoader
    import torch, torchvision
    from torchvision.models.detection import maskrcnn_resnet50_fpn

    imgs_dir = "Fluo-N3DH-CHO/01"
    masks_dir = "Fluo-N3DH-CHO/01_ST/SEG"

    num_slices = 3
    dataset = emrDataset(imgs_dir, masks_dir, num_slices , load_from_cache=True) # num_slice, n=5

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


    imgs_dir = "Fluo-N3DH-CHO/01"
    masks_dir = "Fluo-N3DH-CHO/01_ST/SEG"

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



if __name__ == "__main__":
    # check_dataset_output()
    check_model_forward_pass()
