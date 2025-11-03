def check_dataset_output():
    from emrDataset import emrDataset, emrCollate_fn
    from torch.utils.data import DataLoader
    import torch

    imgs_dir = "Fluo-N3DH-CHO/01"
    masks_dir = "Fluo-N3DH-CHO/01_ST/SEG"
    dataset = emrDataset(imgs_dir, masks_dir, 5) # num_slice, n=5

    from torch.utils.data import DataLoader
    generator = torch.manual_seed(42)

    emrdataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=False, collate_fn=emrCollate_fn)

    i = 5
    for imgs, targets in emrdataloader:
        if i < 0:
            break
        else:
            i-=1
        print(imgs.shape)
        print([t['image_id'].item() for t in targets]) 


if __name__ == "__main__":
    check_dataset_output()