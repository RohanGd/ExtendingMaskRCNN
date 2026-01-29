import torch
from torch.utils.data import DataLoader
from emrDataset import emrDataset, emrCollate_fn

class DataloaderBuilder:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger
        self.imgs_dir, self.masks_dir = dict(), dict()

    def build(self, mode="train"):
        dataset = self.cfg.get("DATASET", "dataset_name")
        self.imgs_dir[mode] = "datasets/" + dataset + "/" + mode + "/imgs"
        self.masks_dir[mode] = "datasets/" + dataset + "/" + mode + "/masks"

        num_slices_per_batch = self.cfg.get_int("MODEL", "num_slices_per_batch")
        batch_size = self.cfg.get_int("LOOP", "batch_size", 1) 

        dataset = emrDataset(
            imgs_dir=self.imgs_dir[mode],
            masks_dir=self.masks_dir[mode],
            n=num_slices_per_batch,
            logger=self.logger,
            mode=mode
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True if mode == "train" else False,
            collate_fn=emrCollate_fn,
            generator=torch.manual_seed(42),
        )

        self.logger.info(
            f"Initialized {mode} loader: {len(dataloader)} batches | "
            f"Total slices: {len(dataset)} | batch_size={batch_size}"
        )

        return dataset, dataloader
