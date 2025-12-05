import torch
from torch.utils.data import DataLoader
from emrDataset import emrDataset, emrCollate_fn

class DataloaderBuilder:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

    def build(self, mode="train"):
        dataset = self.cfg.get("DATASET", "dataset_name")
        imgs_dir = "datasets/" + dataset + "/" + mode + "/imgs"
        masks_dir = "datasets/" + dataset + "/" + mode + "/masks"

        num_slices_per_batch = self.cfg.get_int("MODEL", "num_slices_per_batch")
        batch_size = self.cfg.get_int("LOOP", "batch_size", 1) 

        dataset = emrDataset(
            imgs_dir=imgs_dir,
            masks_dir=masks_dir,
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
            num_workers=12,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

        self.logger.info(
            f"Train loader: {len(dataloader)} batches | "
            f"Total slices: {len(dataset)} | batch_size={batch_size}"
        )

        return dataset, dataloader
