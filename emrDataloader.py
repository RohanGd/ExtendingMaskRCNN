import torch
from torch.utils.data import DataLoader
from emrDataset import emrDataset, emrCollate_fn

class DataloaderBuilder:
    def __init__(self, cfg, logger):
        self.cfg = cfg
        self.logger = logger

    def build(self, mode="train"):
        imgs_dir = self.cfg.get("DATASET", "imgs_dir")
        masks_dir = self.cfg.get("DATASET", "masks_dir")
        num_slices_per_batch = self.cfg.get_int("MODEL", "num_slices_per_batch")
        batch_size = self.cfg.get_int("LOOP", "batch_size", 1) 
        load_from_cache = self.cfg.get_bool("DATASET", "load_from_cache", True)

        dataset = emrDataset(
            imgs_dir=imgs_dir,
            masks_dir=masks_dir,
            n=num_slices_per_batch,
            load_from_cache=load_from_cache,
            logger=self.logger,
            mode=mode
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=emrCollate_fn,
            generator=torch.manual_seed(42)
        )

        self.logger.info(
            f"Train loader: {len(dataloader)} batches | "
            f"Total slices: {len(dataset)} | batch_size={batch_size}"
        )

        return dataset, dataloader
