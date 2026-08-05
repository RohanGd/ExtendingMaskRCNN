import csv
from emrConfigManager import emrConfigManager, setup_logger
from emrModelBuilder import ModelBuilder
from pathlib import Path


def count_params(module):
    if module is None:
        return 0
    return sum(p.numel() for p in module.parameters())


def get_param_breakdown(name, model):
    backbone_params = count_params(model.backbone)
    rpn_params = count_params(model.rpn)
    early_fusion_params = count_params(getattr(model, "early_mlp_fusion_module", None))
    late_fusion_params = count_params(getattr(model.roi_heads, "mask_features_fusion", None))
    roi_heads_params = count_params(model.roi_heads) - late_fusion_params
    fusion_params = early_fusion_params + late_fusion_params
    total_params = sum(p.numel() for p in model.parameters())
    accounted = backbone_params + rpn_params + roi_heads_params + fusion_params
    other_params = total_params - accounted

    return {
        "config": name,
        "backbone": backbone_params,
        "rpn": rpn_params,
        "roi_heads_ex_fusion": roi_heads_params,
        "early_fusion": early_fusion_params,
        "late_fusion": late_fusion_params,
        "fusion_total": fusion_params,
        "other": other_params,
        "total": total_params,
    }


logger = setup_logger("model_params.log", "model_params")

configs = {
    "channel fusion (resnet50)": "config/channelFusion.ini",
    "swin channel fusion": "config/swin_channelFusion.ini",
    "early fusion - Global (Gaussian bias)": "config/earlyFusion_GlobalGaussian.ini",
    "early fusion - GlobalDeep (Gaussian bias)": "config/earlyFusion_GlobalGaussianDeep.ini",
    "late fusion - only_center": "config/lateFusion_onlyCenter.ini",
}

rows = []
for name, cfg_path in configs.items():
    cfg = emrConfigManager(Path(cfg_path))
    model = ModelBuilder(cfg, logger).load_model()
    rows.append(get_param_breakdown(name, model))

out_path = Path("model_params.csv")
with open(out_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote param breakdown for {len(rows)} configs to {out_path}")
