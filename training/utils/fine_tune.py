def freeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad_(False)

def unfreeze_backbone(model):
    for p in model.backbone.parameters():
        p.requires_grad_(True)
