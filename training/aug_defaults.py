"""[2026-08-27] Weak photometric augmentation for training only (XVLA_AUG=1): brightness ±8%, contrast ±8%, saturation ±4%, sharpness 0.8–1.2,
NO hue, NO geometric (lerobot's default tfs include RandomAffine → removed). Overrides ImageTransformsConfig.__post_init__ so any parsed config gets these."""
import os
def install():
    if os.environ.get("XVLA_AUG", "0") != "1": return
    from lerobot.transforms.transforms import ImageTransformsConfig, ImageTransformConfig
    def tfs():
        return {"brightness": ImageTransformConfig(weight=1.0, type="ColorJitter", kwargs={"brightness": (0.92, 1.08)}),
                "contrast": ImageTransformConfig(weight=1.0, type="ColorJitter", kwargs={"contrast": (0.92, 1.08)}),
                "saturation": ImageTransformConfig(weight=1.0, type="ColorJitter", kwargs={"saturation": (0.96, 1.04)}),
                "sharpness": ImageTransformConfig(weight=0.5, type="SharpnessJitter", kwargs={"sharpness": (0.8, 1.2)})}
    orig_init = ImageTransformsConfig.__init__
    def new_init(self, *a, **k):
        orig_init(self, *a, **k)
        self.enable = True; self.max_num_transforms = 3; self.random_order = False; self.tfs = tfs()
        print(f"[aug] ImageTransformsConfig forced: enable={self.enable} tfs={list(self.tfs)} (hue/affine OFF)", flush=True)
    if not getattr(ImageTransformsConfig, "_aug_patched", False):
        ImageTransformsConfig.__init__ = new_init; ImageTransformsConfig._aug_patched = True
    print("[aug] weak photometric transforms forced ON (b±.08 c±.08 s±.04 sharp .8-1.2; hue/affine OFF)", flush=True)
