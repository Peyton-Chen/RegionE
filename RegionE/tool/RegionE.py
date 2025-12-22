config = {
    "FluxKontextPipeline": {"num_inference_steps":28, "warmup_step": 6, "post_step": 2, "refresh_step": "16", "threshold": 0.93, "cache_threshold": 0.04, "erosion_dilation": True},
    "Step1XEditPipeline": {"num_inference_steps":28, "warmup_step": 6, "post_step": 2, "refresh_step": "16", "threshold": 0.88, "cache_threshold": 0.02, "erosion_dilation": True},
    "Step1XEditPipelineV1P2": {"num_inference_steps":28, "warmup_step": 6, "post_step": 2, "refresh_step": "16", "threshold": 0.88, "cache_threshold": 0.02, "erosion_dilation": True},
    "QwenImageEditPipeline": {"num_inference_steps":28, "warmup_step": 6, "post_step": 2, "refresh_step": "16", "threshold": 0.80, "cache_threshold": 0.03, "erosion_dilation": True},
    "QwenImageEditPlusPipeline": {"num_inference_steps":28, "warmup_step": 6, "post_step": 2, "refresh_step": "16", "threshold": 0.80, "cache_threshold": 0.03, "erosion_dilation": True},
}

class RegionEHelper(object):
    def __init__(self, pipeline=None):
        if pipeline is not None: self.pipeline = pipeline
        self.name = self.pipeline.__class__.__name__
        self.config = config[self.name]

    def enable(self):
        assert self.pipeline is not None
        if self.name == "FluxKontextPipeline":
            from ..FluxKontext.inplace import warp_modules
        elif self.name == "Step1XEditPipeline":
            from ..Step1XEdit.inplace import warp_modules
        elif self.name == "Step1XEditPipelineV1P2":
            from ..Step1XEditV1P2.inplace import warp_modules
        elif self.name == "QwenImageEditPipeline":
            from ..QwenImageEdit.inplace import warp_modules
        elif self.name == "QwenImageEditPlusPipeline":
            from ..QwenImageEditPlus.inplace import warp_modules
        self.pipeline = warp_modules(self.pipeline, **self.config)

    def disable(self):
        assert self.pipeline is not None
        if self.name == "FluxKontextPipeline":
            from ..FluxKontext.inplace import unwarp_modules
        elif self.name == "Step1XEditPipeline":
            from ..Step1XEdit.inplace import unwarp_modules
        elif self.name == "Step1XEditPipelineV1P2":
            from ..Step1XEditV1P2.inplace import unwarp_modules
        elif self.name == "QwenImageEditPipeline":
            from ..QwenImageEdit.inplace import unwarp_modules
        elif self.name == "QwenImageEditPlusPipeline":
            from ..QwenImageEditPlus.inplace import unwarp_modules
        self.pipeline = unwarp_modules(self.pipeline)

    def set_params(self, num_inference_steps=28, warmup_step=None, post_step=None, refresh_step=None, threshold=None, cache_threshold=None, erosion_dilation=None):
        assert num_inference_steps == 28, "num_inference_steps must be 28"
        if warmup_step is not None: self.config['warmup_step'] = warmup_step
        if post_step is not None: self.config['post_step'] = post_step
        if refresh_step is not None: self.config['refresh_step'] = refresh_step
        if threshold is not None: self.config['threshold'] = threshold
        if cache_threshold is not None: self.config['cache_threshold'] = cache_threshold
        if erosion_dilation is not None: self.config['erosion_dilation'] = erosion_dilation
        print(f"RegionEHelper: set_params {self.config}")
