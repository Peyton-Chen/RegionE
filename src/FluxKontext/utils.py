import torch
import inspect
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass
from diffusers import FluxKontextPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.flux import FluxPipelineOutput
from diffusers.utils import BaseOutput, is_torch_xla_available
from typing import Optional, Union, List, Dict, Any, Callable, Tuple

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False
    
PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]

def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


@dataclass
class FlowMatchEulerDiscreteSchedulerOutput(BaseOutput):
    """
    Output class for the scheduler's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: torch.FloatTensor


@torch.no_grad()
def pipeline_call(
    self,
    image: Optional[PipelineImageInput] = None,
    prompt: Union[str, List[str]] = None,
    prompt_2: Optional[Union[str, List[str]]] = None,
    negative_prompt: Union[str, List[str]] = None,
    negative_prompt_2: Optional[Union[str, List[str]]] = None,
    true_cfg_scale: float = 1.0,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    sigmas: Optional[List[float]] = None,
    guidance_scale: float = 3.5,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    negative_ip_adapter_image: Optional[PipelineImageInput] = None,
    negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pil",
    return_dict: bool = True,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 512,
    max_area: int = 1024**2,
    _auto_resize: bool = True,
):
    multiple_of = self.vae_scale_factor * 2

    # 1. Preprocess image
    if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
        img = image[0] if isinstance(image, list) else image
        image_height, image_width = self.image_processor.get_default_height_width(img)
        aspect_ratio = image_width / image_height
        if _auto_resize:
            # Kontext is trained on specific resolutions, using one of them is recommended
            _, image_width, image_height = min(
                (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
            )
        image_width = image_width // multiple_of * multiple_of
        image_height = image_height // multiple_of * multiple_of
        image = self.image_processor.resize(image, image_height, image_width)
        image = self.image_processor.preprocess(image, image_height, image_width)
        height, width = image.shape[-2], image.shape[-1]

    else:
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width  or self.default_sample_size * self.vae_scale_factor

        original_height, original_width = height, width
        aspect_ratio = width / height
        width = round((max_area * aspect_ratio) ** 0.5)
        height = round((max_area / aspect_ratio) ** 0.5)

        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

    # 2. Check inputs. Raise error if not correct
    self.check_inputs(
        prompt,
        prompt_2,
        height,
        width,
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt_2,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    # 3. Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    lora_scale = (
        self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
    )
    has_neg_prompt = negative_prompt is not None or (
        negative_prompt_embeds is not None and negative_pooled_prompt_embeds is not None
    )
    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
    (
        prompt_embeds,
        pooled_prompt_embeds,
        text_ids,
    ) = self.encode_prompt(
        prompt=prompt,
        prompt_2=prompt_2,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
        lora_scale=lora_scale,
    )
    if do_true_cfg:
        (
            negative_prompt_embeds,
            negative_pooled_prompt_embeds,
            negative_text_ids,
        ) = self.encode_prompt(
            prompt=negative_prompt,
            prompt_2=negative_prompt_2,
            prompt_embeds=negative_prompt_embeds,
            pooled_prompt_embeds=negative_pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
            lora_scale=lora_scale,
        )

    # 4. Prepare latent variables
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, image_latents, latent_ids, image_ids = self.prepare_latents(
        image,
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    if image_ids is not None:
        latent_ids = torch.cat([latent_ids, image_ids], dim=0)  # dim 0 is sequence dimension

    # 5. Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps) if sigmas is None else sigmas
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.get("base_image_seq_len", 256),
        self.scheduler.config.get("max_image_seq_len", 4096),
        self.scheduler.config.get("base_shift", 0.5),
        self.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
    self._num_timesteps = len(timesteps)

    # handle guidance
    if self.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    else:
        guidance = None

    if (ip_adapter_image is not None or ip_adapter_image_embeds is not None) and (
        negative_ip_adapter_image is None and negative_ip_adapter_image_embeds is None
    ):
        negative_ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
        negative_ip_adapter_image = [negative_ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

    elif (ip_adapter_image is None and ip_adapter_image_embeds is None) and (
        negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None
    ):
        ip_adapter_image = np.zeros((width, height, 3), dtype=np.uint8)
        ip_adapter_image = [ip_adapter_image] * self.transformer.encoder_hid_proj.num_ip_adapters

    if self.joint_attention_kwargs is None:
        self._joint_attention_kwargs = {}

    image_embeds = None
    negative_image_embeds = None
    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = self.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
        )
    if negative_ip_adapter_image is not None or negative_ip_adapter_image_embeds is not None:
        negative_image_embeds = self.prepare_ip_adapter_image_embeds(
            negative_ip_adapter_image,
            negative_ip_adapter_image_embeds,
            device,
            batch_size * num_images_per_prompt,
        )

    # 6. Denoising loop
    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            if self.interrupt:
                continue

            self._current_timestep = t
            if image_embeds is not None:
                self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

            latent_model_input = latents
            if image_latents is not None:
                latent_model_input = torch.cat([latents, image_latents], dim=1)
            timestep = t.expand(latents.shape[0]).to(latents.dtype)

            noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timestep / 1000,
                guidance=guidance,
                pooled_projections=pooled_prompt_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                joint_attention_kwargs=self.joint_attention_kwargs,
                return_dict=False,
            )[0]
            noise_pred = noise_pred[:, : latents.size(1)]

            if do_true_cfg:
                if negative_image_embeds is not None:
                    self._joint_attention_kwargs["ip_adapter_image_embeds"] = negative_image_embeds
                neg_noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=negative_pooled_prompt_embeds,
                    encoder_hidden_states=negative_prompt_embeds,
                    txt_ids=negative_text_ids,
                    img_ids=latent_ids,
                    joint_attention_kwargs=self.joint_attention_kwargs,
                    return_dict=False,
                )[0]
                neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

            if callback_on_step_end is not None:
                callback_kwargs = {}
                for k in callback_on_step_end_tensor_inputs:
                    callback_kwargs[k] = locals()[k]
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)

            # call the callback, if provided
            if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

    self._current_timestep = None

    if output_type == "latent":
        image = latents
    else:
        latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        image = self.vae.decode(latents, return_dict=False)[0]
        image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    if not return_dict:
        return (image,)

    return FluxPipelineOutput(images=image)


def create_kernel(kernel_size=3, kernel_type='square'):
    """
    Create a morphological operation kernel (structuring element).
    
    Args:
        kernel_size (int): The size of the kernel, default is 3x3.
        kernel_type (str): Type of the kernel, either 'square' or 'cross'.
    
    Returns:
        torch.Tensor: The kernel matrix.
    """
    if kernel_type == 'square':
        # Square-shaped kernel
        kernel = torch.ones(1, 1, kernel_size, kernel_size)
    elif kernel_type == 'cross':
        # Cross-shaped kernel
        kernel = torch.zeros(1, 1, kernel_size, kernel_size)
        mid = kernel_size // 2
        kernel[0, 0, mid, :] = 1  # Horizontal line
        kernel[0, 0, :, mid] = 1  # Vertical line
    else:
        raise ValueError("kernel_type must be 'square' or 'cross'")
    
    return kernel


def morphological_erosion(image, kernel):
    """
    Morphological erosion operation.
    
    Args:
        image (torch.Tensor): Input binary image [H, W].
        kernel (torch.Tensor): Structuring element.
    
    Returns:
        torch.Tensor: Eroded image.
    """
    # Convert the image to a 4D tensor [1, 1, H, W]
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    
    # Perform erosion using convolution
    # Erosion: the center pixel is 1 only if all pixels covered by the kernel are 1
    kernel_size = kernel.shape[-1]
    padding = kernel_size // 2
    
    # Convert kernel and image to float type
    kernel = kernel.float()
    image = image.float()
    
    # Perform convolution
    conv_result = F.conv2d(image, kernel, padding=padding)
    
    # Erosion condition: convolution result equals the number of ones in the kernel
    kernel_sum = kernel.sum()
    eroded = (conv_result == kernel_sum).float()
    
    return eroded.squeeze()


def morphological_dilation(image, kernel):
    """
    Morphological dilation operation.
    
    Args:
        image (torch.Tensor): Input binary image [H, W].
        kernel (torch.Tensor): Structuring element.
    
    Returns:
        torch.Tensor: Dilated image.
    """
    # Convert the image to a 4D tensor [1, 1, H, W]
    if image.dim() == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    
    kernel_size = kernel.shape[-1]
    padding = kernel_size // 2
    
    # Convert kernel and image to float type and move kernel to the same device as image
    kernel = kernel.float().to(image)
    image = image.float()
    
    # Perform convolution
    conv_result = F.conv2d(image, kernel, padding=padding)
    
    # Dilation condition: convolution result greater than 0
    dilated = (conv_result > 0).float()
    
    return dilated.squeeze()


def remove_scattered_points(binary_matrix, kernel_size=3, kernel_type='square'):
    """
    Remove isolated points in a binary matrix.
    
    Args:
        binary_matrix (torch.Tensor): Input binary matrix [H, W].
        kernel_size (int): Size of the kernel.
        kernel_type (str): Type of the kernel.
    
    Returns:
        torch.Tensor: Processed matrix with scattered points removed.
    """
    # Create structuring elements
    erosion_kernel = create_kernel(3, 'cross').to(binary_matrix)
    dilation_kernel = create_kernel(5, kernel_type).to(binary_matrix)
    
    # First, perform erosion
    eroded = morphological_erosion(binary_matrix, erosion_kernel)
    
    # Then, perform dilation
    result = morphological_dilation(eroded, dilation_kernel)

    return result


def ids_scatter(gathered_latent, ids, src) -> torch.Tensor:
    """
    Scatter gathered latent vectors back into their original positions.
    
    Args:
        gathered_latent (torch.Tensor): [batch_size, k, dim] - latent vectors to scatter.
        ids (torch.Tensor): [batch_size, k] - target indices where to place the latent vectors.
        src (torch.Tensor): [batch_size, seq_length, dim] - target matrix to store the scattered vectors.
    
    Returns:
        torch.Tensor: Updated target matrix [batch_size, seq_length, dim] with scattered values.
    """
    B, K, D = gathered_latent.shape

    # Use scatter to place gathered_latent back to the corresponding positions
    src[torch.arange(B).unsqueeze(1), ids] = gathered_latent

    return src  # [B, seq_length, D]


def ids_gather(latent, ids, rope=False, condition_length=None) -> torch.Tensor:
    """
    Gather specific latent vectors from a sequence based on indices.
    
    Args:
        latent (torch.Tensor): [batch_size, seq_length, dim] - input sequence of latent vectors.
        ids (torch.Tensor): [batch_size, k] - indices of the positions to gather.
        rope (bool): Optional, not used here (placeholder for future use).
        condition_length: Optional, not used here (placeholder for future use).
    
    Returns:
        torch.Tensor: Gathered latent vectors [batch_size, k, dim].
    """
    B, K = ids.shape

    # Create batch indices for advanced indexing
    batch_indices = torch.arange(B, device=latent.device).unsqueeze(1).expand(-1, K)

    # Gather the latent vectors at the specified positions
    return latent[batch_indices, ids, :]  # [B, K, D]


def token_selector(tensor1, tensor2, k, similarity_type='cosine', height=-1, width=-1,
                   erosion_dilation=False, kernel_size=5, kernel_type='square',
                   patch_size=2, vae_scale_factor=8):
    """
    Select k similar positions along the seq_length dimension from two tensors.
    
    Args:
        tensor1 (torch.Tensor): [batch_size, seq_length, dim]
        tensor2 (torch.Tensor): [batch_size, seq_length, dim]
        k (int): Number of similar positions to select.
        similarity_type (str): Method to compute similarity ('cosine', 'dot', 'euclidean', 'mse', 'diff_std').
        height (int): Height of the 2D feature map (for erosion/dilation).
        width (int): Width of the 2D feature map (for erosion/dilation).
        erosion_dilation (bool): Whether to apply morphological erosion and dilation to remove scattered points.
        kernel_size (int): Kernel size for morphological operations.
        kernel_type (str): Type of kernel for morphological operations ('square' or 'cross').
        patch_size (int): Patch size used in reshaping.
        vae_scale_factor (int): Scaling factor for reshaping.

    Returns:
        indices (torch.Tensor): [batch_size, k] - indices of selected positions (edit region).
        unselected_indices (torch.Tensor): [batch_size, seq_length - k] - indices of unselected positions (non-edit region).
    """
    batch_size, seq_length, dim = tensor1.shape

    # Compute similarity matrix
    if similarity_type == 'cosine':
        # Cosine similarity
        tensor1_norm = F.normalize(tensor1, dim=-1)
        tensor2_norm = F.normalize(tensor2, dim=-1)
        similarity = torch.sum(tensor1_norm * tensor2_norm, dim=-1)  # [batch_size, seq_length]
    elif similarity_type == 'dot':
        # Dot product similarity
        similarity = torch.sum(tensor1 * tensor2, dim=-1)  # [batch_size, seq_length]
    elif similarity_type == 'euclidean':
        # Euclidean distance converted to similarity
        distance = torch.norm(tensor1 - tensor2, dim=-1)
        similarity = -distance  # smaller distance = higher similarity
        similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min())
    elif similarity_type == 'mse':
        # Mean squared error converted to similarity
        diff = tensor1 - tensor2
        similarity = -torch.mean(diff ** 2, dim=-1)
    elif similarity_type == 'diff_std':
        # Standard deviation of differences
        diff = tensor1 - tensor2
        similarity = torch.std(diff, dim=-1)
    else:
        raise ValueError("similarity_type must be 'cosine', 'dot', 'euclidean', 'mse', or 'diff_std'")

    # Threshold selection
    selected_mask = similarity <= k  # [batch_size, seq_length]

    if erosion_dilation:
        # Reshape to 2D mask for morphological processing
        selected_mask = selected_mask.float().squeeze().reshape(
            height // (patch_size * vae_scale_factor),
            width // (patch_size * vae_scale_factor)
        )
        # Remove isolated points
        selected_mask = remove_scattered_points(selected_mask, kernel_size, 'square')
        selected_mask = selected_mask.bool().flatten().unsqueeze(0)

    # Get indices of selected positions
    unselected_indices = torch.arange(seq_length, device=tensor1.device).unsqueeze(0).expand(batch_size, -1)
    indices = unselected_indices[selected_mask].unsqueeze(0)  # selected indices
    n_selected = indices.shape[1]

    # Get indices of unselected positions
    unselected_mask = ~selected_mask
    unselected_indices = unselected_indices[unselected_mask].view(batch_size, seq_length - n_selected)

    return indices, unselected_indices  # [edit region, unedit region]


class Manager:

    def __init__(self) -> None:
        # model config
        self.patch_size = 2
        self.vae_scale_factor = 8
        self.inference_step = 28
        self.txt_length = None
        self.height = None
        self.width = None
        self.latent_length = 0
        self.condition_latent = None
        self.condition_length = 0
        self.latent_ids = None

        # regione config
        self.warmup_step = 8
        self.post_step = 0
        self.erosion_dilation = False
        self.threshold = None
        self.cache_threshold = 0
        self.refresh_step = []

        # realtime data
        self.current_step = 0
        self.edited_ids = None
        self.unedited_ids = None
        self.unedited_latent = None
        self.prev_refresh_step = None
        self.next_refresh_step = None
        self.refresh_step_real_time = []
        self.next_estimate = None

    def set_parameters(self, args) -> None:
        assert args.warmup_step >= 1 and args.num_inference_steps == 28, "Changing the inference step requires fitting a new gamma"
        self.inference_step = args.num_inference_steps
        self.warmup_step = args.warmup_step
        self.post_step = args.post_step
        self.threshold = args.threshold
        self.cache_threshold = args.cache_threshold
        self.erosion_dilation = args.erosion_dilation
        self.refresh_step = sorted([int(item) for item in args.refresh_step.split(',')])
        assert min(self.refresh_step) > self.warmup_step + 1 and max(self.refresh_step) <= self.inference_step - self.post_step - 1
        has_adjacent = lambda nums: any(abs(nums[i] - nums[i+1]) == 1 for i in range(len(nums)-1))
        assert not has_adjacent(self.refresh_step), "Refresh steps must not be adjacent."
        self.refresh_step.append(self.inference_step - self.post_step + 1)

    def step(self, latent, latent_ids) -> torch.Tensor:
        self.current_step += 1

        if self.current_step == self.warmup_step:
            self.unedited_latent = ids_gather(latent, self.unedited_ids)
            latent = ids_gather(latent, self.edited_ids)
            latent_ids = ids_gather(latent_ids.unsqueeze(0), self.edited_ids, rope=True, condition_length=self.condition_length).squeeze(0)

        elif self.current_step == self.inference_step - self.post_step:
            final_latent = torch.zeros_like(self.condition_latent)
            final_latent = ids_scatter(latent, self.edited_ids, final_latent)
            final_latent = ids_scatter(self.unedited_latent, self.unedited_ids, final_latent)
            latent = final_latent
            latent_ids = self.latent_ids
            self.prev_refresh_step = None

        # gather
        elif self.prev_refresh_step != None and self.current_step == self.prev_refresh_step:
            final_latent = torch.zeros_like(self.condition_latent)
            final_latent = ids_scatter(latent, self.edited_ids, final_latent)
            final_latent = ids_scatter(self.unedited_latent, self.unedited_ids, final_latent)
            latent = final_latent
            latent_ids = self.latent_ids

        # scatter
        elif self.prev_refresh_step != None and self.current_step == self.prev_refresh_step + 1:
            self.unedited_latent = ids_gather(latent, self.unedited_ids)
            latent = ids_gather(latent, self.edited_ids)
            latent_ids = ids_gather(latent_ids.unsqueeze(0), self.edited_ids, rope=True, condition_length=self.condition_length).squeeze(0)
            self.prev_refresh_step = self.next_refresh_step

        return latent, latent_ids

    def refresh(
        self,
        latents,
        image_latents,
        latent_ids,
        text_ids,
        patch_size=2,
        vae_scale_factor=8,
        height=None,
        width=None
    ) -> None:
        self.width = width
        self.height = height
        self.patch_size = patch_size
        self.vae_scale_factor = vae_scale_factor
        self.latent_length = latents.size(1)
        self.txt_length = text_ids.size(0)
        self.condition_latent = image_latents
        self.condition_length = image_latents.size(1) if image_latents is not None else 0
        self.current_step = 0
        self.prev_refresh_step = None
        self.next_refresh_step = None
        self.edited_ids = None
        self.unedited_ids = None
        self.unedited_latent = None
        self.latent_ids = latent_ids
        self.refresh_step_real_time = list(self.refresh_step)
        
        self.next_estimate = None

MANAGER = Manager()