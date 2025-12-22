import torch
import inspect
import numpy as np
import torch.nn.functional as F
from dataclasses import dataclass
from diffusers import FluxKontextPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.flux import FluxPipelineOutput
from diffusers.utils import BaseOutput, is_torch_xla_available
from typing import Optional, Union, List, Dict, Any, Callable

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


class FluxKontextManager:

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
        assert args["warmup_step"] >= 1 and args["num_inference_steps"] == 28, "Changing the inference step requires fitting a new gamma"
        self.inference_step = args["num_inference_steps"]
        self.warmup_step = args["warmup_step"]
        self.post_step = args["post_step"]
        self.threshold = args["threshold"]
        self.cache_threshold = args["cache_threshold"]
        self.erosion_dilation = args["erosion_dilation"]
        self.refresh_step = sorted([int(item) for item in args["refresh_step"].split(',')])
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
        