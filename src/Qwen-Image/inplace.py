import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, Union, List, Dict, Any, Callable, Tuple

from diffusers import QwenImageEditPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen
from diffusers.models.attention_processor import Attention
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.pipelines.qwenimage import QwenImagePipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from utils import (
    calculate_dimensions,
    calculate_shift,
    retrieve_timesteps,
    MANAGER,
    ids_gather,
    ids_scatter,
    token_selector,
    FlowMatchEulerDiscreteSchedulerOutput
)
if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False
try:
    import flash_attn
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn = False
from fused_kernels import _partially_linear


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
gamma = torch.tensor([1.0195, 1.0233, 1.0243, 1.0185, 1.0321, 1.0208, 1.0260, 1.0233, 1.0258,
        1.0292, 1.0316, 1.0306, 1.0289, 1.0347, 1.0329, 1.0402, 1.0378, 1.0384,
        1.0413, 1.0444, 1.0526, 1.0400, 1.0555, 1.0439, 1.0357, 1.0118, 0.7603],
       dtype=torch.float16)


def regione_init(model_path, device):

    pipeline = RegionEQwenImageEditPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipeline.scheduler = RegionEFlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline.transformer.forward = RegionEStep1XEditTransformer2DModelforward.__get__(pipeline.transformer, pipeline.transformer.__class__)
    for block in pipeline.transformer.transformer_blocks:
        block.attn.set_processor(RegionEQwenDoubleStreamAttnProcessor2_0(False))
    return pipeline.to(device)


class RegionEQwenImageEditPipeline(QwenImageEditPipeline):

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 4.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        sigmas: Optional[List[float]] = None,
        guidance_scale: Optional[float] = None,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            true_cfg_scale (`float`, *optional*, defaults to 1.0):
                true_cfg_scale (`float`, *optional*, defaults to 1.0): Guidance scale as defined in [Classifier-Free
                Diffusion Guidance](https://huggingface.co/papers/2207.12598). `true_cfg_scale` is defined as `w` of
                equation 2. of [Imagen Paper](https://huggingface.co/papers/2205.11487). Classifier-free guidance is
                enabled by setting `true_cfg_scale > 1` and a provided `negative_prompt`. Higher guidance scale
                encourages to generate images that are closely linked to the text `prompt`, usually at the expense of
                lower image quality.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to None):
                A guidance scale value for guidance distilled models. Unlike the traditional classifier-free guidance
                where the guidance scale is applied during inference through noise prediction rescaling, guidance
                distilled models take the guidance scale directly as an input parameter during forward pass. Guidance
                scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images
                that are closely linked to the text `prompt`, usually at the expense of lower image quality. This
                parameter in the pipeline is there to support future guidance-distilled models when they come up. It is
                ignored when not using guidance distilled models. To enable traditional classifier-free guidance,
                please pass `true_cfg_scale > 1.0` and `negative_prompt` (even an empty negative prompt like " " should
                enable classifier-free guidance computations).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will be generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.qwenimage.QwenImagePipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int` defaults to 512): Maximum sequence length to use with the `prompt`.

        Examples:

        Returns:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] or `tuple`:
            [`~pipelines.qwenimage.QwenImagePipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is a list with the generated images.
        """
        image_size = image[0].size if isinstance(image, list) else image.size
        calculated_width, calculated_height, _ = calculate_dimensions(1024 * 1024, image_size[0] / image_size[1])
        height = height or calculated_height
        width = width or calculated_width

        multiple_of = self.vae_scale_factor * 2
        width = width // multiple_of * multiple_of
        height = height // multiple_of * multiple_of

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt=negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            negative_prompt_embeds_mask=negative_prompt_embeds_mask,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # 3. Preprocess image
        if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            image = self.image_processor.resize(image, calculated_height, calculated_width)
            prompt_image = image
            image = self.image_processor.preprocess(image, calculated_height, calculated_width)
            image = image.unsqueeze(2)

        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )

        if true_cfg_scale > 1 and not has_neg_prompt:
            logger.warning(
                f"true_cfg_scale is passed as {true_cfg_scale}, but classifier-free guidance is not enabled since no negative_prompt is provided."
            )
        elif true_cfg_scale <= 1 and has_neg_prompt:
            logger.warning(
                " negative_prompt is passed but classifier-free guidance is not enabled since true_cfg_scale <= 1"
            )

        do_true_cfg = true_cfg_scale > 1 and has_neg_prompt
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(
            image=prompt_image,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        if do_true_cfg:
            negative_prompt_embeds, negative_prompt_embeds_mask = self.encode_prompt(
                image=prompt_image,
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )

        # 4. Prepare latent variables
        num_channels_latents = self.transformer.config.in_channels // 4
        latents, image_latents = self.prepare_latents(
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

        img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                (1, calculated_height // self.vae_scale_factor // 2, calculated_width // self.vae_scale_factor // 2),
            ]
        ] * batch_size

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
        if self.transformer.config.guidance_embeds and guidance_scale is None:
            raise ValueError("guidance_scale is required for guidance-distilled model.")
        elif self.transformer.config.guidance_embeds:
            guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
            guidance = guidance.expand(latents.shape[0])
        elif not self.transformer.config.guidance_embeds and guidance_scale is not None:
            logger.warning(
                f"guidance_scale is passed as {guidance_scale}, but ignored since the model is not guidance-distilled."
            )
            guidance = None
        elif not self.transformer.config.guidance_embeds and guidance_scale is None:
            guidance = None

        if self.attention_kwargs is None:
            self._attention_kwargs = {}

        txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist() if prompt_embeds_mask is not None else None
        negative_txt_seq_lens = (
            negative_prompt_embeds_mask.sum(dim=1).tolist() if negative_prompt_embeds_mask is not None else None
        )

        latent_ids = torch.arange(latents.shape[1] + image_latents.shape[1], device=latents.device)
        MANAGER.refresh(latents, image_latents, latent_ids, 2, self.vae_scale_factor, height, width)
        cache, should_cache, accumulate, error = None, False, 1, 0
        # 6. Denoising loop
        self.scheduler.set_begin_index(0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                
                assert i == MANAGER.current_step

                if MANAGER.current_step <= MANAGER.warmup_step or MANAGER.current_step > MANAGER.inference_step - MANAGER.post_step - 1 or MANAGER.current_step == MANAGER.prev_refresh_step:
                    should_cache = False
                    accumulate = 1
                    error = 1 - accumulate
                else:
                    ratio = (gamma[i-1]) * (1+(t-timesteps[i-1])/1000)
                    if ratio >= 1:
                        should_cache = False
                        accumulate = 1
                        error = 1 - accumulate
                    else:
                        accumulate = accumulate * ratio
                        error = 1 - accumulate
                        if error > MANAGER.cache_threshold:
                            should_cache = False
                            accumulate = 1
                            error = 1 - accumulate
                        else:
                            should_cache = True

                if should_cache:

                    if cache.shape[1] != latents.shape[1]:
                        cache = ids_gather(cache, MANAGER.edited_ids)
                    noise_pred = cache * ratio

                else:
                    if self.interrupt:
                        continue

                    self._current_timestep = t

                    latent_model_input = latents
                    if image_latents is not None and (MANAGER.current_step <= MANAGER.warmup_step -1 or MANAGER.current_step > MANAGER.inference_step - MANAGER.post_step - 1 or MANAGER.current_step == MANAGER.prev_refresh_step):
                        latent_model_input = torch.cat([latents, image_latents], dim=1)

                    # broadcast to batch dimension in a way that's compatible with ONNX/Core ML 
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)

                    with self.transformer.cache_context("cond"):
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states_mask=prompt_embeds_mask,
                            encoder_hidden_states=prompt_embeds,
                            img_shapes=img_shapes,
                            txt_seq_lens=txt_seq_lens,
                            latent_ids=latent_ids,
                            attention_kwargs={**self.attention_kwargs, **{'tag': 'cond'}},
                            return_dict=False,
                        )[0]
                        noise_pred = noise_pred[:, : latents.size(1)]

                    if do_true_cfg:
                        with self.transformer.cache_context("uncond"):
                            neg_noise_pred = self.transformer(
                                hidden_states=latent_model_input,
                                timestep=timestep / 1000,
                                guidance=guidance,
                                encoder_hidden_states_mask=negative_prompt_embeds_mask,
                                encoder_hidden_states=negative_prompt_embeds,
                                img_shapes=img_shapes,
                                txt_seq_lens=negative_txt_seq_lens,
                                latent_ids=latent_ids,
                                attention_kwargs={**self.attention_kwargs, **{'tag': 'uncond'}},
                                return_dict=False,
                            )[0]
                        neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                        comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)

                        cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
                        noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
                        noise_pred = comb_pred * (cond_norm / noise_norm)
                    cache = noise_pred

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

                latents, latent_ids = MANAGER.step(latents, latent_ids)

        self._current_timestep = None
        if output_type == "latent":
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents.to(self.vae.dtype)
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
                latents.device, latents.dtype
            )
            latents = latents / latents_std + latents_mean
            image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
            image = self.image_processor.postprocess(image, output_type=output_type)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return QwenImagePipelineOutput(images=image)


def RegionEStep1XEditTransformer2DModelforward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    encoder_hidden_states_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_shapes: Optional[List[Tuple[int, int, int]]] = None,
    txt_seq_lens: Optional[List[int]] = None,
    guidance: torch.Tensor = None,  # TODO: this should probably be removed
    latent_ids: torch.Tensor = None,
    attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    return_dict: bool = True,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    """
    The [`QwenTransformer2DModel`] forward method.

    Args:
        hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
            Input `hidden_states`.
        encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        encoder_hidden_states_mask (`torch.Tensor` of shape `(batch_size, text_sequence_length)`):
            Mask of the input conditions.
        timestep ( `torch.LongTensor`):
            Used to indicate denoising step.
        attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.

    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """
    if attention_kwargs is not None:
        attention_kwargs = attention_kwargs.copy()
        lora_scale = attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    tag = attention_kwargs.get('tag')

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if attention_kwargs is not None and attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    hidden_states = self.img_in(hidden_states)

    timestep = timestep.to(hidden_states.dtype)
    encoder_hidden_states = self.txt_norm(encoder_hidden_states)
    encoder_hidden_states = self.txt_in(encoder_hidden_states)

    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = (
        self.time_text_embed(timestep, hidden_states)
        if guidance is None
        else self.time_text_embed(timestep, guidance, hidden_states)
    )

    image_rotary_emb = self.pos_embed(img_shapes, txt_seq_lens, device=hidden_states.device)
    MANAGER.image_rotary_emb = (image_rotary_emb[0][latent_ids, :], image_rotary_emb[1])

    for index_block, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                encoder_hidden_states_mask,
                temb,
                image_rotary_emb,
            )

        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                encoder_hidden_states_mask=encoder_hidden_states_mask,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=attention_kwargs,
            )

        # controlnet residual
        if controlnet_block_samples is not None:
            interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
            interval_control = int(np.ceil(interval_control))
            hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

    # Use only the image part (hidden_states) from the dual-stream blocks
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)


class RegionEFlowMatchEulerDiscreteScheduler(FlowMatchEulerDiscreteScheduler):

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        sample: torch.FloatTensor,
        s_churn: float = 0.0,
        s_tmin: float = 0.0,
        s_tmax: float = float("inf"),
        s_noise: float = 1.0,
        generator: Optional[torch.Generator] = None,
        per_token_timesteps: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[FlowMatchEulerDiscreteSchedulerOutput, Tuple]:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the diffusion
        process from the learned model outputs (most often the predicted noise).

        Args:
            model_output (`torch.FloatTensor`):
                The direct output from learned diffusion model.
            timestep (`float`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.FloatTensor`):
                A current instance of a sample created by the diffusion process.
            s_churn (`float`):
            s_tmin  (`float`):
            s_tmax  (`float`):
            s_noise (`float`, defaults to 1.0):
                Scaling factor for noise added to the sample.
            generator (`torch.Generator`, *optional*):
                A random number generator.
            per_token_timesteps (`torch.Tensor`, *optional*):
                The timesteps for each token in the sample.
            return_dict (`bool`):
                Whether or not to return a
                [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] or tuple.

        Returns:
            [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] or `tuple`:
                If return_dict is `True`,
                [`~schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteSchedulerOutput`] is returned,
                otherwise a tuple is returned where the first element is the sample tensor.
        """

        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `FlowMatchEulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        sample = sample.to(torch.float32)

        if per_token_timesteps is not None:
            per_token_sigmas = per_token_timesteps / self.config.num_train_timesteps

            sigmas = self.sigmas[:, None, None]
            lower_mask = sigmas < per_token_sigmas[None] - 1e-6
            lower_sigmas = lower_mask * sigmas
            lower_sigmas, _ = lower_sigmas.max(dim=0)

            current_sigma = per_token_sigmas[..., None]
            next_sigma = lower_sigmas[..., None]
            dt = current_sigma - next_sigma
        else:
            sigma_idx = self.step_index
            sigma = self.sigmas[sigma_idx]
            sigma_next = self.sigmas[sigma_idx + 1]

            current_sigma = sigma
            next_sigma = sigma_next

            if MANAGER.current_step == MANAGER.warmup_step - 1:
                MANAGER.prev_refresh_step = MANAGER.refresh_step_real_time.pop(0) - 1
                sigma_refresh = self.sigmas[MANAGER.prev_refresh_step]
                dt_final = self.sigmas[-1] - sigma
                dt_direct = sigma_refresh - sigma

            elif MANAGER.prev_refresh_step != None and MANAGER.current_step == MANAGER.prev_refresh_step and len(MANAGER.refresh_step_real_time) != 0:
                MANAGER.next_refresh_step = MANAGER.refresh_step_real_time.pop(0) - 1
                sigma_refresh = self.sigmas[MANAGER.next_refresh_step]
                dt_direct = sigma_refresh - sigma

            dt = sigma_next - sigma

        if self.config.stochastic_sampling:
            x0 = sample - current_sigma * model_output
            noise = torch.randn_like(sample)
            prev_sample = (1.0 - next_sigma) * x0 + next_sigma * noise
        else:
            if MANAGER.current_step == MANAGER.warmup_step - 1:
                
                onestep_estimated_latent = sample + dt_final * model_output
                MANAGER.edited_ids, MANAGER.unedited_ids = token_selector(onestep_estimated_latent, MANAGER.condition_latent, MANAGER.threshold, similarity_type='cosine', height=MANAGER.height, width=MANAGER.width, erosion_dilation=MANAGER.erosion_dilation, patch_size=MANAGER.patch_size, vae_scale_factor=MANAGER.vae_scale_factor)

                selected_sample = ids_gather(sample, MANAGER.edited_ids)
                selected_model_output = ids_gather(model_output, MANAGER.edited_ids)
                selected_prev_sample = selected_sample + dt * selected_model_output

                unselected_sample = ids_gather(sample, MANAGER.unedited_ids)
                unselected_model_output = ids_gather(model_output, MANAGER.unedited_ids)
                unselected_prev_sample = unselected_sample + dt_direct * unselected_model_output

                prev_sample = torch.zeros_like(sample)
                prev_sample = ids_scatter(selected_prev_sample, MANAGER.edited_ids, prev_sample)
                prev_sample = ids_scatter(unselected_prev_sample, MANAGER.unedited_ids, prev_sample)

            elif MANAGER.prev_refresh_step != None and MANAGER.current_step == MANAGER.prev_refresh_step:

                selected_sample = ids_gather(sample, MANAGER.edited_ids)
                selected_model_output = ids_gather(model_output, MANAGER.edited_ids)
                selected_prev_sample = selected_sample + dt * selected_model_output

                unselected_sample = ids_gather(sample, MANAGER.unedited_ids)
                unselected_model_output = ids_gather(model_output, MANAGER.unedited_ids)
                unselected_prev_sample = unselected_sample + dt_direct * unselected_model_output

                prev_sample = torch.zeros_like(sample)
                prev_sample = ids_scatter(selected_prev_sample, MANAGER.edited_ids, prev_sample)
                prev_sample = ids_scatter(unselected_prev_sample, MANAGER.unedited_ids, prev_sample)

            else:
                prev_sample = sample + dt * model_output

        # upon completion increase step index by one
        self._step_index += 1
        if per_token_timesteps is None:
            # Cast sample back to model compatible dtype
            prev_sample = prev_sample.to(model_output.dtype)

        if not return_dict:
            return (prev_sample,)

        return FlowMatchEulerDiscreteSchedulerOutput(prev_sample=prev_sample)


class RegionEQwenDoubleStreamAttnProcessor2_0:
    _attention_backend = None

    def __init__(self, single):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "QwenDoubleStreamAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0."
            )
        
        self.k_cache_odd = None
        self.k_cache_even = None
        self.v_cache_odd = None
        self.v_cache_even = None
        self.single = single

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,  # Image stream
        encoder_hidden_states: torch.FloatTensor = None,  # Text stream
        encoder_hidden_states_mask: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        tag: str = None
    ) -> torch.FloatTensor:
        if encoder_hidden_states is None:
            raise ValueError("QwenDoubleStreamAttnProcessor2_0 requires encoder_hidden_states (text stream)")
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        seq_txt = encoder_hidden_states.shape[1]

        # Compute QKV for image stream (sample projections)
        img_query = attn.to_q(hidden_states)

        # ---------------------------------------------------------------------
        if tag == 'cond':
            if MANAGER.current_step < MANAGER.warmup_step - 1 or MANAGER.current_step > MANAGER.inference_step - MANAGER.post_step - 1:

                img_key = attn.to_k(hidden_states)
                img_value = attn.to_v(hidden_states)
            elif MANAGER.current_step == MANAGER.warmup_step - 1 or MANAGER.current_step == MANAGER.prev_refresh_step:
                img_key = attn.to_k(hidden_states)
                img_value = attn.to_v(hidden_states)
                self.k_cache_even = img_key
                self.v_cache_even = img_value
            elif MANAGER.current_step > MANAGER.warmup_step - 1 and MANAGER.current_step <= MANAGER.inference_step - MANAGER.post_step - 1:
                selection = MANAGER.edited_ids.squeeze(0)

                _partially_linear(
                    hidden_states,
                    attn.to_k.weight,
                    attn.to_k.bias,
                    selection,
                    self.k_cache_even.view(batch_size, self.k_cache_even.shape[1], -1)
                )
                _partially_linear(
                    hidden_states,
                    attn.to_v.weight,
                    attn.to_v.bias,
                    selection,
                    self.v_cache_even.view(batch_size, self.v_cache_even.shape[1], -1)
                )

                img_key = self.k_cache_even
                img_value = self.v_cache_even
        elif tag == 'uncond':
            if MANAGER.current_step < MANAGER.warmup_step - 1 or MANAGER.current_step > MANAGER.inference_step - MANAGER.post_step - 1:
                img_key = attn.to_k(hidden_states)
                img_value = attn.to_v(hidden_states)
            elif MANAGER.current_step == MANAGER.warmup_step - 1 or MANAGER.current_step == MANAGER.prev_refresh_step:
                img_key = attn.to_k(hidden_states)
                img_value = attn.to_v(hidden_states)
                self.k_cache_odd = img_key
                self.v_cache_odd = img_value
            elif MANAGER.current_step > MANAGER.warmup_step - 1 and MANAGER.current_step <= MANAGER.inference_step - MANAGER.post_step - 1:
                
                selection = MANAGER.edited_ids.squeeze(0)

                _partially_linear(
                    hidden_states,
                    attn.to_k.weight,
                    attn.to_k.bias,
                    selection,
                    self.k_cache_odd.view(batch_size, self.k_cache_odd.shape[1], -1)
                )
                _partially_linear(
                    hidden_states,
                    attn.to_v.weight,
                    attn.to_v.bias,
                    selection,
                    self.v_cache_odd.view(batch_size, self.v_cache_odd.shape[1], -1)
                )

                img_key = self.k_cache_odd
                img_value = self.v_cache_odd
        else:
            NotImplementedError(f'Error tag: {tag}')
        
        # Compute QKV for text stream (context projections)
        txt_query = attn.add_q_proj(encoder_hidden_states)
        txt_key = attn.add_k_proj(encoder_hidden_states)
        txt_value = attn.add_v_proj(encoder_hidden_states)

        # Reshape for multi-head attention
        img_query = img_query.unflatten(-1, (attn.heads, -1))
        img_key = img_key.unflatten(-1, (attn.heads, -1))
        img_value = img_value.unflatten(-1, (attn.heads, -1))

        head_dim = img_query.shape[-1]
        
        # Apply QK normalization
        if attn.norm_q is not None:
            img_query = attn.norm_q(img_query)
        if attn.norm_k is not None:
            img_key = attn.norm_k(img_key)

        txt_query = txt_query.unflatten(-1, (attn.heads, -1))
        txt_key = txt_key.unflatten(-1, (attn.heads, -1))
        txt_value = txt_value.unflatten(-1, (attn.heads, -1))

        if attn.norm_added_q is not None:
            txt_query = attn.norm_added_q(txt_query)
        if attn.norm_added_k is not None:
            txt_key = attn.norm_added_k(txt_key)

        # Apply RoPE
        if image_rotary_emb is not None:
            img_freqs, txt_freqs = image_rotary_emb

            if img_query.shape[1] != 0:
                img_query = apply_rotary_emb_qwen(img_query, MANAGER.image_rotary_emb[0], use_real=False)
            img_key = apply_rotary_emb_qwen(img_key, img_freqs, use_real=False)

            txt_query = apply_rotary_emb_qwen(txt_query, txt_freqs, use_real=False)
            txt_key = apply_rotary_emb_qwen(txt_key, txt_freqs, use_real=False)

        # Concatenate for joint attention
        # Order: [text, image]
        joint_query = torch.cat([txt_query, img_query], dim=1)
        joint_key = torch.cat([txt_key, img_key], dim=1)
        joint_value = torch.cat([txt_value, img_value], dim=1)

        # Compute joint attention
        
        if flash_attn is not None:
            joint_hidden_states = flash_attn_func(
                    joint_query, joint_key, joint_value, dropout_p=0.0, causal=False
            )
            joint_hidden_states = joint_hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        else:
            joint_query, joint_key, joint_value = joint_query.transpose(1, 2), joint_key.transpose(1, 2), joint_value.transpose(1, 2)
            joint_hidden_states = F.scaled_dot_product_attention(
                joint_query, joint_key, joint_value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            joint_hidden_states = joint_hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)

        joint_hidden_states = joint_hidden_states.to(joint_query.dtype)

        # Split attention outputs back
        txt_attn_output = joint_hidden_states[:, :seq_txt, :]  # Text part
        img_attn_output = joint_hidden_states[:, seq_txt:, :]  # Image part

        # Apply output projections
        img_attn_output = attn.to_out[0](img_attn_output)
        if len(attn.to_out) > 1:
            img_attn_output = attn.to_out[1](img_attn_output)  # dropout

        txt_attn_output = attn.to_add_out(txt_attn_output)

        return img_attn_output, txt_attn_output
