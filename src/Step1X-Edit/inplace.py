import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, Union, List, Dict, Any, Callable, Tuple

from diffusers import Step1XEditPipeline
from diffusers.image_processor import PipelineImageInput
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_step1x_edit import Step1XEditAttention
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.pipelines.step1x_edit import Step1XEditPipelineOutput
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import (
    is_torch_xla_available,
    logging,
    USE_PEFT_BACKEND,
    scale_lora_layers,
    unscale_lora_layers,
)
from utils import (
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
gamma = torch.tensor([0.9746, 0.9593, 1.0036, 1.0084, 1.0106, 1.0114, 1.0138, 1.0163, 1.0152,
        1.0163, 1.0197, 1.0186, 1.0219, 1.0218, 1.0223, 1.0266, 1.0272, 1.0305,
        1.0311, 1.0362, 1.0385, 1.0423, 1.0500, 1.0536, 1.0671, 1.0866, 1.1015], dtype=torch.float16)


def regione_init(model_path, device):

    pipeline = RegionEStep1XEditPipeline.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipeline.scheduler = RegionEFlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline.transformer.forward = RegionEStep1XEditTransformer2DModelforward.__get__(pipeline.transformer, pipeline.transformer.__class__)
    for block in pipeline.transformer.transformer_blocks:
        block.attn.set_processor(RegionEStep1XEditAttnProcessor(False))
    for block in pipeline.transformer.single_transformer_blocks:
        block.attn.set_processor(RegionEStep1XEditAttnProcessor(True))
    return pipeline.to(device)


class RegionEStep1XEditPipeline(Step1XEditPipeline):

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 6.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 6.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        prompt_embeds_mask: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
        ip_adapter_image: Optional[PipelineImageInput] = None,
        ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        negative_ip_adapter_image: Optional[PipelineImageInput] = None,
        negative_ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        timesteps_truncate: float = 0.93,
        process_norm_power: float = 0.4
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            image (`torch.Tensor`, `PIL.Image.Image`, `np.ndarray`, `List[torch.Tensor]`, `List[PIL.Image.Image]`, or `List[np.ndarray]`):
                `Image`, numpy array or tensor representing an image batch to be used as the starting point. For both
                numpy array and pytorch tensor, the expected value range is between `[0, 1]` If it's a tensor or a list
                or tensors, the expected shape should be `(B, C, H, W)` or `(C, H, W)`. If it is a numpy array or a
                list of arrays, the expected shape should be `(B, H, W, C)` or `(H, W, C)` It can also accept image
                latents as `image`, but if passing latents directly it is not encoded again.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `true_cfg_scale` is
                not greater than `1`).
            true_cfg_scale (`float`, *optional*, defaults to 6.0):
                When > 1.0 and a provided `negative_prompt`, enables true classifier-free guidance.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image. This is set to 1024 by default for the best results.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image. This is set to 1024 by default for the best results.
            num_inference_steps (`int`, *optional*, defaults to 28):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            sigmas (`List[float]`, *optional*):
                Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
                their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
                will be used.
            guidance_scale (`float`, *optional*, defaults to 3.5):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
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
                Whether or not to return a [`~pipelines.step1x_edit.Step1XEditPipelineOutput`] instead of a plain tuple.
            joint_attention_kwargs (`dict`, *optional*):
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

        Examples:

        Returns:
            [`~pipelines.step1x_edit.Step1XEditPipelineOutput`] or `tuple`:
            [`~pipelines.step1x_edit.Step1XEditPipelineOutput`] if `return_dict` is True, otherwise a `tuple`. When
            returning a tuple, the first element is a list with the generated images.
        """
        
        device = self._execution_device

        # 1. Preprocess image
        image, ref_image, img_info, width, height = self.encode_image(
            image, 
            width,
            height,
            device, 
            num_images_per_prompt
        )
        
        # 2. Check inputs. Raise error if not correct
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

        lora_scale = (
            self.joint_attention_kwargs.get("scale", None) if self.joint_attention_kwargs is not None else None
        )
        has_neg_prompt = negative_prompt is not None or (
            negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
        )
        if not has_neg_prompt:
            negative_prompt = "" if image is not None else "worst quality, wrong limbs, unreasonable limbs, normal quality, low quality, low res, blurry, text, watermark, logo, banner, extra digits, cropped, jpeg artifacts, signature, username, error, sketch ,duplicate, ugly, monochrome, horror, geometry, mutation, disgusting"
        do_true_cfg = true_cfg_scale > 1
        (
            prompt_embeds, 
            prompt_embeds_mask, 
            text_ids
        ) = self.encode_prompt(
            ref_image=ref_image,
            prompt=prompt,
            prompt_embeds=prompt_embeds,
            prompt_embeds_mask=prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
        )
        if do_true_cfg:
            (
                negative_prompt_embeds, 
                negative_prompt_embeds_mask, 
                negative_text_ids,
            ) = self.encode_prompt(
                ref_image=ref_image,
                prompt=negative_prompt,
                prompt_embeds=negative_prompt_embeds,
                prompt_embeds_mask=negative_prompt_embeds_mask,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
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

        MANAGER.refresh(latents, image_latents, latent_ids, text_ids, 2, self.vae_scale_factor, height, width)
        cache, should_cache, accumulate, error = None, False, 1, 0
        # 6. Denoising loop
        # We set the index here to remove DtoH sync, helpful especially during compilation.
        # Check out more details here: https://github.com/huggingface/diffusers/pull/11696
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
                    if image_embeds is not None:
                        self._joint_attention_kwargs["ip_adapter_image_embeds"] = image_embeds

                    latent_model_input = latents

                    if image_latents is not None and (MANAGER.current_step <= MANAGER.warmup_step - 1 or MANAGER.current_step > MANAGER.inference_step - MANAGER.post_step - 1 or MANAGER.current_step == MANAGER.prev_refresh_step):
                        latent_model_input = torch.cat([latents, image_latents], dim=1)
                    timestep = t.expand(latents.shape[0]).to(latents.dtype)
                    
                    if do_true_cfg:
                        latent_model_input = torch.cat((latent_model_input, latent_model_input), dim=0)
                        timestep = torch.cat((timestep, timestep), dim=0)
                        prompt_embeds_input = torch.cat((prompt_embeds, negative_prompt_embeds), dim=0)
                        prompt_embeds_mask_input = torch.cat((prompt_embeds_mask, negative_prompt_embeds_mask), dim=0)
                        
                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states=prompt_embeds_input,
                            prompt_embeds_mask=prompt_embeds_mask_input,
                            txt_ids=text_ids,
                            img_ids=latent_ids,
                            joint_attention_kwargs=self.joint_attention_kwargs,
                            return_dict=False,
                        )[0]
                        noise_pred = noise_pred[:, : latents.size(1)]
                        noise_pred, neg_noise_pred = noise_pred.chunk(2)
                        
                        if t.item() > timesteps_truncate:
                            diff = noise_pred - neg_noise_pred
                            diff_norm = torch.norm(diff, dim=(2), keepdim=True)
                            
                            noise_pred = neg_noise_pred + true_cfg_scale * (
                                noise_pred - neg_noise_pred
                            ) / self.process_diff_norm(diff_norm, k=process_norm_power)
                            
                        else:
                            noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
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
            latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
            # image = self._output_process_image(image, img_info)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return Step1XEditPipelineOutput(images=image)


def RegionEStep1XEditTransformer2DModelforward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    prompt_embeds_mask: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
) -> Union[torch.Tensor, Transformer2DModelOutput]:
    """
    The [`Step1XEditTransformer2DModel`] forward method.

    Args:
        hidden_states (`torch.Tensor` of shape `(batch_size, image_sequence_length, in_channels)`):
            Input `hidden_states`.
        encoder_hidden_states (`torch.Tensor` of shape `(batch_size, text_sequence_length, joint_attention_dim)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        pooled_projections (`torch.Tensor` of shape `(batch_size, projection_dim)`): Embeddings projected
            from the embeddings of input conditions.
        timestep ( `torch.LongTensor`):
            Used to indicate denoising step.
        block_controlnet_hidden_states: (`list` of `torch.Tensor`):
            A list of tensors that if specified are added to the residuals of transformer blocks.
        joint_attention_kwargs (`dict`, *optional*):
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
    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

    encoder_hidden_states, y = self.connector(
        encoder_hidden_states, timestep, prompt_embeds_mask
    )
    hidden_states = self.x_embedder(hidden_states)

    temb = self.time_embed(self.time_proj(timestep * 1000).to(timestep))
    temb = temb + self.vec_embed(y)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)

    # ------------------------------------------------------
    MANAGER.image_rotary_emb = self.pos_embed(torch.cat((txt_ids, MANAGER.latent_ids), dim=0))
    # ------------------------------------------------------

    for index_block, block in enumerate(self.transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                joint_attention_kwargs,
            )

        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

    for index_block, block in enumerate(self.single_transformer_blocks):
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                block,
                hidden_states,
                encoder_hidden_states,
                temb,
                image_rotary_emb,
                joint_attention_kwargs,
            )

        else:
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

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


class RegionEStep1XEditAttnProcessor:

    def __init__(self, single):
        
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")

        self.k_cache = None
        self.v_cache = None
        self.single = single

    def __call__(
        self,
        attn: "Step1XEditAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        query = attn.to_q(hidden_states)

        # ---------------------------------------------------------------------
        if MANAGER.current_step < MANAGER.warmup_step - 1 or MANAGER.current_step > MANAGER.inference_step - MANAGER.post_step - 1:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)

        elif MANAGER.current_step == MANAGER.warmup_step - 1 or MANAGER.current_step == MANAGER.prev_refresh_step:
            key = attn.to_k(hidden_states)
            value = attn.to_v(hidden_states)
            self.k_cache = key
            self.v_cache = value

        elif MANAGER.current_step > MANAGER.warmup_step - 1 and MANAGER.current_step <= MANAGER.inference_step - MANAGER.post_step - 1:
            
            if self.single:
                selection = torch.cat((torch.arange(MANAGER.txt_length).to(MANAGER.edited_ids), MANAGER.edited_ids.squeeze(0)+MANAGER.txt_length))
            else:
                selection = MANAGER.edited_ids.squeeze(0)

            _partially_linear(
                hidden_states,
                attn.to_k.weight,
                attn.to_k.bias,
                selection,
                self.k_cache.view(batch_size, self.k_cache.shape[1], -1)
            )
            _partially_linear(
                hidden_states,
                attn.to_v.weight,
                attn.to_v.bias,
                selection,
                self.v_cache.view(batch_size, self.v_cache.shape[1], -1)
            )

            key = self.k_cache
            value = self.v_cache
        # ---------------------------------------------------------------------

        query = query.unflatten(-1, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(-1, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(-1, (attn.heads, -1)).transpose(1, 2)

        head_dim = query.shape[-1]
        
        query = attn.norm_q(query)
        key = attn.norm_k(key)

        encoder_query = encoder_key = encoder_value = None
        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            encoder_query = attn.add_q_proj(encoder_hidden_states)
            encoder_key = attn.add_k_proj(encoder_hidden_states)
            encoder_value = attn.add_v_proj(encoder_hidden_states)
        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1)).transpose(1, 2)
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1)).transpose(1, 2)
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1)).transpose(1, 2)
            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=2)
            key = torch.cat([encoder_key, key], dim=2)
            value = torch.cat([encoder_value, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, MANAGER.image_rotary_emb)

        if flash_attn is not None:
            query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
            hidden_states = flash_attn_func(
                    query, key, value, dropout_p=0.0, causal=False
            )
            hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        else:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
