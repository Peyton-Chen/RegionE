import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional, Union, List, Dict, Any, Callable, Tuple

from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_step1x_edit import Step1XEditAttention
from diffusers.models.modeling_outputs import Transformer2DModelOutput
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
    PipelineImageInput,
    Step1XEditPipelineOutput,
    Step1XEditThinker,
    TextEmbedderOutput,
    Step1XEditPipelineV1P2,
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
gamma = torch.tensor([0.7936, 0.9807, 1.0063, 1.0205, 0.9946, 1.0125, 1.0116, 1.0125, 1.0172,
        1.0171, 1.0183, 1.0170, 1.0170, 1.0236, 1.0263, 1.0264, 1.0277, 1.0321,
        1.0338, 1.0361, 1.0396, 1.0454, 1.0492, 1.0566, 1.0696, 1.0879, 1.1179], dtype=torch.float16)

def regione_init(model_path, device):

    pipeline = RegionEStep1XEditPipelineV1P2.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    pipeline.scheduler = RegionEFlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline.transformer.forward = RegionEStep1XEditTransformer2DModelforward.__get__(pipeline.transformer, pipeline.transformer.__class__)
    for block in pipeline.transformer.transformer_blocks:
        block.attn.set_processor(RegionEStep1XEditAttnProcessor(False))
    for block in pipeline.transformer.single_transformer_blocks:
        block.attn.set_processor(RegionEStep1XEditAttnProcessor(True))
    return pipeline.to(device)


class RegionEStep1XEditPipelineV1P2(Step1XEditPipelineV1P2):
    count =0

    @torch.no_grad()
    def __call__(
        self,
        image: Optional[PipelineImageInput] = None,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        true_cfg_scale: float = 6.0,
        height: Optional[int] = None,
        width: Optional[int] = None,
        size_level: int = 1024,
        num_inference_steps: int = 28,
        sigmas: Optional[List[float]] = None,
        guidance_scale: float = 6.0,
        num_images_per_prompt: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor | TextEmbedderOutput] = None,
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
        process_norm_power: float = 0.4,
        enable_thinking_mode: bool = False,
        enable_reflection_mode: bool = True,
        max_try_cnt: int = 3,
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

        try_cnt = 0
        success = False

        if enable_reflection_mode or enable_thinking_mode:
            thinker = Step1XEditThinker(self.text_encoder, self.processor)
            if enable_thinking_mode:
                reformat_prompt = thinker.think(image, prompt)
            else:
                reformat_prompt = prompt
            prompt = reformat_prompt
            out_images = []
            out_think_info = []
            best_think_info = []
        else:
            max_try_cnt = 1
            out_images = None

        original_ref_image = image
        original_prompt = prompt

        while not success and try_cnt < max_try_cnt:
            # 1. Preprocess image
            image, ref_image, img_info, width, height = self.encode_image(
                image,
                width,
                height,
                size_level,
                device,
                num_images_per_prompt
            )

            # 2. Check inputs. Raise error if not correct
            self.check_inputs(
                prompt,
                height,
                width,
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
            prompt_embeds = self.encode_prompt(
                ref_image=ref_image,
                prompt=prompt,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
            )

            if do_true_cfg:
                negative_prompt_embeds = self.encode_prompt(
                    ref_image=ref_image,
                    prompt=negative_prompt,
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
                prompt_embeds.embedding.dtype,
                device,
                generator,
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

            MANAGER.refresh(latents, image_latents, latent_ids, prompt_embeds.txt_ids, negative_prompt_embeds.txt_ids, 2, self.vae_scale_factor, height, width)
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

                        noise_pred = self.transformer(
                            hidden_states=latent_model_input,
                            timestep=timestep / 1000,
                            guidance=guidance,
                            encoder_hidden_states=prompt_embeds.embedding,
                            prompt_embeds_mask=prompt_embeds.mask,
                            txt_ids=prompt_embeds.txt_ids,
                            img_ids=latent_ids,
                            text_embeddings=prompt_embeds.text_embeds,
                            text_mask=prompt_embeds.text_masks,
                            joint_attention_kwargs={**self.joint_attention_kwargs, **{'tag': 'cond'}},
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
                                encoder_hidden_states=negative_prompt_embeds.embedding,
                                prompt_embeds_mask=negative_prompt_embeds.mask,
                                txt_ids=negative_prompt_embeds.txt_ids,
                                img_ids=latent_ids,
                                text_embeddings=negative_prompt_embeds.text_embeds,
                                text_mask=negative_prompt_embeds.text_masks,
                                joint_attention_kwargs={**self.joint_attention_kwargs, **{'tag': 'uncond'}},
                                return_dict=False,
                            )[0]
                            neg_noise_pred = neg_noise_pred[:, : latents.size(1)]
                        
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


                from PIL import Image
                def token_ids2hw_mask(token_ids, height, width):

                    mask = torch.zeros((1, 1, height, width)).to(token_ids.device)

                    mask[:, :, token_ids // width, token_ids % width] = 1

                    return mask


                def token_ids2rgb_mask(token_ids, height, width, vae_scale_factor=8, patch_size=2):

                    # 断言token_ids的维度为1，即batchsize为1
                    assert token_ids.size(0) == 1 # only support batchsize = 1
                    # 获取token_ids的第一个元素
                    token_ids = token_ids[0]

                    # 计算latent_height和latent_width
                    latent_height = int(height) // (vae_scale_factor * 2)
                    latent_width = int(width) // (vae_scale_factor * 2)

                    # 将token_ids转换为latent_height和latent_width的mask
                    mask = token_ids2hw_mask(token_ids, latent_height, latent_width)
                    # 将mask插值到vae_scale_factor * 2倍的大小
                    mask = F.interpolate(mask, scale_factor=vae_scale_factor * 2, mode='nearest')

                    # 将mask扩展为4通道
                    return mask.expand(-1, 4, -1, -1)
                
                edit_image = image[0].convert("RGBA")
                mask = token_ids2rgb_mask(MANAGER.edited_ids, height, width, self.vae_scale_factor, patch_size=2)  # [1 c h w]
                mask = self.image_processor.pt_to_numpy(mask).squeeze()   # [1 H W C]
                mask[..., :3] = (mask[..., :3] * 255).round()
                mask[..., -1] = (mask[..., -1] * 160).round()
                mask = mask.astype(np.uint8)
                mask = Image.fromarray(mask, mode="RGBA")
                edit_image = [Image.alpha_composite(edit_image, mask)]
                edit_image[0].save(f"/mnt/jfs-test/github/RegionE/result/Step1X-Edit-v1p2/mask/{self.count}.png")
                self.count += 1

            if enable_reflection_mode:
                thinking_info, best_info = thinker.reflect(original_ref_image, image[0], original_prompt)
                success, refine_prompt = thinker.format_text(thinking_info)
                out_images.append(image[0])
                out_think_info.append(thinking_info)
                best_think_info.append(best_info)
                if not success:
                    if refine_prompt is not None:  # type: ignore
                        prompt = refine_prompt
                        image = image[0]
                    else:
                        image = original_ref_image
                        prompt = reformat_prompt
                    try_cnt += 1
            else:
                out_images = image
                break
        
        # Offload all models
        self.maybe_free_model_hooks()

        final_images = [out_images[0]] if out_images else []

        if enable_reflection_mode or enable_thinking_mode:

            if best_think_info and len(best_think_info) > 0 and len(out_images) > 0:
                best_idx = 0
                max_score = -1
                best_has_success = False

                for i, info in enumerate(best_think_info):
                    s1_min = min(info['score1']['score'])
                    s2_min = min(info['score2']['score'])
                    current_score = s1_min * s2_min

                    current_think_str = out_think_info[i] if i < len(out_think_info) else ""
                    current_has_success = "<#Success>" in current_think_str

                    if current_score > max_score:
                        best_idx = i
                        max_score = current_score
                        best_has_success = current_has_success
                
                    elif current_score == max_score:
                        if current_has_success and not best_has_success:
                            best_idx = i
                            best_has_success = True
                        
                        elif current_has_success == best_has_success:
                            best_idx = i

                final_images = [out_images[best_idx]]

            if enable_thinking_mode:
                return Step1XEditPipelineOutput(
                    images=out_images, 
                    reformat_prompt=reformat_prompt, 
                    think_info=out_think_info, 
                    best_info=best_think_info,
                    final_images=final_images
                )
            else:
                return Step1XEditPipelineOutput(
                    images=out_images, 
                    think_info=out_think_info, 
                    best_info=best_think_info,
                    final_images=final_images
                )

        else:
            if not return_dict:
                return (image,)

        return Step1XEditPipelineOutput(images=out_images, final_images=final_images)


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
    text_embeddings: torch.Tensor | None = None,
    text_mask: torch.Tensor | None = None,
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
    
    if self.text_token_mapping is not None:
        text_hidden_states = self.text_token_mapping(text_embeddings)  # type: ignore
        text_hidden_states = text_hidden_states * text_mask[:, :, None]
        encoder_hidden_states = encoder_hidden_states + text_hidden_states

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

        self.k_cache_odd = None
        self.k_cache_even = None
        self.v_cache_odd = None
        self.v_cache_even = None
        self.single = single

    def __call__(
        self,
        attn: "Step1XEditAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        tag: str = None
    ) -> torch.Tensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        query = attn.to_q(hidden_states)

        # ---------------------------------------------------------------------
        if tag == 'cond':
            if MANAGER.current_step < MANAGER.warmup_step - 1 or MANAGER.current_step > MANAGER.inference_step - MANAGER.post_step - 1:
                key = attn.to_k(hidden_states)
                value = attn.to_v(hidden_states)

            elif MANAGER.current_step == MANAGER.warmup_step - 1 or MANAGER.current_step == MANAGER.prev_refresh_step:
                key = attn.to_k(hidden_states)
                value = attn.to_v(hidden_states)
                self.k_cache_even = key
                self.v_cache_even = value

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
                    self.k_cache_even.view(batch_size, self.k_cache_even.shape[1], -1)
                )
                _partially_linear(
                    hidden_states,
                    attn.to_v.weight,
                    attn.to_v.bias,
                    selection,
                    self.v_cache_even.view(batch_size, self.v_cache_even.shape[1], -1)
                )

                key = self.k_cache_even
                value = self.v_cache_even
        elif tag == 'uncond':
            if MANAGER.current_step < MANAGER.warmup_step - 1 or MANAGER.current_step > MANAGER.inference_step - MANAGER.post_step - 1:
                key = attn.to_k(hidden_states)
                value = attn.to_v(hidden_states)

            elif MANAGER.current_step == MANAGER.warmup_step - 1 or MANAGER.current_step == MANAGER.prev_refresh_step:
                key = attn.to_k(hidden_states)
                value = attn.to_v(hidden_states)
                self.k_cache_odd = key
                self.v_cache_odd = value

            elif MANAGER.current_step > MANAGER.warmup_step - 1 and MANAGER.current_step <= MANAGER.inference_step - MANAGER.post_step - 1:
                
                if self.single:
                    selection = torch.cat((torch.arange(MANAGER.neg_txt_length).to(MANAGER.edited_ids), MANAGER.edited_ids.squeeze(0)+MANAGER.neg_txt_length))
                else:
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

                key = self.k_cache_odd
                value = self.v_cache_odd
        else:
            NotImplementedError(f'Error tag: {tag}')
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
