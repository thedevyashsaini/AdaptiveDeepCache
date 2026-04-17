import time

import torch


class DeepCacheSDHelper(object):
    def __init__(self, pipe=None):
        self.pipe = pipe
        self.params = {}
        self.reset_states()

    def enable(self, pipe=None):
        if pipe is not None:
            self.pipe = pipe
        assert self.pipe is not None
        self.reset_states()
        self.wrap_modules()

    def disable(self):
        self.unwrap_modules()
        self.reset_states()

    def set_params(
        self,
        cache_interval=1,
        cache_branch_id=0,
        skip_mode="uniform",
        adaptive=False,
        threshold_early=0.040,
        threshold_mid=0.030,
        threshold_late=0.020,
        early_ratio=0.30,
        mid_ratio=0.70,
        force_refresh_every=0,
    ):
        cache_layer_id = cache_branch_id % 3
        cache_block_id = cache_branch_id // 3
        self.params = {
            "cache_interval": cache_interval,
            "cache_layer_id": cache_layer_id,
            "cache_block_id": cache_block_id,
            "skip_mode": skip_mode,
            "adaptive": adaptive,
            "threshold_early": float(threshold_early),
            "threshold_mid": float(threshold_mid),
            "threshold_late": float(threshold_late),
            "early_ratio": float(early_ratio),
            "mid_ratio": float(mid_ratio),
            "force_refresh_every": int(force_refresh_every),
        }

    def get_step_logs(self):
        return list(self.step_logs)

    def _get_threshold_for_step(self):
        total = max(self.total_inference_steps, 1)
        progress = (self.cur_timestep - self.start_timestep) / max(total - 1, 1)
        if progress < self.params["early_ratio"]:
            return self.params["threshold_early"]
        if progress < self.params["mid_ratio"]:
            return self.params["threshold_mid"]
        return self.params["threshold_late"]

    def _compute_latent_delta(self, latent_model_input):
        latent = latent_model_input.detach()
        if latent.shape[0] > 1:
            latent = latent[:1]
        latent = latent.float()

        if self.prev_latent is None:
            self.prev_latent = latent
            return None

        delta = torch.sqrt(torch.mean((latent - self.prev_latent) ** 2)).item()
        self.prev_latent = latent
        return float(delta)

    def _is_refresh_step(self, delta):
        self.start_timestep = (
            self.cur_timestep if self.start_timestep is None else self.start_timestep
        )

        if not self.params.get("adaptive", False):
            cache_interval = self.params["cache_interval"]
            return (self.cur_timestep - self.start_timestep) % cache_interval == 0

        if self.last_refresh_timestep is None:
            return True

        force_refresh_every = self.params.get("force_refresh_every", 0)
        if (
            force_refresh_every > 0
            and (self.cur_timestep - self.last_refresh_timestep) >= force_refresh_every
        ):
            return True

        if delta is None:
            return True

        threshold = self._get_threshold_for_step()
        return delta > threshold

    def _register_step_decision(self, latent_model_input):
        delta = self._compute_latent_delta(latent_model_input)
        refresh = self._is_refresh_step(delta)

        if refresh:
            self.last_refresh_timestep = self.cur_timestep

        self.step_refresh_map[self.cur_timestep] = refresh

        threshold = (
            self._get_threshold_for_step()
            if self.params.get("adaptive", False)
            else None
        )
        self.step_logs.append(
            {
                "timestep_index": int(self.cur_timestep),
                "delta_latent": None if delta is None else float(delta),
                "threshold": threshold,
                "refresh": bool(refresh),
                "reuse": bool(not refresh),
                "step_unet_time_s": None,
            }
        )

    def is_skip_step(self, block_i, layer_i, blocktype="down"):
        cache_layer_id = self.params["cache_layer_id"]
        cache_block_id = self.params["cache_block_id"]
        refresh = self.step_refresh_map.get(self.cur_timestep, True)

        if refresh:
            return False

        if block_i > cache_block_id or blocktype == "mid":
            return True
        if block_i < cache_block_id:
            return False
        return (
            layer_i >= cache_layer_id
            if blocktype == "down"
            else layer_i > cache_layer_id
        )

    def is_enter_position(self, block_i, layer_i):
        return (
            block_i == self.params["cache_block_id"]
            and layer_i == self.params["cache_layer_id"]
        )

    def wrap_unet_forward(self):
        self.function_dict["unet_forward"] = self.pipe.unet.forward

        def wrapped_forward(*args, **kwargs):
            timestep_value = int(args[1].item())
            if self.timestep_index_map is None:
                self.timestep_index_map = {
                    int(t.item()): i
                    for i, t in enumerate(self.pipe.scheduler.timesteps)
                }
                self.total_inference_steps = len(self.timestep_index_map)

            self.cur_timestep = self.timestep_index_map[timestep_value]
            self._register_step_decision(args[0])

            start = time.perf_counter()
            result = self.function_dict["unet_forward"](*args, **kwargs)
            elapsed = time.perf_counter() - start
            self.step_logs[-1]["step_unet_time_s"] = float(elapsed)
            return result

        self.pipe.unet.forward = wrapped_forward

    def wrap_block_forward(self, block, block_name, block_i, layer_i, blocktype="down"):
        self.function_dict[(blocktype, block_name, block_i, layer_i)] = block.forward

        def wrapped_forward(*args, **kwargs):
            skip = self.is_skip_step(block_i, layer_i, blocktype)
            key = (blocktype, block_name, block_i, layer_i)
            if skip and key in self.cached_output:
                return self.cached_output[key]

            result = self.function_dict[key](*args, **kwargs)
            self.cached_output[key] = result
            return result

        block.forward = wrapped_forward

    def wrap_modules(self):
        self.wrap_unet_forward()
        for block_i, block in enumerate(self.pipe.unet.down_blocks):
            for layer_i, attention in enumerate(getattr(block, "attentions", [])):
                self.wrap_block_forward(attention, "attentions", block_i, layer_i)
            for layer_i, resnet in enumerate(getattr(block, "resnets", [])):
                self.wrap_block_forward(resnet, "resnet", block_i, layer_i)
            for downsampler in (
                getattr(block, "downsamplers", []) if block.downsamplers else []
            ):
                self.wrap_block_forward(
                    downsampler,
                    "downsampler",
                    block_i,
                    len(getattr(block, "resnets", [])),
                )
            self.wrap_block_forward(block, "block", block_i, 0, blocktype="down")

        self.wrap_block_forward(
            self.pipe.unet.mid_block, "mid_block", 0, 0, blocktype="mid"
        )

        block_num = len(self.pipe.unet.up_blocks)
        for block_i, block in enumerate(self.pipe.unet.up_blocks):
            layer_num = len(getattr(block, "resnets", []))
            for layer_i, attention in enumerate(getattr(block, "attentions", [])):
                self.wrap_block_forward(
                    attention,
                    "attentions",
                    block_num - block_i - 1,
                    layer_num - layer_i - 1,
                    blocktype="up",
                )
            for layer_i, resnet in enumerate(getattr(block, "resnets", [])):
                self.wrap_block_forward(
                    resnet,
                    "resnet",
                    block_num - block_i - 1,
                    layer_num - layer_i - 1,
                    blocktype="up",
                )
            for upsampler in (
                getattr(block, "upsamplers", []) if block.upsamplers else []
            ):
                self.wrap_block_forward(
                    upsampler, "upsampler", block_num - block_i - 1, 0, blocktype="up"
                )
            self.wrap_block_forward(
                block, "block", block_num - block_i - 1, 0, blocktype="up"
            )

    def unwrap_modules(self):
        if "unet_forward" not in self.function_dict:
            return

        self.pipe.unet.forward = self.function_dict["unet_forward"]
        for block_i, block in enumerate(self.pipe.unet.down_blocks):
            for layer_i, attention in enumerate(getattr(block, "attentions", [])):
                attention.forward = self.function_dict[
                    ("down", "attentions", block_i, layer_i)
                ]
            for layer_i, resnet in enumerate(getattr(block, "resnets", [])):
                resnet.forward = self.function_dict[
                    ("down", "resnet", block_i, layer_i)
                ]
            for downsampler in (
                getattr(block, "downsamplers", []) if block.downsamplers else []
            ):
                downsampler.forward = self.function_dict[
                    ("down", "downsampler", block_i, len(getattr(block, "resnets", [])))
                ]
            block.forward = self.function_dict[("down", "block", block_i, 0)]

        self.pipe.unet.mid_block.forward = self.function_dict[
            ("mid", "mid_block", 0, 0)
        ]

        block_num = len(self.pipe.unet.up_blocks)
        for block_i, block in enumerate(self.pipe.unet.up_blocks):
            layer_num = len(getattr(block, "resnets", []))
            for layer_i, attention in enumerate(getattr(block, "attentions", [])):
                attention.forward = self.function_dict[
                    (
                        "up",
                        "attentions",
                        block_num - block_i - 1,
                        layer_num - layer_i - 1,
                    )
                ]
            for layer_i, resnet in enumerate(getattr(block, "resnets", [])):
                resnet.forward = self.function_dict[
                    ("up", "resnet", block_num - block_i - 1, layer_num - layer_i - 1)
                ]
            for upsampler in (
                getattr(block, "upsamplers", []) if block.upsamplers else []
            ):
                upsampler.forward = self.function_dict[
                    ("up", "upsampler", block_num - block_i - 1, 0)
                ]
            block.forward = self.function_dict[
                ("up", "block", block_num - block_i - 1, 0)
            ]

    def reset_states(self):
        self.cur_timestep = 0
        self.function_dict = {}
        self.cached_output = {}
        self.start_timestep = None
        self.last_refresh_timestep = None
        self.prev_latent = None
        self.step_refresh_map = {}
        self.step_logs = []
        self.timestep_index_map = None
        self.total_inference_steps = 0
