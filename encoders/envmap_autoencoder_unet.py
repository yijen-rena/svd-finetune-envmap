from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

# from ...configuration_utils import ConfigMixin, register_to_config
# from ...loaders import UNet2DConditionLoadersMixin
# from ...utils import BaseOutput, logging
# from ..attention_processor import CROSS_ATTENTION_PROCESSORS, AttentionProcessor, AttnProcessor
# from ..embeddings import TimestepEmbedding, Timesteps
# from ..modeling_utils import ModelMixin
# from .unet_3d_blocks import UNetMidBlockSpatioTemporal, get_down_block, get_up_block

import diffusers
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import UNet2DConditionLoadersMixin, PeftAdapterMixin
from diffusers.utils import BaseOutput, logging
from diffusers.models.attention_processor import CROSS_ATTENTION_PROCESSORS, AttentionProcessor, AttnProcessor
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.unets.unet_3d_blocks import UNetMidBlockSpatioTemporal, get_down_block, get_up_block

from utils import generate_plucker_rays

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

@dataclass
class UNetEnvMapConditionOutput(BaseOutput):
    """
    Output class for the UNetEnvMapConditionModel.
    
    Args:
        sample (torch.Tensor): Output tensor of shape (batch_size, out_channels, height, width)
        features (Dict[str, torch.Tensor]): Dictionary containing intermediate features from each level
    """
    sample: torch.Tensor
    features: Dict[str, torch.Tensor]

class ResnetBlock(nn.Module):
    """Simple ResNet block for feature extraction"""
    def __init__(self, in_channels, out_channels, eps=1e-6,groups=32, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(groups, in_channels, eps=eps)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(groups, out_channels, eps=eps)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        # Add residual connection if channels don't match
        self.residual = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv1(x)
        
        x = self.norm2(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = x + residual
        
        return x
    
    
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=1):
        super().__init__()
                
        resnets = []
        
        for i in range(num_layers):
            res_in_channels = in_channels if i == 0 else out_channels
            resnets.append(
                ResnetBlock(
                    in_channels=res_in_channels,
                    out_channels=out_channels,
                )
            )
        
        self.resnets = nn.ModuleList(resnets)
        
    def forward(self, hidden_states):
        output_states = ()
        for i, resnet in enumerate(self.resnets):
            hidden_states = resnet(hidden_states)
            output_states += (hidden_states,)
            
        return hidden_states, output_states
    
class UpBlock(nn.Module):
    def __init__(self, in_channels, prev_output_channel, out_channels, num_layers=1):
        super().__init__()
        
        resnets = []
        for i in range(num_layers):
            # in_channels = in_channels if i == 0 else out_channels
            res_in_channels = prev_output_channel if i == 0 else out_channels
            res_skip_channels = in_channels if (i == num_layers - 1) else out_channels
            resnets.append(
                ResnetBlock(
                    in_channels=res_in_channels + res_skip_channels,
                    out_channels=out_channels,
                )
            )
        
        self.resnets = nn.ModuleList(resnets)
        
    def forward(self, hidden_states, res_hidden_states_tuple):
        for i, resnet in enumerate(self.resnets):
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states)
            
        return hidden_states

class UNetEnvMapConditionModel(ModelMixin, ConfigMixin, UNet2DConditionLoadersMixin, PeftAdapterMixin):
    r"""
    A conditional Spatio-Temporal UNet model that takes a noisy video frames, conditional state, and a timestep and
    returns a sample shaped output.

    This model inherits from [`ModelMixin`]. Check the superclass documentation for it's generic methods implemented
    for all models (such as downloading or saving).

    Parameters:
        sample_size (`int` or `Tuple[int, int]`, *optional*, defaults to `None`):
            Height and width of input/output sample.
        in_channels (`int`, *optional*, defaults to 8): Number of channels in the input sample.
        out_channels (`int`, *optional*, defaults to 4): Number of channels in the output.
        down_block_types (`Tuple[str]`, *optional*, defaults to `("CrossAttnDownBlockSpatioTemporal", "CrossAttnDownBlockSpatioTemporal", "CrossAttnDownBlockSpatioTemporal", "DownBlockSpatioTemporal")`):
            The tuple of downsample blocks to use.
        up_block_types (`Tuple[str]`, *optional*, defaults to `("UpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal")`):
            The tuple of upsample blocks to use.
        block_out_channels (`Tuple[int]`, *optional*, defaults to `(320, 640, 1280, 1280)`):
            The tuple of output channels for each block.
        addition_time_embed_dim: (`int`, defaults to 256):
            Dimension to to encode the additional time ids.
        projection_class_embeddings_input_dim (`int`, defaults to 768):
            The dimension of the projection of encoded `added_time_ids`.
        layers_per_block (`int`, *optional*, defaults to 2): The number of layers per block.
        cross_attention_dim (`int` or `Tuple[int]`, *optional*, defaults to 1280):
            The dimension of the cross attention features.
        transformer_layers_per_block (`int`, `Tuple[int]`, or `Tuple[Tuple]` , *optional*, defaults to 1):
            The number of transformer blocks of type [`~models.attention.BasicTransformerBlock`]. Only relevant for
            [`~models.unets.unet_3d_blocks.CrossAttnDownBlockSpatioTemporal`],
            [`~models.unets.unet_3d_blocks.CrossAttnUpBlockSpatioTemporal`],
            [`~models.unets.unet_3d_blocks.UNetMidBlockSpatioTemporal`].
        num_attention_heads (`int`, `Tuple[int]`, defaults to `(5, 10, 10, 20)`):
            The number of attention heads.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 8,
        out_channels: int = 4,
        down_block_types: Tuple[str] = (
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "CrossAttnDownBlockSpatioTemporal",
            "DownBlockSpatioTemporal",
        ),
        up_block_types: Tuple[str] = (
            "UpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
            "CrossAttnUpBlockSpatioTemporal",
        ),
        block_out_channels: Tuple[int] = (320, 640, 1280, 1280),
        addition_time_embed_dim: int = 256,
        projection_class_embeddings_input_dim: int = 768,
        layers_per_block: Union[int, Tuple[int]] = 2,
        cross_attention_dim: Union[int, Tuple[int]] = 1024,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        num_attention_heads: Union[int, Tuple[int]] = (5, 10, 20, 20),
        num_frames: int = 25,
    ):
        super().__init__()

        self.sample_size = sample_size

        # input
        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size=3,
            padding=1,
        )

        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        
        if isinstance(layers_per_block, int):
            layers_per_block = [layers_per_block] * len(down_block_types)
            
        # down
        output_channel = block_out_channels[0]
        for i, down_block_type in enumerate(down_block_types):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            is_final_block = i == len(block_out_channels) - 1

            down_block = DownBlock(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=layers_per_block[i],
            )
            self.down_blocks.append(down_block)

        # mid
        self.mid_block = ResnetBlock(
            block_out_channels[-1],
            block_out_channels[-1]
        )

        # count how many layers upsample the images
        self.num_upsamplers = 0

        # up
        reversed_block_out_channels = list(reversed(block_out_channels))
        reversed_layers_per_block = list(reversed(layers_per_block))

        output_channel = reversed_block_out_channels[0]
        for i, up_block_type in enumerate(up_block_types):
            is_final_block = i == len(block_out_channels) - 1

            prev_output_channel = output_channel
            output_channel = reversed_block_out_channels[i]
            input_channel = reversed_block_out_channels[min(i + 1, len(block_out_channels) - 1)]

            # add upsample block for all BUT final layer
            if not is_final_block:
                add_upsample = True
                self.num_upsamplers += 1
            else:
                add_upsample = False

            up_block = UpBlock(
                in_channels=input_channel,
                prev_output_channel=prev_output_channel,
                out_channels=output_channel,
                num_layers=layers_per_block[i],
            )
            self.up_blocks.append(up_block)
            prev_output_channel = output_channel

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=32, eps=1e-5)
        self.conv_act = nn.SiLU()

        self.conv_out = nn.Conv2d(
            block_out_channels[0],
            out_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        sample: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[UNetEnvMapConditionOutput, Tuple]:
        r"""
        The [`UNetEnvMapConditionOutput`] forward method.

        Args:
            sample (`torch.Tensor`):
                The noisy input tensor with the following shape `(batch, num_frames, channel, height, width)`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] instead
                of a plain tuple.
        Returns:
            [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] or `tuple`:
                If `return_dict` is True, an [`~models.unet_slatio_temporal.UNetSpatioTemporalConditionOutput`] is
                returned, otherwise a `tuple` is returned where the first element is the sample tensor.
        """
        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        # sample = sample.flatten(0, 1)

        # 2. pre-process
        sample = self.conv_in(sample)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            sample, res_samples = downsample_block(sample)

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(sample)

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]
            sample = upsample_block(sample, res_samples)

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # 7. Reshape back to original shape
        # sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])
        sample = sample.reshape(batch_size, *sample.shape[1:])

        if not return_dict:
            return (sample,)

        return UNetEnvMapConditionOutput(sample=sample)
