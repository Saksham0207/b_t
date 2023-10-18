import math
import random
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast

from tortoise.diffusion_prep.xtransformers import RelativePositionBias

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)

def normalization(channels):
    """
    Make a standard normalization layer.

    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    groups = 32
    if channels <= 16:
        groups = 8
    elif channels <= 64:
        groups = 16
    while channels % groups != 0:
        groups = int(groups / 2)
    assert groups > 2
    return GroupNorm32(groups, channels)

class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/output heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, mask=None, rel_pos=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * torch.tensor(self.n_heads))
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / torch.sqrt(torch.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        
        if rel_pos is not None:
            weight = rel_pos(weight.reshape(bs, self.n_heads, weight.shape[-2], weight.shape[-1])).reshape(bs * self.n_heads, weight.shape[-2], weight.shape[-1])
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)

        return a.reshape(bs, -1, length)


class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        do_checkpoint=True,
        relative_pos_embeddings=False,
    ):
        super().__init__()
        self.channels = channels
        self.do_checkpoint = do_checkpoint
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.norm = normalization(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        # split heads before split qkv
        self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(nn.Conv1d(channels, channels, 1))
        if relative_pos_embeddings:
            self.relative_pos_embeddings = RelativePositionBias(scale=(channels // self.num_heads) ** .5, causal=False, heads=num_heads, num_buckets=32, max_distance=64)
        else:
            self.relative_pos_embeddings = None

    def forward(self, x, mask=None):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv, mask, self.relative_pos_embeddings)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)

def is_latent(t):
    return t.dtype == torch.float


def is_sequence(t):
    return t.dtype == torch.long


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    # if dim % 2:
    #     embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    # else:
    #   print("dim > 2")
    return embedding


class TimestepBlock(nn.Module):
    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class ResBlock(TimestepBlock):
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        dims=2,
        kernel_size=3,
        efficient_config=True,
        use_scale_shift_norm=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm
        padding = {1: 0, 3: 1, 5: 2}[kernel_size]
        eff_kernel = 1 if efficient_config else 3
        eff_padding = 0 if efficient_config else 1

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv1d(channels, self.out_channels, eff_kernel, padding=eff_padding),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
                nn.Conv1d(self.out_channels, self.out_channels, kernel_size, padding=padding),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv1d(channels, self.out_channels, eff_kernel, padding=eff_padding)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        # while len(emb_out.shape) < len(h.shape):
        emb_out = emb_out[..., None]
        out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = out_norm(h) * (1 + scale) + shift
        h = out_rest(h)
        return self.skip_connection(x) + h


class DiffusionLayer(TimestepBlock):
    def __init__(self, model_channels, dropout, num_heads):
        super().__init__()
        self.resblk = ResBlock(model_channels, model_channels, dropout, model_channels, dims=1, use_scale_shift_norm=True)
        self.attn = AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True)

    def forward(self, x, time_emb):
        y = self.resblk(x, time_emb)
        return self.attn(y)


class DiffusionTts(nn.Module):
    def __init__(
            self,
            model_channels=512,
            num_layers=8,
            in_channels=100,
            in_latent_channels=512,
            in_tokens=8193,
            out_channels=200,  # mean and variance
            dropout=0,
            use_fp16=False,
            num_heads=16,
            # Parameters for regularization.
            layer_drop=.1,
            unconditioned_percentage=.1,  # This implements a mechanism similar to what is used in classifier-free training.
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.num_heads = num_heads
        self.unconditioned_percentage = unconditioned_percentage
        self.enable_fp16 = use_fp16
        self.layer_drop = layer_drop

        self.inp_block = nn.Conv1d(in_channels, model_channels, 3, 1, 1)
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels),
        )

        # Either code_converter or latent_converter is used, depending on what type of conditioning data is fed.
        # This model is meant to be able to be trained on both for efficiency purposes - it is far less computationally
        # complex to generate tokens, while generating latents will normally mean propagating through a deep autoregressive
        # transformer network.
        self.code_embedding = nn.Embedding(in_tokens, model_channels)
        self.code_converter = nn.Sequential(
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
        )
        self.code_norm = normalization(model_channels)
        self.latent_conditioner = nn.Sequential(
            nn.Conv1d(in_latent_channels, model_channels, 3, padding=1),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
            AttentionBlock(model_channels, num_heads, relative_pos_embeddings=True),
        )
        self.contextual_embedder = nn.Sequential(nn.Conv1d(in_channels,model_channels,3,padding=1,stride=2),
                                                 nn.Conv1d(model_channels, model_channels*2,3,padding=1,stride=2),
                                                 AttentionBlock(model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                 AttentionBlock(model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                 AttentionBlock(model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                 AttentionBlock(model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False),
                                                 AttentionBlock(model_channels*2, num_heads, relative_pos_embeddings=True, do_checkpoint=False))
        self.unconditioned_embedding = nn.Parameter(torch.randn(1,model_channels,1))
        self.conditioning_timestep_integrator = TimestepEmbedSequential(
            DiffusionLayer(model_channels, dropout, num_heads),
            DiffusionLayer(model_channels, dropout, num_heads),
            DiffusionLayer(model_channels, dropout, num_heads),
        )

        self.integrating_conv = nn.Conv1d(model_channels*2, model_channels, kernel_size=1)
        self.mel_head = nn.Conv1d(model_channels, in_channels, kernel_size=3, padding=1)

        self.layers = nn.ModuleList([DiffusionLayer(model_channels, dropout, num_heads) for _ in range(num_layers)] +
                                    [ResBlock(model_channels, model_channels, dropout, dims=1, use_scale_shift_norm=True) for _ in range(3)])

        self.out = nn.Sequential(
            normalization(model_channels),
            nn.SiLU(),
            nn.Conv1d(model_channels, out_channels, 3, padding=1),
        )

    def get_grad_norm_parameter_groups(self):
        groups = {
            'minicoder': list(self.contextual_embedder.parameters()),
            'layers': list(self.layers.parameters()),
            'code_converters': list(self.code_embedding.parameters()) + list(self.code_converter.parameters()) + list(self.latent_conditioner.parameters()) + list(self.latent_conditioner.parameters()),
            'timestep_integrator': list(self.conditioning_timestep_integrator.parameters()) + list(self.integrating_conv.parameters()),
            'time_embed': list(self.time_embed.parameters()),
        }
        return groups

    def get_conditioning(self, conditioning_input):
        speech_conditioning_input = conditioning_input.unsqueeze(1) if len(
            conditioning_input.shape) == 3 else conditioning_input
        conds = []
        for j in range(speech_conditioning_input.shape[1]):
            conds.append(self.contextual_embedder(speech_conditioning_input[:, j]))
        conds = torch.cat(conds, dim=-1)
        conds = conds.mean(dim=-1)
        return conds

    def timestep_independent(self, aligned_conditioning, conditioning_latent, expected_seq_len, return_code_pred):
        # Shuffle aligned_latent to BxCxS format
        if is_latent(aligned_conditioning):
            aligned_conditioning = aligned_conditioning.permute(0, 2, 1)

        cond_scale, cond_shift = torch.chunk(conditioning_latent, 2, dim=1)
        if is_latent(aligned_conditioning):
            code_emb = self.latent_conditioner(aligned_conditioning)
        else:
            code_emb = self.code_embedding(aligned_conditioning).permute(0, 2, 1)
            code_emb = self.code_converter(code_emb)
        code_emb = self.code_norm(code_emb) * (1 + cond_scale.unsqueeze(-1)) + cond_shift.unsqueeze(-1)

        unconditioned_batches = torch.zeros((code_emb.shape[0], 1, 1), device=code_emb.device)
        # Mask out the conditioning branch for whole batch elements, implementing something similar to classifier-free guidance.
        if self.training and self.unconditioned_percentage > 0:
            unconditioned_batches = torch.rand((code_emb.shape[0], 1, 1),
                                               device=code_emb.device) < self.unconditioned_percentage
            code_emb = torch.where(unconditioned_batches, self.unconditioned_embedding.repeat(aligned_conditioning.shape[0], 1, 1),
                                   code_emb)
        expanded_code_emb = F.interpolate(code_emb, size=expected_seq_len, mode='nearest')

        if not return_code_pred:
            return expanded_code_emb
        else:
            mel_pred = self.mel_head(expanded_code_emb)
            # Multiply mel_pred by !unconditioned_branches, which drops the gradient on unconditioned branches. This is because we don't want that gradient being used to train parameters through the codes_embedder as it unbalances contributions to that network from the MSE loss.
            mel_pred = mel_pred * unconditioned_batches.logical_not()
            return expanded_code_emb, mel_pred

    # def forward(self, inp_x, timesteps, precomputed_aligned_embeddings=None, aligned_conditioning=None, conditioning_latent=None, conditioning_free=False, return_code_pred=False):
    def forward(self, inp_x, timesteps, precomputed_aligned_embeddings):
    # def forward(self, inp_x, comb_inp):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param aligned_conditioning: an aligned latent or sequence of tokens providing useful data about the sample to be produced.
        :param conditioning_latent: a pre-computed conditioning latent; see get_conditioning().
        :param precomputed_aligned_embeddings: Embeddings returned from self.timestep_independent()
        :param conditioning_free: When set, all conditioning inputs (including tokens and conditioning_input) will not be considered.
        :return: an [N x C x ...] Tensor of outputs.
        """
        # assert precomputed_aligned_embeddings is not None or (aligned_conditioning is not None and conditioning_latent is not None)
        # assert not (return_code_pred and precomputed_aligned_embeddings is not None)  # These two are mutually exclusive.

        # unused_params = []
        # if conditioning_free:
        #     code_emb = self.unconditioned_embedding.repeat(x.shape[0], 1, x.shape[-1])
            # unused_params.extend(list(self.code_converter.parameters()) + list(self.code_embedding.parameters()))
            # unused_params.extend(list(self.latent_conditioner.parameters()))
        # else:
        # if precomputed_aligned_embeddings is not None:
        
        code_emb = precomputed_aligned_embeddings
        # code_emb = comb_inp[0, :, :].unsqueeze(0)

        # else:
        #     code_emb, mel_pred = self.timestep_independent(aligned_conditioning, conditioning_latent, x.shape[-1], True)
        #     if is_latent(aligned_conditioning):
        #         unused_params.extend(list(self.code_converter.parameters()) + list(self.code_embedding.parameters()))
        #     else:
        #         unused_params.extend(list(self.latent_conditioner.parameters()))

        # unused_params.append(self.unconditioned_embedding)
        time_emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        # time_emb = self.time_embed(timestep_embedding(comb_inp[1, 0, 0].unsqueeze(0).to(torch.int),
        #                            self.model_channels))
        
        code_emb = self.conditioning_timestep_integrator(code_emb, time_emb)
        x = self.inp_block(inp_x)
        x = torch.cat([x, code_emb], dim=1)
        x = self.integrating_conv(x)
        for i, lyr in enumerate(self.layers):
            with autocast(x.device.type, enabled=self.enable_fp16 and i != 0):
                x = lyr(x, time_emb)
          

        x = x.float()
        out = self.out(x)

        code_emb_uncond = self.unconditioned_embedding.repeat(x.shape[0], 1, x.shape[-1])
        
        time_emb_uncond = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        # time_emb_uncond = self.time_embed(timestep_embedding(comb_inp[0, 0, 0].unsqueeze(0).to(torch.int),
        #                            self.model_channels))
        
        code_emb = self.conditioning_timestep_integrator(code_emb_uncond, time_emb_uncond)
        x = self.inp_block(inp_x)
        x = torch.cat([x, code_emb], dim=1)
        x = self.integrating_conv(x)
        for i, lyr in enumerate(self.layers):
            with autocast(x.device.type, enabled=self.enable_fp16 and i != 0):
                x = lyr(x, time_emb)
      
        x = x.float()
        out_no_cond = self.out(x)
        # print(out.shape, out_no_cond.shape)
        return out, out_no_cond


if __name__ == '__main__':
    clip = torch.randn(2, 100, 400)
    aligned_latent = torch.randn(2,388,512)
    aligned_sequence = torch.randint(0,8192,(2,100))
    cond = torch.randn(2, 100, 400)
    ts = torch.LongTensor([600, 600])
    model = DiffusionTts(512, layer_drop=.3, unconditioned_percentage=.5)
    # Test with latent aligned conditioning
    #o = model(clip, ts, aligned_latent, cond)
    # Test with sequence aligned conditioning
    o = model(clip, ts, aligned_sequence, cond)

