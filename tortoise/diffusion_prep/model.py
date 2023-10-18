import os
import random
from urllib import request

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import progressbar

from tortoise.diffusion_prep.diffusion_decoder import DiffusionTts
from tortoise.diffusion_prep.diffusion import SpacedDiffusion, space_timesteps, get_named_beta_schedule

pbar = None
DEFAULT_MODELS_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'tortoise', 'models')
MODELS_DIR = os.environ.get('TORTOISE_MODELS_DIR', DEFAULT_MODELS_DIR)
MODELS = {
    'autoregressive.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/autoregressive.pth',
    'classifier.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/classifier.pth',
    'clvp2.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/clvp2.pth',
    'cvvp.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/cvvp.pth',
    'diffusion_decoder.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/diffusion_decoder.pth',
    'vocoder.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/vocoder.pth',
    'rlg_auto.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/rlg_auto.pth',
    'rlg_diffuser.pth': 'https://huggingface.co/jbetker/tortoise-tts-v2/resolve/main/.models/rlg_diffuser.pth',
}
DEFAULT_MEL_NORM_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../data/mel_norms.pth')

def download_models(specific_models=None):
    """
    Call to download all the models that Tortoise uses.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    def show_progress(block_num, block_size, total_size):
        global pbar
        if pbar is None:
            pbar = progressbar.ProgressBar(maxval=total_size)
            pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            pbar.update(downloaded)
        else:
            pbar.finish()
            pbar = None
    for model_name, url in MODELS.items():
        if specific_models is not None and model_name not in specific_models:
            continue
        model_path = os.path.join(MODELS_DIR, model_name)
        if os.path.exists(model_path):
            continue
        print(f'Downloading {model_name} from {url}...')
        request.urlretrieve(url, model_path, show_progress)
        print('Done.')


def get_model_path(model_name, models_dir=MODELS_DIR):
    """
    Get path to given model, download it if it doesn't exist.
    """
    if model_name not in MODELS:
        raise ValueError(f'Model {model_name} not found in available models.')
    model_path = os.path.join(models_dir, model_name)
    if not os.path.exists(model_path) and models_dir == MODELS_DIR:
        download_models([model_name])
    return model_path


TACOTRON_MEL_MAX = 2.3143386840820312
TACOTRON_MEL_MIN = -11.512925148010254

def denormalize_tacotron_mel(norm_mel):
    return ((norm_mel+1)/2)*(TACOTRON_MEL_MAX-TACOTRON_MEL_MIN)+TACOTRON_MEL_MIN


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.diffuser = SpacedDiffusion(use_timesteps=space_timesteps(4000, [6]), model_mean_type='epsilon',
                           model_var_type='learned_range', loss_type='mse', betas=get_named_beta_schedule('linear', 4000),
                           conditioning_free=True, conditioning_free_k=2.0)
        self.diffusion = DiffusionTts(model_channels=1024, num_layers=10, in_channels=100, out_channels=200,
                                      in_latent_channels=1024, in_tokens=8193, dropout=0, use_fp16=False, num_heads=16,
                                      layer_drop=0, unconditioned_percentage=0).cuda().eval()
        self.diffusion.load_state_dict(torch.load(get_model_path('diffusion_decoder.pth', MODELS_DIR)))

    def forward(self, diffusion_conds, best_results, best_latents):
        diffusion_conditioning = self.diffusion.get_conditioning(diffusion_conds)
        
        codes = best_results[0].unsqueeze(0)
        latents = best_latents[0].unsqueeze(0)

        # Find the first occurrence of the "calm" token and trim the codes to that.
        ctokens = 0
        for k in range(codes.shape[-1]):
            if codes[0, k] == 83:
                ctokens += 1
            else:
                ctokens = 0
            if ctokens > 8:  # 8 tokens gives the diffusion model some "breathing room" to terminate speech.
                latents = latents[:, :k]
                break
                
        with torch.no_grad():
            output_seq_len = latents.shape[1] * 4 * 24000 // 22050  # This diffusion model converts from 22kHz spectrogram codes to a 24kHz spectrogram signal.
            output_shape = (latents.shape[0], 100, output_seq_len)
            precomputed_embeddings = self.diffusion.timestep_independent(latents,
                                            diffusion_conditioning, output_seq_len, False)

            noise = torch.randn(output_shape, device=latents.device) * 1
            
            mel = self.diffuser.p_sample_loop(self.diffusion, output_shape, noise=noise,
                                          model_kwargs={'precomputed_aligned_embeddings': precomputed_embeddings},
                                        progress=False)
        return denormalize_tacotron_mel(mel)[:,:,:output_seq_len]
        



