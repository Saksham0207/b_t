import os
from urllib import request

import torch
import torch.nn as nn
import progressbar

from tortoise.vocoder_prep.vocoder import UnivNetGenerator

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

class Vocoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocoder = UnivNetGenerator().cuda()
        self.vocoder.load_state_dict(torch.load(get_model_path('vocoder.pth', MODELS_DIR), map_location=torch.device('cuda'))['model_g'])
        self.vocoder.eval(inference=True)


    def forward(self, c):
        zero = torch.full((c.shape[0], self.vocoder.mel_channel, 10), -11.5129).to(c.device)
        mel = torch.cat((c, zero), dim=2)
        z = torch.randn(c.shape[0], self.vocoder.noise_dim, mel.size(2)).to(mel.device)
        with torch.no_grad():
            audio = self.vocoder(mel, z)
            audio = audio[:, :, :-(self.vocoder.hop_length * 10)]
            audio = audio.clamp(min=-1, max=1)
            return audio