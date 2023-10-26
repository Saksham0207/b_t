import os
import random
from urllib import request

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import progressbar
# from tortoise.preprocessing.tokenizer import VoiceBpeTokenizer
# from tortoise.preprocessing.audio import load_voices

# from autoregressive import UnifiedVoice
# from tokenizer import VoiceBpeTokenizer
# from audio import load_voices

from tortoise.ar_prep.autoregressive import UnifiedVoice
from tortoise.ar_prep.tokenizer import VoiceBpeTokenizer
from tortoise.ar_prep.audio import load_voices, wav_to_univnet_mel

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

def fix_autoregressive_output(codes, stop_token, complain=True):
    """
    This function performs some padding on coded audio that fixes a mismatch issue between what the diffusion model was
    trained on and what the autoregressive code generator creates (which has no padding or end).
    This is highly specific to the DVAE being used, so this particular coding will not necessarily work if used with
    a different DVAE. This can be inferred by feeding a audio clip padded with lots of zeros on the end through the DVAE
    and copying out the last few codes.

    Failing to do this padding will produce speech with a harsh end that sounds like "BLAH" or similar.
    """
    # Strip off the autoregressive stop token and add padding.
    stop_token_indices = (codes == stop_token).nonzero()
    if len(stop_token_indices) == 0:
        if complain:
            print("No stop tokens found in one of the generated voice clips. This typically means the spoken audio is "
                  "too long. In some cases, the output will still be good, though. Listen to it and if it is missing words, "
                  "try breaking up your input text.")
        return codes
    else:
        codes[stop_token_indices] = 83
    stm = stop_token_indices.min().item()
    codes[stm:] = 83
    if stm - 3 < codes.shape[0]:
        codes[-3] = 45
        codes[-2] = 45
        codes[-1] = 248

    return codes

def pad_or_truncate(t, length):
    """
    Utility function for forcing <t> to have the specified sequence length, whether by clipping it or padding it with 0s.
    """
    if t.shape[-1] == length:
        return t
    elif t.shape[-1] < length:
        return F.pad(t, (0, length-t.shape[-1]))
    else:
        return t[..., :length]

class TorchMelSpectrogram(nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024, n_mel_channels=80, mel_fmin=0, mel_fmax=8000,
                 sampling_rate=22050, normalize=False, mel_norm_file=DEFAULT_MEL_NORM_FILE):
        super().__init__()
        # These are the default tacotron values for the MEL spectrogram.
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.sampling_rate = sampling_rate
        self.mel_stft = torchaudio.transforms.MelSpectrogram(n_fft=self.filter_length, hop_length=self.hop_length,
                                                             win_length=self.win_length, power=2, normalized=normalize,
                                                             sample_rate=self.sampling_rate, f_min=self.mel_fmin,
                                                             f_max=self.mel_fmax, n_mels=self.n_mel_channels,
                                                             norm="slaney")
        self.mel_norm_file = mel_norm_file
        if self.mel_norm_file is not None:
            self.mel_norms = torch.load(self.mel_norm_file)
        else:
            self.mel_norms = None

    def forward(self, inp):
        if len(inp.shape) == 3:  # Automatically squeeze out the channels dimension if it is present (assuming mono-audio)
            inp = inp.squeeze(1)
        assert len(inp.shape) == 2
        if torch.backends.mps.is_available():
            inp = inp.to('cpu')
        self.mel_stft = self.mel_stft.to(inp.device)
        mel = self.mel_stft(inp)
        # Perform dynamic range compression
        mel = torch.log(torch.clamp(mel, min=1e-5))
        if self.mel_norms is not None:
            self.mel_norms = self.mel_norms.to(mel.device)
            mel = mel / self.mel_norms.unsqueeze(0).unsqueeze(-1)
        return mel

class AR(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = VoiceBpeTokenizer(
                  vocab_file=None,
                  use_basic_cleaners=False,
              )
        self.autoregressive = UnifiedVoice(max_mel_tokens=604, max_text_tokens=402, max_conditioning_inputs=2, layers=30,
                                          model_dim=1024,
                                          heads=16, number_text_tokens=255, start_text_token=255, checkpointing=False,
                                          train_solo_embeddings=False).cuda().eval()
        self.autoregressive.load_state_dict(torch.load(get_model_path('autoregressive.pth', MODELS_DIR)), strict=False)
        self.autoregressive.post_init_gpt2_config(use_deepspeed=True, kv_cache=True, half=True)

    def format_conditioning(self, clip, cond_length=132300, device="cuda"):
        """
        Converts the given conditioning signal to a MEL spectrogram and clips it as expected by the models.
        """
        gap = clip.shape[-1] - cond_length
        if gap < 0:
            clip = F.pad(clip, pad=(0, abs(gap)))
        elif gap > 0:
            rand_start = random.randint(0, gap)
            clip = clip[:, rand_start:rand_start + cond_length]
        mel_clip = TorchMelSpectrogram()(clip.unsqueeze(0)).squeeze(0)
        return mel_clip.unsqueeze(0).to(device)

    def forward(self, text, voice):
        word_count = len(text.split(" "))
        max_mel_tokens = word_count*13 + 15
        text_tokens = torch.IntTensor(self.tokenizer.encode(text)).unsqueeze(0).cuda()
        text_tokens = F.pad(text_tokens, (0, 1))

        if '&' in voice:
            voice_sel = voice.split('&')
        else:
            voice_sel = [voice]

        voice_samples, conditioning_latents = load_voices(voice_sel) 
        # with torch.no_grad():
        voice_samples = [v.to(torch.device('cuda')) for v in voice_samples]

        auto_conds = []
        if not isinstance(voice_samples, list):
            voice_samples = [voice_samples]
        for vs in voice_samples:
            auto_conds.append(self.format_conditioning(vs, device=torch.device('cuda')))
        auto_conds = torch.stack(auto_conds, dim=1)
        auto_latent = self.autoregressive.get_conditioning(auto_conds)

        diffusion_conds = []
        for sample in voice_samples:
            # The diffuser operates at a sample rate of 24000 (except for the latent inputs)
            sample = torchaudio.functional.resample(sample, 22050, 24000)
            sample = pad_or_truncate(sample, 102400)
            cond_mel = wav_to_univnet_mel(sample.to(torch.device("cuda")), do_normalization=False, device=torch.device("cuda"))
            diffusion_conds.append(cond_mel)
        diffusion_conds = torch.stack(diffusion_conds, dim=1)
        
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                codes = self.autoregressive.inference_speech(auto_latent, text_tokens,
                                                                    do_sample=True,
                                                                    top_p=0.05,
                                                                    temperature=.3,
                                                                    num_return_sequences=1,
                                                                    length_penalty=-5.0,
                                                                    repetition_penalty=5.0,
                                                                    max_generate_length=max_mel_tokens)
                padding_needed = max_mel_tokens - codes.shape[1]
                codes = F.pad(codes, (0, padding_needed),
                            value=self.autoregressive.stop_mel_token)
                for i in range(codes.shape[0]):
                    codes[i] = fix_autoregressive_output(codes[i],
                                                        self.autoregressive.stop_mel_token)

                best_latents = self.autoregressive(auto_latent.repeat(1, 1), text_tokens.repeat(1, 1),
                                                        torch.tensor([text_tokens.shape[-1]], device=text_tokens.device), codes,
                                                        torch.tensor([codes.shape[-1]*self.autoregressive.mel_length_compression], device=text_tokens.device),
                                                        return_latent=True, clip_inputs=False)
        return diffusion_conds, codes, best_latents

if __name__ == "__main__":
    text = "Joining two modalities results in a surprising increase in generalization! What would happen if we combined them all?"
    voice = "pat"

    preprocessor = AR()
    d, c, b = preprocessor(text, voice)
    print(c.shape, b.shape)