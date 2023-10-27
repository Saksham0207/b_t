import os
from glob import glob

import torch
import numpy as np
from scipy.io.wavfile import read
import boto3

from tortoise.ar_prep.stft import STFT


BUILTIN_VOICES_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../voices')

aws_access_key_id = os.environ.get('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
bucket_name = "tortoise-test"
session = boto3.Session(aws_access_key_id=aws_access_key_id, aws_secret_access_key=aws_secret_access_key)
s3 = session.client('s3')

TACOTRON_MEL_MAX = 2.3143386840820312
TACOTRON_MEL_MIN = -11.512925148010254


def normalize_tacotron_mel(mel):
    return 2 * ((mel - TACOTRON_MEL_MIN) / (TACOTRON_MEL_MAX - TACOTRON_MEL_MIN)) - 1


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    """
    PARAMS
    ------
    C: compression factor
    """
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression(x, C=1):
    """
    PARAMS
    ------
    C: compression factor used to compress
    """
    return torch.exp(x) / C


def load_voice(voice, extra_voice_dirs=[]):
    if voice == 'random':
        return None, None
    files = s3.list_objects(Bucket=bucket_name, Prefix=f"voices/{voice}/")["Contents"]

    conds = []
    for content in files:
        cond = np.frombuffer(s3.get_object(Bucket=bucket_name,
                                         Key=content["Key"])["Body"].read(),
                                         dtype=np.int16)
        if cond.dtype == np.int32:
            norm_fix = 2 ** 31
        elif cond.dtype == np.int16:
            norm_fix = 2 ** 15
        elif cond.dtype == np.float16 or cond.dtype == np.float32:
            norm_fix = 1.
        else:
            raise NotImplemented(f"Provided cond dtype not supported: {cond.dtype}")
        audio, lsr = torch.FloatTensor(cond.astype(np.float32)) / norm_fix, 22050
        if len(audio.shape) > 1:
            if audio.shape[0] < 5:
                audio = audio[0]
            else:
                assert audio.shape[1] < 5
                audio = audio[:, 0]
        if torch.any(audio > 2) or not torch.any(audio < 0):
            print(f"Error with {voice}. Max={audio.max()} min={audio.min()}")
        audio.clip_(-1, 1)
        conds.append(audio.unsqueeze(0))
    return conds, None


def load_voices(voices, extra_voice_dirs=[]):
    latents = []
    clips = []
    for voice in voices:
        if voice == 'random':
            if len(voices) > 1:
                print("Cannot combine a random voice with a non-random voice. Just using a random voice.")
            return None, None
        clip, latent = load_voice(voice, extra_voice_dirs)
        if latent is None:
            assert len(latents) == 0, "Can only combine raw audio voices or latent voices, not both. Do it yourself if you want this."
            clips.extend(clip)
        elif clip is None:
            assert len(clips) == 0, "Can only combine raw audio voices or latent voices, not both. Do it yourself if you want this."
            latents.append(latent)
    if len(latents) == 0:
        return clips, None
    else:
        latents_0 = torch.stack([l[0] for l in latents], dim=0).mean(dim=0)
        latents_1 = torch.stack([l[1] for l in latents], dim=0).mean(dim=0)
        latents = (latents_0,latents_1)
        return None, latents


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        from librosa.filters import mel as librosa_mel_fn
        mel_basis = librosa_mel_fn(
            sr=sampling_rate, n_fft=filter_length, n_mels=n_mel_channels, fmin=mel_fmin, fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert(torch.min(y.data) >= -10)
        assert(torch.max(y.data) <= 10)
        y = torch.clip(y, min=-1, max=1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output


def wav_to_univnet_mel(wav, do_normalization=False, device='cuda' if not torch.backends.mps.is_available() else 'mps'):
    stft = TacotronSTFT(1024, 256, 1024, 100, 24000, 0, 12000)
    stft = stft.to(device)
    mel = stft.mel_spectrogram(wav)
    if do_normalization:
        mel = normalize_tacotron_mel(mel)
    return mel
