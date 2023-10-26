import torch
import torch.nn as nn
from tortoise.postprocess.wav2vec_alignment import Wav2VecAlignment

class Postprocess(nn.Module):
    def __init__(self):
        super().__init__()
        self.aligner = Wav2VecAlignment()

    def forward(self, clip, text):
        with torch.no_grad():
            return self.aligner.redact(clip.squeeze(1), text).unsqueeze(1)
