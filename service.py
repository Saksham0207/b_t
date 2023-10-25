import argparse
import asyncio

import bentoml


class TextToSpeech(bentoml.Runnable):
    """
    Main entry point into Tortoise.
    """
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True
    
    def __init__(self):

        """
        Constructor
        """

        
        from tortoise.ar_prep.model import AR
        from tortoise.diffusion_prep.model import Diffusion
        from tortoise.vocoder_prep.model import Vocoder
        from tortoise.postprocess.model import Postprocess

        self.ar = AR()
        self.diffusion = Diffusion()
        self.vocoder = Vocoder()
        self.aligner = Postprocess()

        
    #This uses the given settings to generate the audio
    @bentoml.Runnable.method(batchable=False, batch_dim=0)
    def tts_with_preset(self, inputs):
        """
        Calls TTS with one of a set of preset generation parameters. Options:
            'fast': Very fast, equivalent quality to a poor microphone
            'standard': Fast and good enough to use
        """
        import torch
        text, voice = inputs.split("====")
        with torch.no_grad():
            calm_token = 83  # This is the token for coding silence, which is fixed in place with "fix_autoregressive_output"
            diff_conds, best_results, best_latents = self.ar(text, voice)
            mel = self.diffusion(diff_conds, best_results, best_latents)
            wav = self.vocoder(mel).cpu()
            wav = self.aligner(wav, text)
            return wav

class AR(bentoml.Runnable):
    
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True
    
    def __init__(self):
        
        from tortoise.ar_prep.model import AR
        self.ar = AR()

    @bentoml.Runnable.method(batchable=False, batch_dim=0)
    def infer(self, text, voice):
        diff_conds, best_results, best_latents = self.ar(text, voice)
        return diff_conds, best_results, best_latents

class Diffusion(bentoml.Runnable):
    
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True
    
    def __init__(self):
        
        from tortoise.diffusion_prep.model import Diffusion
        
        self.diffusion = Diffusion()
    
    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def infer(self, diff_conds, best_results, best_latents):
        mel = self.diffusion(diff_conds, best_results, best_latents)
        return mel
        
class Vocoder(bentoml.Runnable):
    
    SUPPORTED_RESOURCES = ("nvidia.com/gpu", "cpu")
    SUPPORTS_CPU_MULTI_THREADING = True
    
    def __init__(self):
        
        from tortoise.vocoder_prep.model import Vocoder
        from tortoise.postprocess.model import Postprocess
        
        self.vocoder = Vocoder()
        self.aligner = Postprocess()
        
    @bentoml.Runnable.method(batchable=True, batch_dim=0)
    def infer(self, text, mel):
        wav = self.vocoder(mel).cpu()
        wav = self.aligner(wav, text)
        return wav

# tortoise_runner =  bentoml.Runner(TextToSpeech)
# svc = bentoml.Service("tortoise_tts", runners=[tortoise_runner])

ar_runner = bentoml.Runner(AR)
diff_runner = bentoml.Runner(Diffusion)
vocoder_runner = bentoml.Runner(Vocoder)
svc = bentoml.Service("tortoise_tts", runners=[ar_runner, diff_runner, vocoder_runner])


@svc.api(input=bentoml.io.Text(), output=bentoml.io.Text())
async def main(inputs):
    # gen = await tortoise_runner.tts_with_preset.async_run(inputs)
    text, voice = inputs.split("====")
    diff_conds, best_results, best_latents = await ar_runner.infer.async_run(text, voice)
    mel = await diff_runner.infer.async_run(diff_conds, best_results, best_latents)
    wav = await vocoder_runner.infer.async_run(text, mel)
    return wav

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, default="Joining two modalities results in a surprising increase in generalization!")
    parser.add_argument("--voice", type=str, default="pat")
    args = parser.parse_args()
    inp = args.text + "====" + args.voice
    gen = main(inp)
    # gen = asyncio.run(main(inp))
    # torchaudio.save('gen.wav', gen.squeeze(0).cpu(), 24000)
