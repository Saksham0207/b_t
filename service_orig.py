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
            diff_conds, best_results, best_latents = self.ar(text, voice)
            mel = self.diffusion(diff_conds, best_results, best_latents)
            wav = self.vocoder(mel).cpu()
            wav = self.aligner(wav, text)
            return wav.numpy()

tortoise_runner =  bentoml.Runner(TextToSpeech)
svc = bentoml.Service("tortoise_tts", runners=[tortoise_runner])


@svc.api(input=bentoml.io.Text(), output=bentoml.io.NumpyNdarray())
async def main(inputs):
    return await tortoise_runner.tts_with_preset.async_run(inputs)
    
if __name__ == "__main__":
    text = "Joining two modalities results in a surprising increase in generalization!"
    voice = "pat"
    inp = text + "====" + voice
    gen = main(inp)
    