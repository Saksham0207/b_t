class TextToSpeech():
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
            return wav

tortoise_runner = TextToSpeech()
def main(inputs):
    gen = tortoise_runner.tts_with_preset(inputs)
    print(type(gen), gen.device)
    return gen

if __name__ == "__main__":
    import torchaudio
    text = "Joining two modalities results in a surprising increase in generalization!"
    voice = "pat"
    inp = text + "====" + voice
    gen = main(inp)
    torchaudio.save('genas.wav', gen.squeeze(0), 24000)