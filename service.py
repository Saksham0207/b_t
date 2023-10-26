import argparse
import bentoml

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
        from tortoise.vocoder_prep.model import Vocoder
        from tortoise.postprocess.model import Postprocess
        
        self.diffusion = Diffusion()
        self.vocoder = Vocoder()
        self.aligner = Postprocess()
    
    @bentoml.Runnable.method(batchable=False, batch_dim=0)
    def infer(self, diff_conds, best_results, best_latents, text):
        mel = self.diffusion(diff_conds, best_results, best_latents)
        wav = self.vocoder(mel).cpu()
        wav = self.aligner(wav, text)
        return wav.detach().numpy()

ar_runner = bentoml.Runner(AR)
diff_runner = bentoml.Runner(Diffusion)
svc = bentoml.Service("tortoise_tts", runners=[ar_runner, diff_runner])


@svc.api(input=bentoml.io.Text(), output=bentoml.io.Text())
async def main(inputs):
    # gen = await tortoise_runner.tts_with_preset.async_run(inputs)
    text, voice = inputs.split("====")
    diff_conds, best_results, best_latents = await ar_runner.infer.async_run(text, voice)
    wav = await diff_runner.infer.async_run(diff_conds, best_results, best_latents, text)
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
