import bentoml
from trotoise.api import svc


if __name__ == "__main__":
    text = "Joining two modalities results in a surprising increase in generalization!"
    voice = "pat"
    inp = text + "====" + voice
    server = bentoml.HTTPServer(svc, api_workers=1)
    with server.start() as client:
        gen = client.tts_with_preset(inp)