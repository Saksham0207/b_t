import bentoml
from tortoise.service import svc

if __name__ == "__main__":
    text = "Joining two modalities results in a surprising increase in generalization!"
    voice = "pat"
    inp = text + "====" + voice

    server = bentoml.HTTPServer(svc, api_workers=1, port=3000)
    with server.start() as client:
        result = client.tts_with_preset("Write a python function that sort list of integers")
    print(result)