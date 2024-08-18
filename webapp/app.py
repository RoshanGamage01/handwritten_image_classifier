import numpy as np
import network as nn
import read_data as rd

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles

app = FastAPI()

class ImageInputData(BaseModel):
    image_data: list


@app.post("/clasify")
async def clasify_image(image: ImageInputData):
    image_data = np.array(image.image_data)
    
    network = nn.Network([784, 30, 30, 10])
    network.load_parameters("./data/parameters_95acc_opt.json")

    image_data = image_data.reshape(-1, 1)

    result = network.feedforward(image_data)

    prediction = np.argmax(result)

    probability_percentage = [[prob[0] * 100] for prob in result]

    return {
        "prediction": int(prediction),
        "probabitlities": probability_percentage
    }

@app.get("/loadRandom")
async def load_random_image():
    _, _, test_data = rd.load_data()

    random_index = np.random.randint(0, 10000)
    return {"data": test_data[0][random_index].tolist()}

app.mount("/", StaticFiles(directory="static", html=True), name="static")