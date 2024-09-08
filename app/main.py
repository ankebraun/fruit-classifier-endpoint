from pydantic import BaseModel # this has nothing to do my machine learning models
# this is just the parent class for everything that is strictly typed in Pydantic
from fastapi import FastAPI, Depends, UploadFile, File
# we need this to upload images to fastAPI
# this is the Python image library
from PIL import Image
# This is what we use the BaseModel for
# the result is strictly typed so that it returns
# a string for the category (label that we predict)
# and a float for the confidence (the probability for the label)
class Result(BaseModel):
    category: str
    confidence: float
# this creates an instance for the endpoint
app = FastAPI()
@app.post('/predict', response_model=Result )
async def predict(
    input_image: UploadFile = File(...),
):
    pass