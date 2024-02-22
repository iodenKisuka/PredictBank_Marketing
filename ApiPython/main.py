from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse,FileResponse
import AlgoritmPredict
import re
import pandas as pd
import os
import csv

app = FastAPI()

UPLOAD_DIRECTORY = "uploads"
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)


@app.get("/imgdispersion")
def plot_iris():
    try:
        url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
        iris = pd.read_csv(url)

        plt.scatter(iris['sepal_length'], iris['sepal_width'])
        plt.savefig('iris.png')

        with open('iris.png', mode="rb") as file:
            try:
                return StreamingResponse(file, media_type="image/png")
            finally:
                file.close()  # Cerrar el archivo
    except Exception as e:
        #return {"message": "Error al generar la imagen", "error": str(e)}
        raise HTTPException(status_code=500, detail="Archivo no encontrado")




@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    decoded_content = content.decode('utf-8')
    csv_reader = csv.reader(decoded_content.splitlines(), delimiter=',')
    # Continuar con el procesamiento del archivo CSV...


@app.post("/upload2")
async def upload_csv(file: UploadFile = File(...)):

    # Verifica que el archivo sea un archivo CSV y xlsx
    allowed_extensions = [".csv", ".xlsx"]
    file_extension = os.path.splitext(file.filename)[1]
    if file_extension.lower() not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Only CSV and XLSX files are allowed")

    # Guarda el archivo en el directorio de subidas
    with open(os.path.join(UPLOAD_DIRECTORY, file.filename), "wb") as buffer:
        buffer.write(await file.read())

    # Envia el archivo a la clase AlgoritmPredict
    #predictor = AlgoritmPredict(os.path.join(UPLOAD_DIRECTORY, file.filename))
    
    return {"filename": file.filename}