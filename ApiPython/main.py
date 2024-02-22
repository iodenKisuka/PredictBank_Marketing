from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse,FileResponse
import re
import pandas as pd
import os
import csv
#import AlgoritmPredict
import seaborn as sns

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




@app.post("/upload2")
async def upload_file(file: UploadFile = File(...)):
    content = await file.read()
    decoded_content = content.decode('utf-8')
    csv_reader = csv.reader(decoded_content.splitlines(), delimiter=',')
    # Continuar con el procesamiento del archivo CSV...


@app.post("/upload")
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
    predict = AlgoritmPredict(os.path.join(UPLOAD_DIRECTORY, file.filename))
    #return {"filename": file.filename}
    FilePredict= predict.TrainPredictor()
    return {"Prediciones": FilePredict}


class SeparateXY():
    def __init__(self, file_path):
        self.file_path = file_path
        self.X = None
        self.Y = None
        SeparaXeY()

    def SeparaXeY():
        # Carga de datos
        tableAll = pd.read_csv(self.file_path)
        print("\nConjunto de datos:")
        # Convertir a DataFrame
        df = pd.DataFrame(tableAll)
        # Seleccionar solo las características categóricas
        categorical_features = df.select_dtypes(include=['object'])
        # Crear un codificador OneHotEncoder
        one_hot_encoder = OneHotEncoder(sparse=False, drop='first')
        # Aplicar OneHotEncoder a las características categóricas y transformarlas
        one_hot_encoded = one_hot_encoder.fit_transform(categorical_features)
        # Convertir las características transformadas en un DataFrame
        one_hot_encoded_df = pd.DataFrame(one_hot_encoded, columns=one_hot_encoder.get_feature_names_out(categorical_features.columns))
        # Concatenar las características codificadas con el DataFrame original
        df_encoded = pd.concat([df.drop(columns=categorical_features.columns), one_hot_encoded_df], axis=1)
        # División de características y variable objetivo
        self.X = df_encoded.drop(['y_yes'], axis=1)
        self.Y = df_encoded['y_yes']


class AlgoritmPredict:
    def __init__(self, file_path):
        self.file_path = file_path
        self.attributes= None
    
    
    def TrainPredictor():
        cvsorigin= "https://docs.google.com/spreadsheets/d/1rcPNLga760uabPzX-5YB739ipkpwDn2XIkIF4T8u3kE/export?format=csv"
        separacionxy = SeparateXY(cvsorigin)

        # Estandarización de características
        scaler = StandardScaler()
        # Ajustar y transformar los datos
        tableXStandarize = scaler.fit_transform(X)
        # Convertir el resultado a DataFrame
        X = pd.DataFrame(tableXStandarize, columns=X.columns)

        correlation_matrix = tableAll.corr()
        attributes = X.columns

        #Remove attributes X with low correlation respect to Y      
        selector = SelectKBest(f_regression, k=10)
        X =selector.fit_transform(X, Y)

        #Selected features
        cols = selector.get_support(indices=True)
        self.attributes = attributes[cols]
        print(cols)
        standarX= tableXStandarize[cols]
        print("\nSelected Features:")
        print(attributes)
        #Use 90% of instances to train and 10% to test.
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.9, random_state=0)

        # Entrenamiento del modelo de regresión logística
        clf = LogisticRegression(random_state=0)
        clf.fit(x_train, y_train)

        # Predicción y evaluación del modelo
        y_pred = clf.predict(x_test)
        print(y_pred)
        accuracy = clf.score(x_test, y_test)
        print("\nPrecisión del modelo:", accuracy)

        testSeparacionxy = SeparateXY(self.file_path)
        FilePredict = clf.predict(testSeparacionxy.X) 
        print("Hola")
        print(FilePredict)
        #return FilePredict
        return "hola"

#