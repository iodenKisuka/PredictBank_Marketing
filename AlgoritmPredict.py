import pandas as pd

class AlgoritmPredict:
    def __init__(self, file_path):
        self.file_path = file_path
    
    def process_file(self):
        # Lee el archivo CSV usando pandas
        df = pd.read_csv(self.file_path)
        
        # Realiza cualquier procesamiento adicional que necesites aquí
        # Por ejemplo, puedes realizar operaciones en el DataFrame df
        
        # Retorna el DataFrame o realiza cualquier acción adicional
        return df.to_dict()


    def Predictor():
        print("predictor ")
        tableAll = pd.read_csv("https://docs.google.com/spreadsheets/d/1rcPNLga760uabPzX-5YB739ipkpwDn2XIkIF4T8u3kE/export?format=csv")
        print("\nConjunto de datos:")
        print(tableAll.head())

        X = tableAll.drop(['y'], axis=1)
        Y = tableAll['y']