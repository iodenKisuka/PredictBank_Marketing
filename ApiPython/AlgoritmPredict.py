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
