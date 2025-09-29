import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

#Se for usar a data completa descomente a linha abaixo e comente a proxima
#df = pd.read_csv(r"data/IMDB Dataset.csv")
df = pd.read_csv(r"data/SubData.csv")

#Preparar strings para a vetorização
df["review"] = df["review"].str.lower()
df["review"] = df["review"].str.replace(r'[^a-z0-9\s]', "", regex= True)

#Trocar positivo para 1 e negativo para 0
df["sentiment"] = df["sentiment"].map({"positive" : 1, "negative" : 0})

x = (df["review"])
y = df["sentiment"]

x_treino, x_teste, y_treino, y_teste = train_test_split(x,y,test_size = 0.2, random_state = 100)

#vetorização das strings
vectorizer = TfidfVectorizer()
x_treino_vetorizado = vectorizer.fit_transform(x_treino)
x_teste_vetorizado = vectorizer.transform(x_teste)

print(f"Dados de treino vetorizado{x_treino_vetorizado.shape}")
print(f"Dados de teste vetorizado{x_teste_vetorizado.shape}")

import joblib
import os

#salvar os dados na pasta modelo
Pasta_modelo = "Dados_tratados_IMBD"
if not os.path.exists(Pasta_modelo):
    os.makedirs(Pasta_modelo)

joblib.dump(vectorizer, os.path.join(Pasta_modelo, "vectorizer.joblib"))

joblib.dump(y_treino, os.path.join(Pasta_modelo, "y_treino.joblib"))
joblib.dump(x_treino_vetorizado, os.path.join(Pasta_modelo, "x_treino_vetorizado.joblib"))

joblib.dump(x_teste_vetorizado, os.path.join(Pasta_modelo, "x_teste_vetorizado.joblib"))
joblib.dump(y_teste, os.path.join(Pasta_modelo, "y_teste.joblib"))