import joblib
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report

Pasta_modelo = "Dados_tratados_IMBD"

#Carrega as variaveis geradas no outro programa
try:
    vectorizer = joblib.load(os.path.join(Pasta_modelo, "vectorizer.joblib"))

    x_treino_vetorizado = joblib.load(os.path.join(Pasta_modelo, "x_treino_vetorizado.joblib"))
    x_teste_vetorizado = joblib.load(os.path.join(Pasta_modelo, "x_teste_vetorizado.joblib"))

    y_treino = joblib.load(os.path.join(Pasta_modelo, "y_treino.joblib"))
    y_teste = joblib.load(os.path.join(Pasta_modelo, "y_teste.joblib"))

except FileNotFoundError:
    print("Arquivo não encontrado")
    exit()

#Treino do modelo

model = LogisticRegression(random_state=100)

model.fit(x_treino_vetorizado,y_treino)

model_predicts = model.predict(x_teste_vetorizado)

print("\n--- Avaliação de performance ---\n")
print(classification_report(y_teste,model_predicts))

#salva O modelo para uso futuro
joblib.dump(model, os.path.join(Pasta_modelo, "Modelo.joblib"))