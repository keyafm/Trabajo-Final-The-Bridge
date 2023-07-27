from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
import xgboost as xgb

app = Flask(__name__)
model = pickle.load(open("model/xgb_film_model.pkl", "rb"))

# Mapeo de categorías de calificación por edades y género a números
calificacion_mapper = {'R': 1, 'PG': 2, 'G': 3, 'Not Rated': 4, 'NC-17': 5, 'Approved': 6, 'PG-13': 7, 'Unrated': 8, 'X': 9, 'TV-PG': 10, 'TV-MA': 11, 'TV-14': 12}
genero_mapper = {'Drama': 1, 'Adventure': 2, 'Action': 3, 'Comedy': 4, 'Horror': 5, 'Biography': 6, 'Crime': 7, 'Fantasy': 8, 'Family': 9, 'Animation': 10, 'Romance': 11, 'Music': 12, 'Western': 13, 'Thriller': 14, 'Sci-Fi': 15, 'Mystery': 16, 'Sport': 17, 'Musical': 18}

@app.route("/")
def home():
    return render_template("index.html") 

@app.route("/predict", methods=["POST"])
def predict_income():
    # Obtener los valores ingresados desde el formulario
    rating_text = request.form.get("rating")
    genre_text = request.form.get("genre")
    year = int(request.form.get("year"))
    votes = int(request.form.get("votes"))
    budget = int(request.form.get("budget"))
    runtime = int(request.form.get("runtime"))

    print("Valores ingresados:")
    print("Rating:", rating_text)
    print("Genre:", genre_text)
    print("Year:", year)
    print("Votes:", votes)
    print("Budget:", budget)
    print("Runtime:", runtime)

    # Convertir calificación por edades y género a números usando el mapeo
    rating = calificacion_mapper.get(rating_text, 0)
    genre = genero_mapper.get(genre_text, 0)

    # Crear un DataFrame con las características
    data = pd.DataFrame([[rating, genre, year, votes, budget, runtime]], columns=['rating', 'genre', 'year', 'votes', 'budget', 'runtime'])

    # Realizar la predicción usando el modelo
    prediction = model.predict(data)[0]

    return render_template("indext.html", prediction_text="El Ingreso Bruto de nuestra película es: {}".format(prediction))

if __name__ == "__main__":
    app.run(debug=True)

