from fastapi import FastAPI
import pandas as pd
from fastapi.responses import JSONResponse
from sklearn.metrics.pairwise import cosine_similarity

# Nota para mi: Cuando abras este proyecto, activa siempre el ambiente desde la terminal.
# Estando en la carpeta del poyecto: fastapi-env\Scripts\activate.bat y luego uvicorn main:app --reload

# http://127.0.0.1:8000

app = FastAPI()


# 1)
@app.get("/developer")
async def developer(desarrollador: str):
    df_games = pd.read_parquet('D:/escritorio/STEAM/parquet/output_game.parquet')
    
    # Filtra por empresa desarrolladora
    filtro_dev = df_games[df_games['developer'] == desarrollador.capitalize()] 
    # Según el desarrollador, agrupa los datos por año y cuenta la cantidad de valores de cada año
    cantidad_x_anio = filtro_dev.groupby('release_year')['id'].size()
    # Cuenta la cantidad de juegos Free (0) para ese desarrollador por año
    cantidad_gratis = filtro_dev[filtro_dev['price'] == 0].groupby('release_year')['id'].count()
    # Calcula el porcentaje
    porcentaje_contenido = (cantidad_gratis / cantidad_x_anio * 100).fillna(0).astype(int)
    # Crear un DataFrame con los valores
    output = pd.DataFrame({'Año': cantidad_x_anio.index, 'Cantidad': cantidad_x_anio.values, 'porcentaje contenido free': porcentaje_contenido.values})
    # Convertir el DataFrame a un diccionario
    output_dict = output.to_dict(orient='records')
    
    # Devolver el diccionario como JSON
    return JSONResponse(content=output_dict)

# 2)
@app.get('/userdata')
async def userdata(user_id:str):
    df = pd.read_parquet('D:/escritorio/STEAM/parquet/engineer.parquet')
    
    usuario = df[df['user_id'] == user_id ]
    #Cuenta el total de gastos de un usuario
    gasto = usuario['price'].sum()
    # Cuenta la cantidad de recomendaciones de un usuario
    recomendaciones = usuario[usuario['recommend'] == True ]['recommend'].count()
    #Calcula el porcentaje de recomendaciones
    porcentaje_recomen = f"{recomendaciones / len(usuario)*100}%" 
    cant_items = usuario['items_count'].sum()
    
    output = {'Dinero gastado': round(gasto,2), 'Cantidad de items': cant_items, 'Porcentaje de recomendación': porcentaje_recomen}  
     
    return output

# 3)
@app.get('/userforgenre')
async def userForGenre(genero:str):
    df = pd.read_parquet('D:/escritorio/STEAM/parquet/engineer.parquet')
    
    genre_game = df[df['genres']== genero.capitalize()]
    # Dado el genero, agrupa a los usarios por cantidad de horas jugadas
    horas_jugadas = genre_game.groupby(['user_id'])['playtime_forever'].sum()
    #Almacena el usuario con mas horas jugadas
    usuario_mas_horas = horas_jugadas.idxmax()
    # Filtrar  por el usuario con más horas jugadas
    user_masHoras = genre_game[genre_game['user_id'] == usuario_mas_horas]
    #Agrupa las horas jugadas por año para el usuario con más horas jugadas
    horas_por_anio = user_masHoras.groupby('release_year')['playtime_forever'].sum() / 60
    horas_por_anio = horas_por_anio.round(1)  
    

    output = {'Genero': genero, 'Usuario': usuario_mas_horas, 'Horas jugadas por año':[horas_por_anio.to_dict()]}
                 
    return output

# 4)
@app.get('/developer_reviews')
async def developer_reviews_analysis( desarrolladora : str ):
    df = pd.read_parquet('D:/escritorio/STEAM/parquet/engineer.parquet')
    
    df_dev = df[df['developer'] == desarrolladora.capitalize()]
    
    # Cuenta los sentimientos positivos y negativos
    positivos = (df_dev['sentiment_analysis'] == 2 ).sum()
    negativos = (df_dev['sentiment_analysis'] == 0 ).sum()
    
    output = {desarrolladora: f"[Positivos: {positivos}, Negativos: {negativos}]" }
    
    return output


# Modelo Machine Learning
@app.get('/recomendacion_juego')
async def recomendacion_juego(id:int):
    df_modelo = pd.read_parquet('D:/escritorio/STEAM/parquet/modelo.parquet')
    # Crea un df con los datos de el id pasado como parametro
    juego = df_modelo[df_modelo['item_id'] == id]
    # En caso que el ID no este en el df, muestra el siguiente mensaje:
    if id not in juego.values:
        return f"El juego con ID {id} no se encuentra."

    # En caso de estar. Obtiene el índice del juego en el DataFrame.
    idx = juego.index
    idx = idx[0]

    # Tamaño de la muestra
    muestra = 2000
    # Ajustamos la semilla aleatoria
    df_sample = df_modelo.sample(n=muestra, random_state=42)
    
    # Calculamos la similitud de contenido solo para el juego dado y la muestra
    sim_scores = cosine_similarity([df_modelo.iloc[idx, 3:]], df_sample.iloc[:, 3:])
    # Obtenemos las puntuaciones de similitud del juego dado con otros juegos
    sim_scores = sim_scores[0]

    # Ordenamos los juegos por similitud en orden descendente
    similar_games = [(i, sim_scores[i]) for i in range(len(sim_scores)) if i != idx]
    similar_games = sorted(similar_games, key=lambda x: x[1], reverse=True)

    # Obtenemos los 5 juegos más similares
    similar_game_indices = [i[0] for i in similar_games[:5]]
    # Listamos los juegos similares (solo nombres)
    similar_game_names = df_sample['app_name'].iloc[similar_game_indices].tolist()

    return {"Juegos similares": similar_game_names}