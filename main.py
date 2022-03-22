from datetime import datetime
import os
from flask import Flask, request, jsonify
from flask_basicauth import BasicAuth
from textblob import TextBlob
from models.yolov3_img import AiTwo
from models.classification import AiOne
from models.text_detection1 import AiOcr
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('data/train.csv')
mapa = {
    "LotArea" : "tamanho",
    "YrSold" : "ano",
    "GarageCars" : "garagem",
    "SalePrice": "preço"
}
dataset.rename(columns = mapa, inplace = True)
dataset = dataset[['tamanho', 'ano', 'garagem', 'preço']]
colunas = dataset[['tamanho', 'ano', 'garagem']]
X = dataset.drop('preço', axis=1)
y = dataset['preço']
X_train,X_test,y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
modelo = LinearRegression()
modelo.fit(X_train,y_train)









#instanciar classse
app = Flask(__name__)
app.config['BASIC_AUTH_USERNAME'] = 'root'
app.config['BASIC_AUTH_PASSWORD'] = 'root'

basic_auth = BasicAuth(app)

#definir rotas da API
@app.route('/')
@basic_auth.required
def home():
    return "Teste de API."

#pra passar a imagem dps, tem q por dentro dessa tag <frase>
@app.route('/sentimento/<frase>')
def sentimento(frase):
    tb = TextBlob(frase)
    tb_en = tb.translate(to="en")
    polaridade = tb_en.sentiment.polarity
    if (polaridade > 0):
        sentimento = 'algo positivo'
    elif (polaridade < 0):
        sentimento = 'algo negativo'
    else:
        sentimento = 'neutro'
    return "polaridade: {}% - {}".format(polaridade * 100, sentimento)


@app.route('/cotacao', methods=['POST'])
def cotacao ():
    dados = request.get_json()
    dados_input = [dados[col] for col in colunas]
    preco = modelo.predict([dados_input])
    return jsonify(preco=round(preco[0],2))

@app.route('/yolo')
def classify_yolo ():
    class_names = 'coco.names'
    weights = 'yolov3.weights'
    cfg = 'yolov3.cfg'

    Ai_2 = AiTwo('Ai_2', weights, class_names, cfg)
    Ai_2.predict()
    return jsonify(status='done!')

@app.route('/text-detect', methods=['POST'])
def text_detection():
    body = request.get_json()
    img = body['img']
    Ai_Ocr = AiOcr('Ai_Ocr',img)
    res = Ai_Ocr.predict()
    return jsonify(res)

@app.route('/classification')
def classify_clothes ():
    model = 'AUG_K_TUNED-CNN2.model'
    Ai_1 = AiOne('Ai_1', model)
    Ai_1.predict()
    return jsonify(status='done!')


@app.route('/test/pred', methods=['POST'])
def predict():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            return "someting went wrong 1"

        user_file = request.files['file']
        temp = request.files['file']
        if user_file.filename == '':
            return "file name not found ..."

        else:
            path = os.path.join(os.getcwd() + '\\modules\\static\\' + user_file.filename)
            user_file.save(path)


            return jsonify({
                "status": "success",
                "path": path,
                "upload_time": datetime.now()
            })


app.run(debug=True)