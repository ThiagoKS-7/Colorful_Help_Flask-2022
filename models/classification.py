# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:59:55 2020

@author: Thiago Kasper de Souza
"""


def serial_send(serial, ctg, color):
    # FUNÇÕES PARA RODAR EM CONJUNTO COM ARDUINO
    sum = (str(ctg) + color)
    print("Código concatenado: ", sum)
    # codifica_dados(int(sum))
    # serial.write(str(codifica_dados(int(sum))).encode())
    serial.close()
    return True
# def color_distinct(filepath, engine):
#     #color = distingue_cor(filepath)
#     engine.say(f"e sua cor é: {color}")
#     engine.runAndWait()
#     engine.stop()

class AiOne:
    def __init__(self, name, model):
        self.name = name
        self._model = model

    def predict(self):
        import cv2
        import tensorflow as tf
        # from ..lib.color.color_analysis import color_distinct
        from matplotlib import pyplot as plt
        import numpy as np
        import imageio
        import pyttsx3  # TTS = Text-To-Speech

        engine = pyttsx3.init()
        engine.setProperty('voice','pt')
        class_names = ["Bota","Camisa", "Camiseta","Tenis"]

        def prepare(filepath):  # PREPARAR IMAGEM
            tamanho= 40
            image = cv2.imread(filepath)
            image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            resized2 = cv2.resize(image, (tamanho, tamanho))
            return resized2.reshape(1,tamanho,tamanho,-1)

        model = tf.keras.models.load_model('data/classification_models/AUG_K_TUNED-CNN2.model')
        # print("Testes de predição:")
        # print()

        capture = cv2.VideoCapture(0)
        cont = 0

        while cont != 25:
            ret, frame = capture.read()
            cont += 1

        cv2.imwrite("assets/new_img2.jpg", frame)
        capture.release()
        cv2.destroyAllWindows()
        roupa = 'assets/new_img2.jpg'
        prediction = model.predict([prepare(roupa)])
        ctg = np.argmax(prediction.flatten())
        print(f'Fig #1: {class_names[ctg]}')

        #CONDIÇÃO PARA FALAR O ARTIGO CERTO: SE CTG
        #VALER ATÉ 3, ARTIGO É FEMININO, SE FOR MAIOR, É MASCULINO
        if ctg < 3  and ctg > 0:
            engine.say("isso é uma" + class_names[ctg],)
            engine.runAndWait()
        elif ctg == 3:
            engine.say("isso é um tênis")
            engine.runAndWait()
        print("Posição da predição encontrada:", ctg)

        # TODO: CORREÇÃO DAS CORES DA IMAGEM, PRA APARECER NO PYPLOT
        # TODO: BOUNDING BOX ESTÁTICA CENTRALIZADA (NÃO LOCALIZA NOS CANTOS DA IMG)

        img = imageio.imread(roupa)
        img = cv2.resize(img,(200,200))
        obj_found = cv2.rectangle(img, (45, 30),(170, 190),(0, 255, 0), 1)
        img_predicted = cv2.putText(obj_found, "Label: {}".format(class_names[ctg]),(12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 254, 0), 2)
        #plt.imshow(img_predicted)
        #plt.show()
        #print()



        #DELETA TODAS AS VARIAVEIS PRA POUPAR ESPAÇO E OTIMIZAR O PROGRAMA
        #del class_names
        #del roupa
        del prediction
        del ctg, img, img_predicted
        del engine
        #del d
        #del cor
        #del soma
        #del captura, ret,frame,cont

if __name__ == "__main__":
    model = 'AUG_K_TUNED-CNN2.model'
    Ai_1 = AiOne('Ai_1', model)
    Ai_1.predict()