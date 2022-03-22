import pytesseract
from pytesseract import Output
import cv2
import numpy as np
from PIL import ImageFont, Image,ImageDraw


fonte = 'Fontes/calibri.ttf'

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

class AiOcr:
    def __init__(self, name, img, frase = ''):
        self.name = name
        self.img = img
        self.frase = frase



    def predict(self):
        img = cv2.imread(self.img)  # INPUT
        # =====_____CONFIG THRESHOLDS____======= #
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        otsu = otsu_thresh(img)
        adap = adap_thresh(img)
        bin = bin_thresh(img, adap)
        threshs = {
            'rgb': rgb,
            'otsu': otsu,
            'adap': adap,
            'bin':bin
        }
        # /=====_____CONFIG THRESHOLDS____======= #
        # =====_____CONFIG INPUTS____======= #
        input, otsu_input, adap_input, bin_input = config_inputs(threshs)
        inputs = {
            'input': input,
            'otsu_input': otsu_input,
            'adap_input': adap_input,
            'bin_input': bin_input
        }
        textos, img = find_text(threshs,inputs)
        # /=====_____CONFIG INPUTS____======= #
        response = build_phrase(self.frase, textos, img)
        return response

# CONFIGURA O INPUT PRO MÉTODO DE PREDIÇÃO
def config_input(img, lang = 'por', dict = Output.DICT):
    return pytesseract.image_to_data(img, lang=lang, output_type=dict)

# MONTA O CONTORNO  DA IMAGEM
def bounding_box(input, img,i, cor=(0, 255, 0), tam_fonte=2):
    x = input['left'][i]
    y = input['top'][i]
    w = input['width'][i]
    h = input['height'][i]

    cv2.rectangle(img, (x, y), (x + w, y + h), cor, tam_fonte)
    # proto_ROI = img[y:y+h, x:x+w]

    return x, y, img

# FUNÇÃO QUE ESCREVE O TEXTO NA IMAGEM
def write_text(text,x,y,img,font,tamanho_texto=19):
    fonte = ImageFont.truetype(font,tamanho_texto)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x,y-tamanho_texto), text, font = fonte)
    img = np.array(img_pil)
    return img

# CRIA O TEXTO DA RESPONSE E UMA NOVA IMAGEM COM O TEXTO PREDITO DE LABEL
def predict_text(resultado, min_conf, img,n):
    img_copia = img.copy()
    textos = []
    for i in range(0, len(resultado['text'])):
        confianca = float(resultado['conf'][i])
        if int(confianca) > min_conf:
            texto = resultado['text'][i]
            if not texto.isspace() and len(texto) > 0:
                x, y, img_copia = bounding_box(resultado, img_copia, i)
                textos.append(texto)
                texto = resultado['text'][i] + ' - ' + str(int(float(resultado['conf'][i]))) + '%'
                img_copia = write_text(texto, x, y, img_copia, fonte)
    if(len(img.shape) == 3):
        cv2.imwrite(f'C:/Users/W10/fetch_data/Images/predicted{n}.jpg', img_copia)  # formato BGR
        return textos, f'predicted{n}.jpg'
    else:
        cv2.imwrite(f'C:/Users/W10/fetch_data/Images/thresholded{n}.jpg', img_copia) # binária
        return textos, f'thresholded{n}.jpg'


def otsu_thresh(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    val, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print(f'[INFO] thresh escolhido: {val}')
    cv2.imwrite('../assets/Images/Thresholds/otsu_thresh.jpg', otsu)
    return otsu

def adap_thresh(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    adap = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 9)
    cv2.imwrite('../assets/Images/Thresholds/adap_gauss_thresh.jpg', adap)
    return adap

def bin_thresh(img,adap):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    val, bin = cv2.threshold(gray, 94,255, cv2.THRESH_BINARY)
    cv2.imwrite('../assets/Images/Thresholds/bin_thresh.jpg', adap)
    return bin

# CHECA QUAL DOS THRESHOLDS FOI ESCOLHIDO, PRA MANDAR PRA PREDIÇÃO
def find_text(threshs, inputs):
    if len(pytesseract.image_to_string(threshs['bin'])) > 0:
        return predict_text(inputs['bin_input'], 40, threshs['bin'], '_bin')

    if len(pytesseract.image_to_string(threshs['otsu'])) > 0:
        return predict_text(inputs['otsu_input'], 40, threshs['otsu'], '_otsu')

    if len(pytesseract.image_to_string(threshs['adap'])) > 0:
        return predict_text(inputs['adap_input'], 40, threshs['adap'], '_adap')

    elif len(pytesseract.image_to_string(threshs['rgb'])) > 0:
        return predict_text(inputs['input'],40, threshs['rgb'], '_rgb')

    else:
        return "Nada encontrado"

# MONTA O BODY DA RESPONSE
def build_phrase(frase, textos, img):
    for i in textos:
        if len(frase) <= 1:
            frase = i
        else:
            frase = frase + " " + i
    response = {
        'frase': frase,
        'img': img,
        'status':200
    }
    return response


def config_inputs(threshs):
    input = config_input(threshs['rgb'])
    otsu_input = config_input(threshs['otsu'])
    adap_input = config_input(threshs['adap'])
    bin_input = config_input(threshs['bin'])
    return input,otsu_input, adap_input,bin_input


if __name__ == '__main__':
    img = " ../app/storage/emulated/0/Android/data/com.example.foto/files/Pictures/scaled_3d1da06b-aafd-484b-93c7-49beb76df18b3436057806284215591.jpg"
    Ai_Ocr = AiOcr('Ai_Ocr',img)
    res = Ai_Ocr.predict()
    print(res)

