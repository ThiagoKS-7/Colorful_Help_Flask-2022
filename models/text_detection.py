import pytesseract
from pytesseract import Output
import cv2
import numpy as np
from PIL import ImageFont, Image,ImageDraw


fonte = 'Fontes/calibri.ttf'

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

def config_input(img, lang = 'por', dict = Output.DICT):
    return pytesseract.image_to_data(img, lang=lang, output_type=dict)


def bounding_box(input, img,i, cor=(0, 255, 0), tam_fonte=2):
    x = input['left'][i]
    y = input['top'][i]
    w = input['width'][i]
    h = input['height'][i]

    cv2.rectangle(img, (x, y), (x + w, y + h), cor, tam_fonte)
    # proto_ROI = img[y:y+h, x:x+w]

    return x, y, img


def write_text(texto,x,y,img,font,tamanho_texto=19):
    fonte = ImageFont.truetype(font,tamanho_texto)
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((x,y-tamanho_texto), texto, font = fonte)
    img = np.array(img_pil)
    return img


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
        cv2.imwrite(f'../assets/Images/Predictions/predicted{n}.jpg', img_copia)  # formato BGR
        return textos, f'predicted{n}.jpg'
    else:
        cv2.imwrite(f'../assets/Images/Predictions/thresholded{n}.jpg', img_copia) # binária
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


def bin_thresh(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    val, bin = cv2.threshold(gray, 94,255, cv2.THRESH_BINARY)
    cv2.imwrite('../assets/Images/Thresholds/bin_thresh.jpg', adap)
    return bin


def find_text():
    if len(pytesseract.image_to_string(bin)) > 0:
        return predict_text(bin_input, 40, bin, '_bin')

    if len(pytesseract.image_to_string(otsu)) > 0:
        return predict_text(otsu_input, 40, otsu, '_otsu')

    if len(pytesseract.image_to_string(adap)) > 0:
        return predict_text(adap_input, 40, adap, '_adap')

    elif len(pytesseract.image_to_string(rgb)) > 0:
        return predict_text(input,40, rgb, '_rgb')

    else:
        return "Nada encontrado"


def build_phrase(frase):
    for i in textos:
        if len(frase) <= 1:
            frase = i
        else:
            frase = frase + " " + i
    response = {
        'frase': frase,
        'img': img
    }
    return response


def config_inputs(rgb,otsu, adap,bin):
    input = config_input(rgb)
    otsu_input = config_input(otsu)
    adap_input = config_input(adap)
    bin_input = config_input(bin)
    return input,otsu_input, adap_input,bin_input


if __name__ == '__main__':
    img = cv2.imread("../assets/Images/data.jpg") #INPUT
    frase = ''
    #=====_____CONFIG IMAGES____======= #
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    otsu = otsu_thresh(img)
    adap = adap_thresh(img)
    bin = bin_thresh(img)
    # /=====_____CONFIG IMAGES____======= #

    # =====_____CONFIG INPUTS____======= #
    input, otsu_input, adap_input, bin_input = config_inputs(rgb,otsu,adap,bin)
    textos, img = find_text()
    # /=====_____CONFIG INPUTS____======= #
    response = build_phrase(frase)
    print(response)

