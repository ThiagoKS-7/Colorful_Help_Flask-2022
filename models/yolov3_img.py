"""
Course:  Training YOLO v3 for Objects Detection with Custom Data
Edits: Thiago Kasper de Souza

Section-2
Objects Detection on Image with YOLO v3 and OpenCV
File: yolo-3-image.py
"""


# Detecting Objects on Image with OpenCV deep learning library
#
# Algorithm:
# Reading RGB image --> Getting Blob --> Loading YOLO v3 Network -->
# --> Implementing Forward Pass --> Getting Bounding Boxes -->
# --> Non-maximum Suppression --> Drawing Bounding Boxes with Labels
def read_input():
    import cv2
    print(f'Versão do OpenCV: {cv2.__version__}')
    captura = cv2.VideoCapture(0)
    cont = 0
    while True:
        ret, frame = captura.read()
        cont += 1
        if cont >= 25:
            break

    cv2.imwrite('assets/old_img.jpg', frame)
    captura.release()
    cv2.destroyAllWindows()  # img = "C:/Users/W10/OneDrive - liberato.com.br/Images/models.jpg"
    image_BGR = cv2.imread('assets/old_img.jpg')
    # Getting spatial dimension of input image

    blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    return [blob, image_BGR]


def article_correction(engine, pred, labels, num, val):
    print('entrou aqui')
    female = [0, 1, 3, 9, 11, 18, 19, 22, 23, 24, 26, 27, 28, 39, 40, 43, 44, 45, 46, 47, 49, 51,
              53, 56, 57, 59, 60, 61, 70, 71, 76, 79]

    male = [2, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 20, 21, 25, 29,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 41, 42, 48, 50, 52, 54, 55, 58,
            62, 63, 64, 65, 66, 67, 68, 69, 72, 73, 74, 75, 77, 78, 80]

    article = ''

    for item in male:
        print('entrou no for')
        if (pred[num - val] == labels[female[item]] or pred[num - val] == 'pessoa'):
            article = 'uma'
            break
        else:
            article = 'um'
            break

    engine.say(f'isso é {article}  {pred[num - val]}')
    engine.runAndWait()
    engine.stop()


class AiTwo:
    def __init__(self, name, weights, class_names, cfg):
        self.name = name
        self._weights = weights
        self._class_names = class_names
        self._cfg = cfg

    @staticmethod
    def foward_pass(network, blob, layers_names_output):
        network.setInput(blob)  # setting blob as input to the network
        output_from_network = network.forward(layers_names_output)
        # print('Objects Detection took {:.5f} seconds'.format(end - start))
        return output_from_network

    @staticmethod
    def non_maximum_supression(bounding_boxes, confidences, probability_minimum, threshold):
        import cv2
        # It is needed to make sure that data type of the boxes is 'int'
        # and data type of the confidences is 'float'
        # https://github.com/opencv/opencv/issues/12789
        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                                   probability_minimum, threshold)
        return results

    @staticmethod
    def speech(engine, pred, labels):
        import speech_recognition as sr
        r = sr.Recognizer()

        mic = sr.Microphone()

        with mic as fonte:
            r.adjust_for_ambient_noise(fonte)
            engine.say("O que devo fazer em seguida?")
            engine.runAndWait()
            audio = r.listen(fonte)
            print("um momento...")
            try:
                texto = r.recognize_google(audio, language="pt-BR")
                print("Você disse: {}".format(texto))

                # FRASES QUE ELA ENTENDE:
                if len(texto) == 16:

                    if texto == "defina objeto um":
                        num = 1
                        article_correction(engine, pred, labels, num, 1)

                elif len(texto) == 15:
                    corte = texto[14]
                    num = int(corte)
                    print(num)
                    print(pred[num - 1])
                    if texto == f"defina objeto {corte}" or f"defina objetos {corte}":
                        article_correction(engine, pred, labels, num, 1)
                engine.stop()

            except:
                engine.say("Desculpe, mas não entendi. Tente de novo")
            finally:
                engine.stop()

    def predict(self):
        # Importing needed libraries
        import numpy as np
        import cv2
        # import time
        import imageio
        from matplotlib import pyplot as plt
        import pyttsx3

        pred = []
        engine = pyttsx3.init()
        engine.setProperty('voice', 'pt')

        input = read_input()
        blob = input[0]
        image_BGR = input[1]
        h, w = image_BGR.shape[:2]  # Slicing from tuple only first two elements

        """
        Start of:
        Loading YOLO v3 network
        """
        with open(f'data/yolo-coco-data/{self._class_names}') as f:
            labels = [line.strip() for line in f]

        # with the help of 'dnn' library from OpenCV
        # Pay attention! If you're using Windows, yours paths might look like:
        # r'yolo-coco-data\yolov3.cfg'
        # r'yolo-coco-data\yolov3.weights'
        # or:
        # 'yolo-coco-data\\yolov3.cfg'
        # 'yolo-coco-data\\yolov3.weights'

        network = cv2.dnn.readNetFromDarknet(f'data/yolo-coco-data/{self._cfg}',
                                             f'data/yolo-coco-data/{self._weights}')
        layers_names_all = network.getLayerNames()

        layers_names_output = ['yolo_82', 'yolo_94', 'yolo_106']
        # print(layers_names_output)  # ['yolo_82', 'yolo_94', 'yolo_106']

        probability_minimum = 0.5
        threshold = 0.3
        colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

        """
        End of:
        Loading YOLO v3 network
        """

        output_from_network = AiTwo.foward_pass(network, blob, layers_names_output)

        """
        Start of:
        Getting bounding boxes
        """
        bounding_boxes = []
        confidences = []
        class_numbers = []

        for result in output_from_network:
            for detected_objects in result:
                scores = detected_objects[5:]
                class_current = np.argmax(scores)
                confidence_current = scores[class_current]

                if confidence_current > probability_minimum:
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])

                    # Now, from YOLO data format, we can get top left corner coordinates
                    # that are x_min and y_min
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                    # Adding results into prepared lists
                    bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)

        """
        End of:
        Getting bounding boxes
        """

        results = AiTwo.non_maximum_supression(bounding_boxes, confidences, probability_minimum, threshold)

        """
        Start of:
        Drawing bounding boxes and labels
        """
        # Defining counter for detected objects
        counter = 1

        # Checking if there is at least one detected object after non-maximum suppression
        if len(results) > 0:
            for i in results.flatten():
                print('Objeto {0}: {1}'.format(counter, labels[int(class_numbers[i])]))
                pred.append(labels[int(class_numbers[i])])
                counter += 1

                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

                colour_box_current = colours[class_numbers[i]].tolist()

                cv2.rectangle(image_BGR, (x_min, y_min),
                              (x_min + box_width, y_min + box_height),
                              colour_box_current, 2)

                text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                       confidences[i])

                cv2.putText(image_BGR, text_box_current, (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)

        cv2.imwrite("assets/new_img.jpg", image_BGR)
        image_RGB = imageio.imread("assets/new_img.jpg")

        # mostrar imagem no pyplot
        # plt.imshow(image_RGB)
        # plt.show()

        # Comparing how many objects where before non-maximum suppression
        # and left after
        print()
        print('Total de objetos detectados:', len(bounding_boxes))
        print('Objetos deixados após non-maximum suppression:', counter - 1)
        if (counter - 1) >= 2:
            engine.say("Eu encontrei" + str(counter - 1) + "objetos.")
            engine.runAndWait()
            engine.stop()
        else:
            pass
        """
        End of:
        Drawing bounding boxes and labels
        """

        """
        Start of:
        Speech Recognition
        """
        if (counter - 1) == 1:
            num = 0
            article_correction(engine, pred, labels, num, 0)

        else:
            AiTwo.speech(engine, pred, labels)
        """
        End of:
        Speech Recognition
        """


"""
Some comments

With OpenCV function 'cv2.dnn.blobFromImage' we get 4-dimensional
so called 'blob' from input image after mean subtraction,
normalizing, and RB channels swapping. Resulted shape has:
- number of images
- number of channels
- width
- height
E.G.: blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
"""

if __name__ == '__main__':
    class_names = 'coco.names'
    weights = 'yolov3.weights'
    cfg = 'yolov3.cfg'

    Ai_2 = AiTwo('Ai_2', weights, class_names, cfg)
    Ai_2.predict()

