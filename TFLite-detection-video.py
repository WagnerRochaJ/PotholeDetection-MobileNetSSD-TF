# pacotes para importar
import os
import argparse
import cv2
import numpy as np
import cvzone
import sys
import importlib.util
import datetime


# Definindo as entradas parse
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--video', help='Name of the video file',
                    default='test.mp4')
args = parser.parse_args()

# Entradas do parse
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
VIDEO_NAME = args.video
min_conf_threshold = float(args.threshold)

# importando tensorflow lib
# se tflite_runtime estiver instalado, importar interpreter from tflite_runtime, se nao import from regular tensorflow
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

# retorna a pasta atual
CWD_PATH = os.getcwd()

# caminho para o video
VIDEO_PATH = os.path.join(CWD_PATH,VIDEO_NAME)

# caminho para o arquivo .tflite, que contem o modelo usado para a detecçao 
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# caminho para o label map
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# carrega o label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# carrega o modelo do Tensorflow Lite.
interpreter = Interpreter(model_path=PATH_TO_CKPT)
#aloca os tensores necessários para o modelo. Os tensores são estruturas de dados que representam os dados de entrada e saída do modelo.
interpreter.allocate_tensors()

# obtêm os detalhes dos tensores de entrada e saída do modelo. Os detalhes incluem informações como formato, tipo de dados e índices.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

#Essa linha verifica se o modelo é quantizado ou não. Se for um modelo quantizado, os valores dos pixels da imagem de entrada precisam ser normalizados.
floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Verifica o nome da camada de saída para determinar se este modelo foi criado com TF2 ou TF1,
# porque as saídas são ordenadas de forma diferente para os modelos TF2 e TF1
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# abre o arquivo de video
video = cv2.VideoCapture(VIDEO_PATH)
imW = video.get(cv2.CAP_PROP_FRAME_WIDTH)
imH = video.get(cv2.CAP_PROP_FRAME_HEIGHT)

while(video.isOpened()):
    
    start = datetime.datetime.now()

    # pega o frame e redimenciona paro o shape [1xHxWx3]
    ret, frame = video.read()
    if not ret:
      print('Reached the end of the video!')
      break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (width, height))
    input_data = np.expand_dims(frame_resized, axis=0)

    # verifica se o modelo é um “floating model”(modelo nao quantizado) ou nao e depois normaliza os valores dos pixels da imagem de entrada
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # definindo os valores dos tensores de entrada do modelo
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()#executa a inferência do modelo

    # Retrieve detection results
    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] #  coordenadas das boxes do objeto
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # index com as classes
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # confidence do objeto objects

    # loop pelo os objetos e desenha as boxes caso seja maior que o minimo de threshold
    for i in range(len(scores)):
        if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

            # pegar as coordenadas das bounding box e desenhar
            # O interpreter pode retornar coordenadas que estão fora das dimensões da imagem, precisa forçá-las a estarem dentro da imagem usando max() e min() para forçar numero inteiro
            ymin = int(max(1,(boxes[i][0] * imH)))
            xmin = int(max(1,(boxes[i][1] * imW)))
            ymax = int(min(imH,(boxes[i][2] * imH)))
            xmax = int(min(imW,(boxes[i][3] * imW)))
            
            #retangulo do objeto
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (8, 75, 252), 4)

            # recebe a label
            object_name = labels[int(classes[i])] # nome do objeto descrito no label
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Exemplo: 'person: 72%'
            #desenha o texto com a label
            cvzone.putTextRect(frame,label, (xmin, ymin),scale =0.8,thickness=1,colorT=(255,255,255),colorR=(0,0,0),font= cv2.FONT_HERSHEY_PLAIN,offset=5)
            
    end = datetime.datetime.now()
    total = (end - start).total_seconds()
    print(f"tempo para 1 frame: {total * 1000:.0f} milisegundos")
    fps = f"FPS: {1/ total:.2f}"
    cv2.putText(frame,fps,(50,50),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(5,100,255),4)
    # a box foi desenhada na imagem, imshow para mostrar a imagem
    cv2.imshow('Video', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) == ord('q'):
        break

# Clean up
video.release()
cv2.destroyAllWindows()

#python3 TFLite-detection-video.py --modeldir=custom_model_lite --video=test_video/video1.mp4 