# pacotes para importar
import os
import argparse
import cv2
import numpy as np
import sys
import cvzone
import glob
import importlib.util
import datetime
import time



# Definindo as entradas parse
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',required=True)
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',default=0.5)
parser.add_argument('--image', help='Name of the single image to perform detection on. To run detection on multiple images, use --imagedir',default=None)
parser.add_argument('--imagedir', help='Name of the folder containing images to perform detection on. Folder must contain only images.',default=None)
parser.add_argument('--save_results', help='Save labeled images and annotation data to a results folder',action='store_true')

args = parser.parse_args()


# Entradas do parse
MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels

min_conf_threshold = float(args.threshold)

save_results = args.save_results 

IM_NAME = args.image
IM_DIR = args.imagedir

# para selecionar a imagem ou o diretorio contendo varias imagens
if (IM_NAME and IM_DIR):
    print('Error! Please only use the --image argument or the --imagedir argument, not both. Issue "python TFLite_detection_image.py -h" for help.')
    sys.exit()

# se não selecionar nenhuma da opcoes, ira por padrao na imagem pothole.jpg
if (not IM_NAME and not IM_DIR):
    IM_NAME = 'testImages/pothole.jpg'

# importando tensorflow lib
# se tflite_runtime estiver instalado, importar interpreter from tflite_runtime, se nao import from regular tensorflow
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
else:
    from tensorflow.lite.python.interpreter import Interpreter

# retorna a pasta atual
CWD_PATH = os.getcwd()

# juntar todas as imagens em uma lista
if IM_DIR:
    PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_DIR)
    images = glob.glob(PATH_TO_IMAGES + '/*.jpg') + glob.glob(PATH_TO_IMAGES + '/*.png') + glob.glob(PATH_TO_IMAGES + '/*.bmp')
    if save_results:
        RESULTS_DIR = IM_DIR + '_results'

elif IM_NAME:
    PATH_TO_IMAGES = os.path.join(CWD_PATH,IM_NAME)
    images = glob.glob(PATH_TO_IMAGES)
    if save_results:
        RESULTS_DIR = 'results'

# cria um diretorio com os resultados caso o save_results for True
if save_results:
    RESULTS_PATH = os.path.join(CWD_PATH,RESULTS_DIR)
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)

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

# caminha pela lista de imagens
for image_path in images:

    # carrega a imagem e redimenciona para o shape [1xHxWx3]
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    imH, imW, _ = image.shape 
    image_resized = cv2.resize(image_rgb, (width, height))
    input_data = np.expand_dims(image_resized, axis=0)

    # # verifica se o modelo é um “floating model”(modelo nao quantizado) ou nao e depois normaliza os valores dos pixels da imagem de entrada
    if floating_model:
        input_data = (np.float32(input_data) - input_mean) / input_std

    # definindo os valores dos tensores de entrada do modelo
    interpreter.set_tensor(input_details[0]['index'],input_data)
    interpreter.invoke()#executa a inferência do modelo

    boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # coordenadas das boxes do objeto
    classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]# index com as classes
    scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # confidence do objeto detectado

    detections = []

    start_inference = datetime.datetime.now()
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
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (8, 75, 252), 2)

            # recebe a label
            object_name = labels[int(classes[i])] # nome do objeto descrito no label
            label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Exemplo: 'person: 72%'
            #desenha o texto com a label
            cvzone.putTextRect(image,label, (xmin, ymin),scale =0.8,thickness=1,colorT=(255,255,255),colorR=(0,0,0),font= cv2.FONT_HERSHEY_PLAIN,offset=5)
            #carrega alguns valores em uma lista
            detections.append([object_name, (scores[i]), xmin, ymin, xmax, ymax])
            
    end_inference = datetime.datetime.now()
    #calculo do tempo de inferencia
    total = (end_inference - start_inference).total_seconds()
    print(f"tempo de inferencia = {total*1000:.2f} ms")   
    # a box foi desenhada na imagem, imshow para mostrar a imagem
    cv2.imshow('Imagem', image)
    # pressione qualquer tecla para passar as imagens, ou "q" para sair
    if cv2.waitKey(0) == ord('q'):
        break

    #salva a label da imagem para a pasta de results
    if save_results:

        # pegar o nome das imagens e caminhos
        image_fn = os.path.basename(image_path)
        image_savepath = os.path.join(CWD_PATH,RESULTS_DIR,image_fn)
        
        base_fn, ext = os.path.splitext(image_fn)
        txt_result_fn = base_fn +'.txt'
        txt_savepath = os.path.join(CWD_PATH,RESULTS_DIR,txt_result_fn)

        # salva a imagem
        cv2.imwrite(image_savepath, image)

        # escreve os resultados em um arquivo txt
        with open(txt_savepath,'w') as f:
            for detection in detections:
                f.write('%s %.2f %d %d %d %d\n' % (detection[0], detection[1], detection[2], detection[3], detection[4], detection[5]))

# Clean up
cv2.destroyAllWindows()

#exemplo de execucão
#python3 TFLite-detection-image.py --modeldir=custom_model_lite --image=test_images/pothole.jpg
