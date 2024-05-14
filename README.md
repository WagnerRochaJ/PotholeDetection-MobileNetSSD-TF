Detecção de buracos usando um modelo Mobilenet-SSD treinado com um custom dataset na biblioteca TensorFlow

Dependencias Necessarias>

*tensorflow

*opencv

*numpy

*cvzone

execucão com imagem
```
python TFLite-detection-image.py --modeldir=custom_model_lite --image=test_images/pothole.jpg
```
execução com um diretorio de imagens
```
python TFLite-detection-image.py --modeldir=custom_model_lite --imagedir=test_images
```
execução com video
```
python TFLite-detection-video.py --modeldir=custom_model_lite --video=test_video/video1.mp4 
```
