Detecção de buracos usando um modelo Mobilenet-SSD treinado com um custom dataset na biblioteca TensorFlow

<br>
<img src="resultados/mobiledetect.gif">
<img src="resultados/pothole4.jpg" width=500 height=300>
<br>

Codigo de Predição baseado em:
<br>
https://github.com/EdjeElectronics/TensorFlow-Lite-Object-Detection-on-Android-and-Raspberry-Pi/tree/master

https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/examples/python/label_image.py

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
caso queira alterar a quantidade de detecções, adicione "--threshold= 0.5"
<br>
exemplo para 40%>
```
python TFLite-detection-image.py --modeldir=custom_model_lite --imagedir=test_images --threshold= 0.4

```
<img src="resultados/maps2.png" width=500 height=300>
<img src="resultados/pothole7.jpg" width=500 height=300>
<img src="resultados/pothole8.jpg" width=500 height=300>

comparação de duas imagens. Uma com 50% e outra com 30%
<br>
detecção a partir de 50%
<br>
<img src="resultados/pothole.jpg" width=500 height=300>
<br>
detecção a partir de 30%
<br>
<img src="resultados/30percent.jpg" width=500 height=300>


