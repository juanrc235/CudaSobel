# CONVOLUTIONer

El objetivo de este programa es aplicar el filtro de Sobel a images y videos. El programa puede mostrar los resultados obtenidos y generar un informe de el tiempo empleado comparando la implementación usando la CPU (un único hilo) y la GPU, usando la API de CUDA.

Para compilar usaremos `make`. Podemos limpiar el proyecto con `make clean`.

Ejemplos de uso:

- `./CONVOLUTIONer -h`: nos mostrará la ayuda del programa.
- `./CONVOLUTIONer --sv <VIDEO>`: procesará y mostrará dicho archivo.
- `./CONVOLUTIONer --si <IMG>`: procesará y mostrará dicho archivo.
- `./CONVOLUTIONer --pv <VIDEO>`: procesará la imagen y mostrará un informe con el tiempo empleado y FPS obtenidos por cada implementación.
- `./CONVOLUTIONer --pi <IMG>`: procesará la imagen y mostrará un informe con el tiempo empleado por cada implementación.
- `./CONVOLUTIONer -w N`: mostrará la salida de la webcam después de aplicar Sobel.



## Implementación sin uso de memoria compartida

En este caso tenemos un 16x16 hilos por bloque y un número de bloques que depende de la imagen procesada. Para una imagen 4K (3840x2160 píxeles) tendremos 240x135 bloques con 16x16 hilos cada uno. Estos valores (de los hilos por bloque) no son al azar ya que son los que maximizan la utilización de GPU (se puede ver suando el *nvvp*).

A continuación se muestrán los resultados al procesar imagenes de diferente resolución (el tiempo se expresa en ms):

|            | Implementación |        |        | Speed-Up      |            |
|------------|----------------|--------|--------|---------------|------------|
|            |       CPU      | OpenCV |   GPU  | GPU vs OpenCV | GPU vs CPU |
|  Imagen SD |     22.651     |  2.246 |  1.249 |     1.79X     |   18.14X   |
|  Imagen HD |     62.304     |  4.457 |  3.683 |     1.21X     |   16.91X   |
| Imagen FHD |     140.341    |  9.570 |  7.717 |     1.24X     |   18.18X   |
|  Imagen 4K |     536.586    | 38.775 | 24.909 |     1.55X     |   21.54X   |



### Ejemplo de stdout procesando imagenes

```
➜  CudaSobel git:(main) ✗ ./CONVOLUTIONer --pi resources/imageSD.jpg 
Image: resources/imageSD.jpg
 - resolution: 640x480
 - channels: 3
 - type: 8UC3
[CPU] time: 22.651ms
[GPU] time: 1.24968ms
[OPENCV] time: 2.24651ms
➜  CudaSobel git:(main) ✗ ./CONVOLUTIONer --pi resources/imgHD.jpg 
Image: resources/imgHD.jpg
 - resolution: 1280x720
 - channels: 3
 - type: 8UC3
[CPU] time: 62.3047ms
[GPU] time: 3.6827ms
[OPENCV] time: 4.4574ms
➜  CudaSobel git:(main) ✗ ./CONVOLUTIONer --pi resources/imgFHD.jpg
Image: resources/imgFHD.jpg
 - resolution: 1920x1080
 - channels: 3
 - type: 8UC3
[CPU] time: 140.341ms
[GPU] time: 7.71746ms
[OPENCV] time: 9.57063ms
➜  CudaSobel git:(main) ✗ ./CONVOLUTIONer --pi resources/img4K.jpg 
Image: resources/img4K.jpg
 - resolution: 3840x2160
 - channels: 3
 - type: 8UC3
[CPU] time: 536.586ms
[GPU] time: 24.9097ms
[OPENCV] time: 38.7752ms
```

### Ejemplo de stdout procesando videos

```
➜  CudaSobel git:(main) ./CONVOLUTIONer --pv resources/videoSD.mp4 
Video: resources/videoSD.mp4
 - resolution: 640x360
 - fps: 30
 - nframes: 1742
 - duration: 58.0667 seconds 
[GPU] - time: 2.6196s
      - fps: 664.987
[CPU] - time: 27.8629s
      - fps: 62.5204
```