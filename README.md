# CudaSobel

El objetivo de este programa es aplicar el filtro de Sobel a images y videos. El programa puede mostrar los resultados obtenidos y generar un informe de el tiempo empleado comparando la implementación usando la CPU (un único hilo) y la GPU, usando la API de CUDA.

Para compilar usaremos `make`. Podemos limpiar el proyecto con `make clean`.

Ejemplos de uso:

- `./CONVOLUTIONer -h`: nos mostrará la ayuda del programa.
- `./CONVOLUTIONer --sv <VIDEO>`: procesará y mostrará dicho archivo.
- `./CONVOLUTIONer --si <IMG>`: procesará y mostrará dicho archivo.
- `./CONVOLUTIONer --pv <VIDEO>`: procesará la imagen y mostrará un informe con el tiempo empleado y FPS obtenidos por cada implementación.
- `./CONVOLUTIONer --pi <IMG>`: procesará la imagen y mostrará un informe con el tiempo empleado por cada implementación.
- `./CONVOLUTIONer -w N`: mostrará la salida de la webcam después de aplicar Sobel.


