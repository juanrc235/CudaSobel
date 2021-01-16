# Procesamiento de Im ́agenes

El objetivo de este programa es aplicar el filtro de Sobel a images y videos. El programa puede mostrar los resultados obtenidos y generar un informe de el tiempo empleado comparando la implementación usando la CPU (un único hilo) y la GPU, usando la API de CUDA.

Para compilar usaremos `make`. Podemos limpiar el proyecto con `make clean`.

Ejemplos de uso:

- `./CONVOLUTIONer -h`: nos mostrará la ayuda del programa.
- `./CONVOLUTIONer --sv <VIDEO>`: procesará y mostrará dicho archivo.
- `./CONVOLUTIONer --si_gpu <IMG>`: procesará y mostrará dicho archivo usando la GPU.
- `./CONVOLUTIONer --si_cpu <IMG>`: procesará y mostrará dicho archivo usando la CPU.
- `./CONVOLUTIONer --pv <VIDEO>`: procesará la imagen y mostrará un informe con el tiempo empleado y FPS obtenidos por cada implementación.
- `./CONVOLUTIONer --pi <IMG>`: procesará la imagen y mostrará un informe con el tiempo empleado por cada implementación.
- `./CONVOLUTIONer -w [0|1|2]`: mostrará la salida de la webcam después de aplicar Sobel [CPU|GPU|SIN_FILTRO].


