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


Resultado de `deviceQuery` en mi ordenador:

```
DeviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "GeForce 920MX"
  CUDA Driver Version / Runtime Version          11.1 / 11.1
  CUDA Capability Major/Minor version number:    5.0
  Total amount of global memory:                 2004 MBytes (2101870592 bytes)
  ( 2) Multiprocessors, (128) CUDA Cores/MP:     256 CUDA Cores
  GPU Max Clock rate:                            993 MHz (0.99 GHz)
  Memory Clock rate:                             900 Mhz
  Memory Bus Width:                              64-bit
  L2 Cache Size:                                 1048576 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
  Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        65536 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 1 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            No
  Supports Cooperative Kernel Launch:            No
  Supports MultiDevice Co-op Kernel Launch:      No
  Device PCI Domain ID / Bus ID / location ID:   0 / 3 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 11.1, CUDA Runtime Version = 11.1, NumDevs = 1
Result = PASS
```


