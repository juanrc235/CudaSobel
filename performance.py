import os
import sys
import matplotlib.pyplot as plt

try:
    iter = int(sys.argv[2])
except ValueError:
    print('Valor del número de repeticiones incorrecto')
    exit(0)

command = sys.argv[1]

output = os.popen(command).read()

if 'Error' in output:
   print('El comando especificado genera un error: ', output, end='')
   exit(0) 

print(f'Comando a ejecutar: {command}')
print(f'Número de repeticiones: {iter}')

cpu_times = []
gpu_times = []
ntest = []
for i in range(iter):
    output = os.popen(command).read()
    output = output.split()
    cpu_times.append(float(output[13].replace('ms', '')))
    gpu_times.append(float(output[16].replace('ms', '')))
    ntest.append(i)

print('Tiempo medio:')
print(f' - GPU: {round(sum(gpu_times)/(i+1), 2)} ms')
print(f' - CPU: {round(sum(cpu_times)/(i+1), 2)} ms')

fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)

media = [sum(gpu_times)/(i+1)]*len(ntest)
ax1.plot(ntest, gpu_times, label='Datos')
ax1.plot(ntest, media, linestyle='--', label='Media')
ax1.set_title('Tiempos en GPU')
ax1.set_ylabel('Tiempo (ms)')
ax1.set_xlabel('Número de test')

media = [sum(cpu_times)/(i+1)]*len(ntest)
ax2.plot(ntest, cpu_times, label='Datos')
ax2.plot(ntest, media, linestyle='--', label='Media')
ax2.set_title('Tiempos en CPU')
ax2.set_ylabel('Tiempo (ms)')
ax2.set_xlabel('Número de test')


ax1.legend()
plt.show()