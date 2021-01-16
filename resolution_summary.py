import os
import matplotlib.pyplot as plt

def execute (command, iter):
    cpu_times = []
    gpu_times = []
    ntest = []
    for i in range(iter):
        output = os.popen(command).read()
        output = output.split()
        cpu_times.append(float(output[13].replace('ms', '')))
        gpu_times.append(float(output[16].replace('ms', '')))
        ntest.append(i)
    
    return { 'cpu': cpu_times, 'gpu': gpu_times}

iter = 5
base_command = './CONVOLUTIONer --pi resources/{}'
image_list = ['imgSD.jpg', 'imgHD.jpg', 'imgFHD.jpg', 'img4K.jpg', 'img8K.jpg']

ntest = list(range(0, len(image_list)))
results = {}
cpu_means = []
gpu_means = []
for img in image_list:
    print(f'Image: {img}')
    r = execute(base_command.format(img), iter)
    results[img] = r
    cpu_means.append(sum(r['cpu'])/iter)
    gpu_means.append(sum(r['gpu'])/iter)

p1 = plt.plot(ntest, gpu_means, 0.35)
p2 = plt.plot(ntest, cpu_means, 0.35)
plt.ylabel('Tiempo (ms)')
plt.xlabel('Imagen')
plt.xticks(ntest, image_list)
plt.legend((p1[0], p2[0]), ('Tiempo GPU', 'Tiempo CPU'))
plt.show()
