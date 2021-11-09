import matplotlib.pyplot as plt
import numpy as np

filename = '/home/azamhamidinekoo/Documents/DeepLearing/pix2pixHD/checkpoints/label2city_512p_singLablsOnly/loss_log.txt'

mynumbers = []
epoch = []
iters = []
G_GAN = []
G_GAN_Feat = []
G_VGG = []
D_real = []
D_fake = []
alls = []
with open(filename) as f:
    for line in f:
        if 'time' in line:
            splits = line.strip().split(',')
            epoch0, iter0 = splits[0].split(': '), splits[1].split(': ')
            epoch.append(list(map(float,epoch0[1])))
            iters.append(list(map(float,iter0[1])))
            
            
            x = [xx.split(' ') for xx in splits[2].split(': ')]
            G_GAN.append(list(map(float,x[2][0])))
            G_GAN_Feat.append(list(map(float,x[3][0])))
            G_VGG.append(list(map(float,x[4][0])))
            D_real.append(list(map(float,x[5][0])))
            D_fake.append(list(map(float,x[6][0])))
            
            '''G_GAN.append(float(x[2][0]))
            G_GAN_Feat.append(float(x[3][0]))
            G_VGG.append(float(x[4][0]))
            D_real.append(float(x[5][0]))
            D_fake.append(float(,x[6][0]))'''
            
            '''alls[0].append(float(epoch0[1]))
            alls[1].append(float(iter0[1]))
            alls[2].append(float(x[2][0]))
            alls[3].append(float(x[3][0]))
            alls[4].append(float(x[4][0]))
            alls[5].append(float(x[5][0]))
            alls[6].append(float(x[6][0]))'''

#print(np.shape(alls))
plt.figure()
plt.plot(epoch,G_GAN, epoch,G_GAN_Feat, epoch, G_VGG, epoch, D_real, epoch, D_fake)


epoch = list(map(float, x[2][0]))
G_GAN = [float(s) for s in G_GAN]
plt.plot(np.array(epoch),np.array(G_GAN))

plt.show()


import matplotlib.pyplot as plt
with open(filename) as m:
    for line in m:
        if 'time' in line:
            m_float = map(float,line.split())
            plt.plot(m_float,'bo')
            plt.ylabel('FLOC - % of line')
            plt.xlabel('Sample Number')
            plt.axis([-10,10,0,5])
        plt.show()