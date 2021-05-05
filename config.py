import numpy as np

IMG_SIZE=(432, 768, 3)
SCALES = [125.11191814638539, 149.2321302947448, 173.3523424431042, 213.5526960237032, 253.7530496043022]
RATIO = [(1/np.sqrt(3), np.sqrt(3)), (1/np.sqrt(2), np.sqrt(2)), (1, 1), (np.sqrt(2), 1/np.sqrt(2)), (np.sqrt(3), 1/np.sqrt(3))]
K = 5*5
N_SAMPLE = 32
BACKBONE = 'resnet50'
RPN_LAMBDA = 10**3
POOL_SIZE = 7