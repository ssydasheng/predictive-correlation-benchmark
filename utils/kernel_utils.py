from core.gpflowSlim.neural_kernel_network import NKNWrapper, NeuralKernelNetwork
from core.kernels import KernelWrapper
import numpy as np
from core.gpflowSlim.kernels import RBF

# ====================================== NKN Info =================================================
def NKNInfo(input_dim, ls):
    kernel = [
        {'name': 'Linear',
         'params': {'input_dim': input_dim, 'ARD': True, 'name': 'Linear1'}},
        {'name': 'Linear',
         'params': {'input_dim': input_dim, 'ARD': True, 'name': 'Linear2'}},
        {'name': 'RBF',
         'params': {'input_dim': input_dim, 'lengthscales': ls / 6., 'ARD': True, 'min_ls': 1e-4, 'name': 'RBF1'}},
        {'name': 'RBF',
         'params': {'input_dim': input_dim, 'lengthscales': ls * 2 / 3., 'ARD': True, 'min_ls': 1e-4, 'name': 'RBF2'}},
        {'name': 'RatQuad',
         'params': {'input_dim': input_dim, 'alpha': 0.1, 'lengthscales': ls / 3., 'min_ls': 1e-4, 'name': 'RatQuad1'}},
        {'name': 'RatQuad',
         'params': {'input_dim': input_dim, 'alpha': 1., 'lengthscales': ls / 3., 'min_ls': 1e-4, 'name': 'RatQuad2'}}]

    wrapper = [
            {'name': 'Linear', 'params': {'input_dim': 6, 'output_dim': 8, 'name': 'layer1'}},
            {'name': 'Product', 'params': {'input_dim': 8, 'step': 2, 'name': 'layer2'}},
            {'name': 'Linear', 'params': {'input_dim': 4, 'output_dim': 4, 'name': 'layer3'}},
            {'name': 'Product', 'params': {'input_dim': 4, 'step': 2, 'name': 'layer4'}},
            {'name': 'Linear', 'params': {'input_dim': 2, 'output_dim': 1, 'name': 'layer5'}}]
    wrapper = NKNWrapper(wrapper)
    nkn = NeuralKernelNetwork(input_dim, KernelWrapper(kernel), wrapper)
    return nkn

def GP_additive_rbf_kernel(input_dim, ls, k_type=0):
    if k_type > 0:
        kern = RBF(1., lengthscales=ls[0], active_dims=[0], name='K1')
        for idx in range(1, k_type):
            kern = kern + RBF(1., lengthscales=ls[idx], active_dims=[idx], name='K%d'%(idx+1))
        if k_type < input_dim:
            kern = kern + RBF(1., lengthscales=ls[k_type:], active_dims=list(range(k_type, input_dim)), name='K-full', ARD=True)
    else:
        kern = RBF(1., lengthscales=ls, name='K-full', ARD=True)

    return kern


