import torch

layer = 15
dt = '29-19-48'

cov = torch.load('memit_num_edits_1024_layer_{}_{}-cov.pt'.format(layer, dt))
cur_zs = torch.load('memit_num_edits_1024_layer_{}_{}-cur_zs.pt'.format(layer, dt))
zs = torch.load('memit_num_edits_1024_layer_{}_{}-zs.pt'.format(layer, dt))
layer_ks = torch.load('memit_num_edits_1024_layer_{}_{}-layer_ks.pt'.format(layer, dt))

targets = zs - cur_zs
print(torch.linalg.norm(zs, dim=0))
print(torch.linalg.norm(cur_zs, dim=0))
print(torch.linalg.norm(targets, dim=0))

layer = 15
dt = '29-19-45'
cov = torch.load('mrome_num_edits_1024_layer_{}_{}-cov.pt'.format(layer, dt))
cur_zs = torch.load('mrome_num_edits_1024_layer_{}_{}-cur_zs.pt'.format(layer, dt))
zs = torch.load('mrome_num_edits_1024_layer_{}_{}-zs.pt'.format(layer, dt))
layer_ks = torch.load('mrome_num_edits_1024_layer_{}_{}-layer_ks.pt'.format(layer, dt))

targets = zs - cur_zs
print(torch.linalg.norm(zs, dim=0))
print(torch.linalg.norm(cur_zs, dim=0))
print(torch.linalg.norm(targets, dim=0))