from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights

# from torchvision.models import efficientnet_v2_s,EfficientNet_V2_S_Weights

# from msra_resnet import get_pose_net
# from resnet_dcn import get_pose_net
# from dlav0 import get_pose_net 

# from networks.msra_resnet import get_pose_net
# from networks.msra_resnet_cut import get_pose_net
# from networks.resnet_dcn import get_pose_net 
# from networks.resnet_dcn_cut import get_pose_net 
from networks.resnet_dcn_face import get_pose_net_dcn_face 
# from networks.eff_v2_s import get_pose_net 
# from networks.mob_v2 import get_pose_net
# from networks.mob_v3_s import get_pose_net
# from networks.mob_v3_l import get_pose_net
# import torchvision.models as models      


import torch 
from thop import profile
from torchsummary import summary

from torchstat import stat

# img = read_image("/home/owenserver/Python/CenterNet_gaze/src/tools/grace_hopper_517x606.jpg")

# Step 1: Initialize model with the best available weights
# weights = ResNet50_Weights.DEFAULT
# weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1

num_layers = 18

heads = {'hm': 1,'reg':2}

head_conv = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = get_pose_net(heads,head_conv).to(device)
# model = get_pose_net_dcn_face(num_layers,heads,head_conv).to(device)
# model = get_pose_net_dcn(num_layers,heads,head_conv).to(device)

# model = get_pose_net(num_layers,heads,head_conv)
# model = get_pose_net(num_layers= num_layers, heads =heads ).to(device)


# model = get_pose_net_dcn_face(num_layers,heads,head_conv)
model = get_pose_net_dcn_face(num_layers= num_layers, heads =heads ).to(device)

# model = models.mobilenet_v2()
# summary(model, input_size=  (3,480,272))
# summary(model, input_size=  (3,224,224))
summary(model, input_size=  (3,512,512))
# summary(model, input_size=  (3,672,384))



# stat(model, (3, 480, 272))


# input = torch.randn(1, 3, 480,272)
# flops, params = profile(model, inputs=(input, ))
# print("flops ",flops)
# print("params",params)

