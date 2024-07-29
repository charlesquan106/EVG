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
from networks.resnet_dcn import get_pose_net 
# from networks.resnet_dcn_cut import get_pose_net 
# from networks.resnet_dcn_face import get_pose_net_dcn_face 
# from networks.eff_v2_s import get_pose_net 
# from networks.mob_v2 import get_pose_net
# from networks.mob_v2_face import get_pose_net_mob_face
# from networks.mob_v2_05 import get_pose_net
# from networks.mob_v2_035 import get_pose_net
# from networks.mob_v3_s import get_pose_net
# from networks.mob_v3_l import get_pose_net
# import torchvision.models as models    
# from torchvision.models import mobilenet_v2
# from networks.pose_dla_dcn import get_pose_net as get_dla_dcn
# from networks.large_hourglass import get_large_hourglass_net

# from networks.msra_resnet_CBAM import get_pose_net_CBAM as get_pose_net   

import torch 
from thop import profile
from torchsummary import summary

from torchstat import stat
# calflops
from calflops import calculate_flops

from thop import profile


# img = read_image("/home/owenserver/Python/CenterNet_gaze/src/tools/grace_hopper_517x606.jpg")

# Step 1: Initialize model with the best available weights
# weights = ResNet50_Weights.DEFAULT
# weights = EfficientNet_V2_S_Weights.IMAGENET1K_V1

num_layers = 50
# num_layers = 34

heads = {'hm': 1,'reg':2}

head_conv = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = get_pose_net(heads,head_conv).to(device)
# model = get_pose_net_dcn_face(num_layers,heads,head_conv).to(device)
# model = get_pose_net_dcn(num_layers,heads,head_conv).to(device)

# model = get_pose_net(num_layers,heads,head_conv).to(device)
# model = get_pose_net_mob_face(num_layers,heads,head_conv).to(device)

# model = MobileNetV2(width_mult=0.5)

model = get_pose_net(num_layers,heads,head_conv)
# model = mobilenet_v2(pretrained=True, progress=True)

# model = get_pose_net_mob_face(num_layers,heads,head_conv)


# model = get_pose_net(num_layers,heads,head_conv)
# model = get_pose_net(num_layers= num_layers, heads =heads ).to(device)


# model = get_pose_net_dcn_face(num_layers,heads,head_conv)
# model = get_pose_net_dcn_face(num_layers= num_layers, heads =heads ).to(device)

# model = get_dla_dcn(num_layers,heads)

# model = get_large_hourglass_net(num_layers,heads,head_conv)

# model = models.mobilenet_v2()
# summary(model, input_size=  (3,480,272))
# summary(model, input_size=  (3,224,224))
# summary(model, input_size=  (3,512,512))
# summary(model, input_size=  (3,672,384))


#######  torchstat import stat  #######
# stat(model, (3, 480, 272))
# input = torch.randn(1, 3, 480,272)
# flops, params = profile(model, inputs=(input, ))
# print("flops ",flops)
# print("params",params)


######  calflops import calculate_flops  #######
# flops, macs, params = calculate_flops(model, input_shape=(1, 3, 480, 272))

# pading to 32 times (padding black babckground) -> model size 
# (640, 360) -> 672, 384
# (480, 272) -> 512, 288 

# hourglass special padding number 128 times
# (480, 272) -> 512, 384 

# #####  Paper input size  #####
# flops, macs, params = calculate_flops(model, input_shape=(1, 3, 640, 384))

# #####  Himax input size  #####
# flops, macs, params = calculate_flops(model, input_shape=(1, 3, 672, 384))
flops, macs, params = calculate_flops(model, input_shape=(1, 3, 512, 288))
# flops, macs, params = calculate_flops(model, input_shape=(1, 3, 512, 384))
# flops, macs, params = calculate_flops(model, input_shape=(1, 3, 480, 272))
# flops, macs, params = calculate_flops(model, input_shape=(1, 3, 224, 224))
print("flops ",flops)
# print("macs ",macs)
print("params",params)

