import re
import argparse
from PIL import Image
import onnx
import torch
from onnx import onnx_pb
from onnx_coreml import convert
import torch.nn as nn
from model import *
import torchvision.transforms as transforms
import cv2
import numpy as np
import logging
from skimage.morphology import remove_small_objects, remove_small_holes

unloader = transforms.ToPILImage()
def tensor_to_PIL(tensor):
    image = tensor.squeeze(0)
    image = unloader(image)
    return image

def init_unet(state_dict):
    #model = UnetMobilenetV2(pretrained=False, num_classes=1, num_filters=32, Dropout=.2)
    model=UnetResNet(pretrained=False)
    model.load_state_dict(state_dict["state_dict"])
    return model

parser = argparse.ArgumentParser(description='crnn_ctc_loss')
parser.add_argument('--tmp_onnx', type=str,default='./train.onnx')
parser.add_argument('--weights_path', type=str,default='./dataset/resnet50_model/resnet50_model_checkpoint_metric.pth')
parser.add_argument('--img_H', type=int, default= 512)
parser.add_argument('--img_W', type=int, default= 512)
args = parser.parse_args()
globals().update(vars(args))

#convert and save ONNX

model = init_unet(torch.load(weights_path, map_location=lambda storage, loc: storage))
#x=torch.randn(1, 3, img_H, img_W)
loader = transforms.Compose([
transforms.ToTensor(),
transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
#pil_img = Image.open('./trump.jpg')
img = cv2.imread('./data/traindata/0624_3.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = np.array(img, dtype=np.uint8)
w,h,c=img.shape
img = cv2.resize(img, (512, 640), interpolation=cv2.INTER_LANCZOS4)
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s
x=img.transpose((2, 0, 1))
x=torch.from_numpy(x/255)
x=x.unsqueeze(0)
print(x.shape)
x= x.to('cpu', torch.float)  
y=model(x)
y_pred=y.cpu().data.numpy()
y_pred=np.squeeze(y_pred)
y_pred=sigmoid(y_pred)
print(y_pred.shape)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
y_pred = remove_small_holes(remove_small_objects(y_pred > .3,min_size=3000))
y_pred = (y_pred * 255).astype(np.uint8)
y_pred = cv2.resize(y_pred, (h, w), interpolation=cv2.INTER_LANCZOS4) 
img=cv2.resize(img, (h, w), interpolation=cv2.INTER_LANCZOS4)       
_,alpha = cv2.threshold(y_pred.astype(np.uint8),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
b, g, r = cv2.split(img)
bgra = [r,g,b, alpha]
y_pred = cv2.merge(bgra,4)
y_pred[:, :, -1] = cv2.morphologyEx(y_pred[:, :, -1], cv2.MORPH_OPEN, kernel)

cv2.imwrite('mask.png',y_pred[:,:,3])
cv2.imwrite('out2.png',y_pred)
out=cv2.imread('./out2.png')
np.set_printoptions(threshold = 1e6)
print(alpha)
for i in range(3):
  out[:,:,i]=out[:,:,i]*(alpha/255)
for i in range (w):
  for j in range(h):
            if out[i,j,2]==0:
                out[i,j,2]=255
cv2.imwrite('out3.jpg',out)
os._exit()


#######################################################
def _convert_upsample(builder, node, graph, err):
    if 'scales' in node.attrs:
        scales = node.attrs['scales']
    elif len(node.input_tensors):
        scales = node.input_tensors[node.inputs[1]]
    else:
        # HACK: Manual scales
        # PROVIDE MANUAL SCALE HERE
        scales = [1, 1, 0.5, 0.5]

    scale_h = scales[2]
    scale_w = scales[3]
    input_shape = graph.shape_dict[node.inputs[0]]
    target_height = int(input_shape[-2] * scale_h)
    target_width = int(input_shape[-1] * scale_w)

    builder.add_resize_bilinear(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        target_height=target_height,
        target_width=target_width,
        mode='UPSAMPLE_MODE'
    )


def _convert_slice_v9(builder, node, graph, err):
    '''
    convert to CoreML Slice Static Layer:
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#L5082
    '''
    INT_MAX = 2 ** 30
    logging.warn(graph.shape_dict)

    data_shape = graph.shape_dict[node.inputs[0]]
    len_of_data = len(data_shape)
    begin_masks = [True] * len_of_data
    end_masks = [True] * len_of_data

    default_axes = list(range(len_of_data))
    default_steps = [1] * len_of_data

    ip_starts = node.attrs.get('starts')
    ip_ends = node.attrs.get('ends')
    axes = node.attrs.get('axes', default_axes)
    steps = node.attrs.get('steps', default_steps)

    starts = [0] * len_of_data
    ends = [0] * len_of_data

    for i in range(len(axes)):
        current_axes = axes[i]
        starts[current_axes] = ip_starts[i]
        ends[current_axes] = ip_ends[i]
        if ends[current_axes] != INT_MAX or ends[current_axes] < data_shape[current_axes]:
            end_masks[current_axes] = False

        if starts[current_axes] != 0:
            begin_masks[current_axes] = False

    builder.add_slice_static(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        begin_ids=starts,
        end_ids=ends,
        strides=steps,
        begin_masks=begin_masks,
        end_masks=end_masks
    )



# ########
torch.onnx.export(model=model,
                  args=x,
                   f=tmp_onnx,
                   opset_version=9,  
                   verbose=True,
    input_names=['data'],
    output_names=['output'],
    do_constant_folding=False,
    export_params=True,
    training=False,
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK)
# # # #

# ##python -m onnxsim tmp.onnx fasttmp.onnx
tmp_onnx='./fasttrain.onnx'
model_proto=onnx.load(tmp_onnx)
coreml_model = convert(
     model_proto,
    #preprocessing_args={'is_bgr': True, 'image_scale': 1.0/255.0},
    preprocessing_args= {'image_scale' : (1.0 / (255.0*0.226)),
                                                          'blue_bias':(-0.406/ 0.226 ),
                                                          'green_bias':(-0.456/ 0.226),
                                                          'red_bias':(-0.485/ 0.226),
                                                          'is_bgr':True},
    image_input_names=['data'],
    image_output_names=['output3'],
    minimum_ios_deployment_target='12',
   # target_ios='13',
    custom_conversion_functions={
       "Slice": _convert_slice_v9,
        'Upsample': _convert_upsample
    },
    #disable_coreml_rank5_mapping=True
     
)



coreml_model.save('./TrianPerson.mlmodel')
print(coreml_model)


