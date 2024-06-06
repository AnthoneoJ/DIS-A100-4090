VERSION = '1.2.0'
"""
- Download image if URL given as input
- Handle bad URL
- Add requests requirement
https://github.com/xuebinqin/DIS
https://github.com/HUANGYming/DIS-A100-4090
https://huggingface.co/spaces/doevent/dis-background-removal/blob/main/app.py
"""
import os, traceback
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

import requests
import numpy as np
import torch
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from huggingface_hub import hf_hub_download

from data_loader_cache import normalize, im_preprocess
from models_isnet import ISNetDIS


def text_to_image(text: str) -> Image.Image:
    font = ImageFont.load_default()

    # Create a temporary ImageDraw object to calculate the text size
    temp_image = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(temp_image)

    # Calculate the text width and height
    text_width, text_height = draw.textsize(text, font=font)

    # Create a new image with the calculated size and white background
    width = text_width + 5
    height = text_height + 5
    image = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(image)

    # Draw the text on the image
    x = (width - text_width) / 2
    y = (height - text_height) / 2
    draw.text((x, y), text, fill='black', font=font)

    return image


class GOSNormalize(object):
    '''
    Normalize the Image using torch.transforms
    '''
    def __init__(self, mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]):
        self.mean = mean
        self.std = std

    def __call__(self,image):
        image = normalize(image,self.mean,self.std)
        return image
    

class ModelHandler:
    def __init__(self, use_gpu: bool = True) -> None:
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.hypar = self.build_hypar()

        # Download model
        self.model_dir = os.path.abspath(self.hypar["model_path"])
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        self.model_path = os.path.join(self.model_dir, self.hypar["restore_model"])
        if not os.path.exists(self.model_path):
            hf_hub_download(repo_id="doevent/dis-background-removal", repo_type="space", filename="isnet.pth", 
                            local_dir=self.model_dir)
            
        self.transform =  transforms.Compose([GOSNormalize([0.5,0.5,0.5],[1.0,1.0,1.0])])
        self.net = self.build_model()

    def get_prediction(self, input_data: dict):
        input_text: str = input_data["input_text"]
        if input_text and input_text.lower()!="placeholder":
            try:
                response = requests.get(input_text)
                input_image = Image.open(BytesIO(response.content))
            except Exception as e:
                text = str(traceback.format_exc())
                return text_to_image(text)
        else:
            input_image: Image.Image = input_data["input_image"]
        if input_image.mode != "RGB":
            input_image = input_image.convert("RGB")
        input_image_np = np.array(input_image)
        image_tensor, orig_size = self.load_image(input_image_np)
        mask = self.predict(image_tensor, orig_size)
        pil_mask = Image.fromarray(mask).convert('L')
        
        im_rgb = input_image.convert("RGB")
        im_rgba = im_rgb.copy()
        im_rgba.putalpha(pil_mask)
        
        return [im_rgba, pil_mask]

    def build_hypar(self) -> dict:
        """ Set Parameters """
        hypar = {} # paramters for inferencing

        hypar["model_path"] ="./saved_models" ## load trained weights from this path
        hypar["restore_model"] = "isnet.pth" ## name of the to-be-loaded weights
        hypar["interm_sup"] = False ## indicate if activate intermediate feature supervision

        ##  choose floating point accuracy --
        hypar["model_digit"] = "full" ## indicates "half" or "full" accuracy of float number
        hypar["seed"] = 0

        hypar["cache_size"] = [1024, 1024] ## cached input spatial resolution, can be configured into different size

        ## data augmentation parameters ---
        hypar["input_size"] = [1024, 1024] ## mdoel input spatial size, usually use the same value hypar["cache_size"], which means we don't further resize the images
        hypar["crop_size"] = [1024, 1024] ## random crop size from the input, it is usually set as smaller than hypar["cache_size"], e.g., [920,920] for data augmentation

        hypar["model"] = ISNetDIS()

        return hypar
    
    def build_model(self):
        net = self.hypar["model"]#GOSNETINC(3,1)

        # convert to half precision
        if(self.hypar["model_digit"]=="half"):
            net.half()
            for layer in net.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.float()

        net.to(self.device)

        if(self.hypar["restore_model"]!=""):
            net.load_state_dict(torch.load(self.hypar["model_path"]+"/"+self.hypar["restore_model"], 
                                           map_location=self.device))
            net.to(self.device)
        net.eval()  
        return net
    
    def load_image(self, im):
        im, im_shp = im_preprocess(im, self.hypar["cache_size"])
        im = torch.divide(im,255.0)
        shape = torch.from_numpy(np.array(im_shp))
        return self.transform(im).unsqueeze(0), shape.unsqueeze(0) # make a batch of image, shape
    
    def predict(self, inputs_val, shapes_val) -> np.ndarray:
        '''
        Given an Image, predict the mask
        '''
        self.net.eval()

        if(self.hypar["model_digit"]=="full"):
            inputs_val = inputs_val.type(torch.FloatTensor)
        else:
            inputs_val = inputs_val.type(torch.HalfTensor)

    
        inputs_val_v = Variable(inputs_val, requires_grad=False).to(self.device) # wrap inputs in Variable
    
        ds_val = self.net(inputs_val_v)[0] # list of 6 results

        pred_val = ds_val[0][0,:,:,:] # B x 1 x H x W    # we want the first one which is the most accurate prediction

        ## recover the prediction spatial size to the orignal image size
        pred_val = torch.squeeze(F.upsample(torch.unsqueeze(pred_val,0),
                                            (shapes_val[0][0], shapes_val[0][1]),
                                            mode='bilinear'))

        ma = torch.max(pred_val)
        mi = torch.min(pred_val)
        pred_val = (pred_val-mi)/(ma-mi) # max = 1

        if self.device == 'cuda': torch.cuda.empty_cache()
        return (pred_val.detach().cpu().numpy()*255).astype(np.uint8) # it is the mask we need
    

if __name__ == '__main__':
    model = ModelHandler()

    from IPython.display import display
    import urllib.request
    url = "https://huggingface.co/spaces/doevent/dis-background-removal/resolve/main/ship.png"
    urllib.request.urlretrieve(url,"_temp.png")
    img = Image.open("_temp.png").convert("RGB")
    input_data = {"input_image": img}
    outputs = model.get_prediction(input_data)
    output = outputs[1]
    im_rgb = img.convert("RGB")
    im_rgba = im_rgb.copy()
    im_rgba.putalpha(output)
    print(type(output))
    print("Masked image:")
    display(im_rgba) # masked
    print("Mask:")
    display(output) # mask

    im0 = im_rgba.save("masked.png") 
    im1 = output.save("mask.png") 

    print('END')
