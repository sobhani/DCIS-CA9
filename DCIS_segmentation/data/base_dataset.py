import torch.utils.data as data
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
import numpy as np
import random

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize            
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))
    
    #flip = random.random() > 0.5
    flip = True
    rotation = True
    blur = True
    return {'crop_pos': (x, y), 'flip': flip, 'rotation': rotation, 'blur': blur}

def get_transform(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, method))   
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))
        
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip_vertical(img, params['flip'])))
        transform_list.append(transforms.Lambda(lambda img: __flip_horizontal(img, params['flip'])))

    if opt.isTrain and not opt.no_rotate:
        transform_list.append(transforms.Lambda(lambda img: __rotate_90(img, params['rotation'])))
        transform_list.append(transforms.Lambda(lambda img: __rotate_180(img, params['rotation'])))
        transform_list.append(transforms.Lambda(lambda img: __rotate_270(img, params['rotation'])))

    if opt.isTrain and not opt.no_blur:
        transform_list.append(transforms.Lambda(lambda img: __blur(img, params['blur'])))


    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def normalize():    
    return transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size        
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if (h == oh) and (w == ow):
        return img
    return img.resize((w, h), method)

def __scale_width(img, target_width, method=Image.BICUBIC):
    ow, oh = img.size
    if (ow == target_width):
        return img    
    w = target_width
    h = int(target_width * oh / ow)    
    return img.resize((w, h), method)

def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):        
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

def __flip_vertical(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img

def __flip_horizontal(img, flip):
    if flip:
        return img.transpose(Image.FLIP_TOP_BOTTOM)
    return img

def __rotate_90(img, rotation):
    if rotation:
        return img.transpose(Image.ROTATE_90)
    return img

def __rotate_180(img, rotation):
    if rotation:
        return img.transpose(Image.ROTATE_180)
    return img

def __rotate_270(img, rotation):
    if rotation:
        return img.transpose(Image.ROTATE_270)
    return img

def __blur(img, blur):
    if blur:
        return img.filter(ImageFilter.GaussianBlur(radius = 2)) 
    return img

