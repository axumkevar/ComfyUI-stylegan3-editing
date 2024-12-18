import time
import os
import sys
import pprint
import numpy as np
from PIL import Image, ImageOps, ImageSequence, ImageFile
import dataclasses
import torch
import torchvision.transforms as transforms
from custom_nodes.ComfyUI_stylegan3_editing.editing.interfacegan.face_editor import FaceEditor
from custom_nodes.ComfyUI_stylegan3_editing.editing.styleclip_global_directions import edit as styleclip_edit
from custom_nodes.ComfyUI_stylegan3_editing.models.stylegan3.model import GeneratorType
from custom_nodes.ComfyUI_stylegan3_editing.utils.common import tensor2im
from custom_nodes.ComfyUI_stylegan3_editing.utils.inference_utils import run_on_batch, load_encoder, get_average_image
import dlib
import folder_paths
import node_helpers

from custom_nodes.ComfyUI_stylegan3_editing.utils.alignment_utils import align_face, crop_face, get_stylegan_transform

from comfy.utils import PROGRESS_BAR_ENABLED, ProgressBar

#Preset Models Directory
if "styleCLIP" not in folder_paths.folder_names_and_paths:
    current_paths = [os.path.join(folder_paths.models_dir, "styleCLIP")]
else:
    current_paths, _ = folder_paths.folder_names_and_paths["styleCLIP"]
folder_paths.folder_names_and_paths["styleCLIP"] = (current_paths, '.dat')


class GetImagePath:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {"required":
                    {"image": (sorted(files), {"image_upload": True})},
                }

    RETURN_TYPES = ("STR",)
    RETURN_NAMES = ("path",)
    CATEGORY = "StyleCLIP"
    FUNCTION = "get_image_path"
    def get_image_path(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        return (image_path,)

class LoadImageByPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required":
                    {"image_path": ("STR",{"default": ""}) },
                }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"
    def load_image(self, image_path):
        img = node_helpers.pillow(Image.open, image_path)
        
        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ['MPO']
        
        for i in ImageSequence.Iterator(img):
            i = node_helpers.pillow(ImageOps.exif_transpose, i)

            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]
            
            if image.size[0] != w or image.size[1] != h:
                continue
            
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)

    @classmethod
    def IS_CHANGED(s, image_path):
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

class AlignImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"image_path": ("STR",{"default": ""}) , "predictor" : ("predictor", {}), "detector" : ("detector", {}) },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("aligned_image",)
    CATEGORY = "StyleCLIP"
    FUNCTION = "run_alignment"

    def run_alignment(self, image_path, predictor, detector):
    	path_to_image= str(image_path)
    	aligned_image =align_face(filepath=str(image_path), detector=detector, predictor=predictor) 
    	return (aligned_image,)

class LoadPredictor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": { 
            	"predictor_path" : (folder_paths.get_filename_list("styleCLIP"), ), 
            },
        }

    RETURN_TYPES = ("predictor","detector",)
    RETURN_NAMES = ("predictor","detector",)
    CATEGORY = "StyleCLIP"
    FUNCTION = "load_predictor"

    def load_predictor(self, predictor_path):
    	predictor = dlib.shape_predictor(folder_paths.get_full_path("styleCLIP", predictor_path))
    	detector = dlib.get_frontal_face_detector()
    	return (predictor,detector,)

NODE_CLASS_MAPPINGS = {
    "Get Image Path": GetImagePath,
    "Load Image By Path": LoadImageByPath,
    "Align Image": AlignImage,
    "Load Landmarks": LoadPredictor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Get Image Path": "Get Image Path",
    "Load Image By Path": "Load Image By Path",
    "Align Image": "Align Image(StyleCLIP)",
    "Load Landmarks": "Load Landmarks (StyleCLIP)",
}
