# ==========================================================================
# Eses Image Offset
# ==========================================================================
#
# Description:
# The 'Eses Image Offset' node offers basic image
# manipulation including offsetting capabilities within ComfyUI.
# It allows shifting image and mask content horizontally and/or vertically,
# with an option to wrap content around the canvas edges for a tiling effect.
#
# This node takes an image and an an optional mask as input, applies the
# specified offset, and outputs the processed image, mask, the
# applied X and Y offsets, and a descriptive info string.
#
# Version: 1.0.0
# License: See LICENSE.txt
#
# ==========================================================================


import torch
from PIL import Image, ImageOps, ImageChops
import numpy as np

# Define step values as variables 
SMALL_STEP = 1

class EsesImageOffset:
    """
    Applies offset (translation) transformations to an image and an
    optional mask. It also supports wrapping content for tiling effects and
    provides control over fill color for newly exposed areas.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "offset_x": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": SMALL_STEP}),
                "offset_y": ("INT", {"default": 0, "min": -4096, "max": 4096, "step": SMALL_STEP}),
                "wrap_around": (["Off", "On"],),
                "fill_color": ("STRING", {"default": "0,0,0"}),
                "invert_mask_output": (["No", "Yes"],),
            },
            "optional": {
                "image": ("IMAGE",),
                "mask": ("MASK",),
            },
            "hidden": {
                "__output_info__": ("STRING", {"_print_": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "STRING",)
    RETURN_NAMES = ("IMAGE", "MASK", "offset_x", "offset_y", "info",)
    FUNCTION = "apply_image_transformations"
    CATEGORY = "Eses Nodes/Image"


    def _parse_color_string(self, color_string):
        color_string = color_string.strip()
        if not color_string: return (0, 0, 0, 0)
        try:
            if len(color_string) == 6: return tuple(int(color_string[i:i+2], 16) for i in (0, 2, 4)) + (255,)
            if len(color_string) == 8: return tuple(int(color_string[i:i+2], 16) for i in (0, 2, 4, 6))
        except ValueError: pass
        try:
            parts = [int(p.strip()) for p in color_string.split(',')]
            if len(parts) == 3: return tuple(parts) + (255,) # Default to opaque if only RGB is provided
            if len(parts) == 4: return tuple(parts)
        except ValueError: pass
        
        return (0, 0, 0, 0)

    def apply_image_transformations(self, offset_x, offset_y, wrap_around, fill_color, invert_mask_output, image=None, mask=None):

        if image is None and mask is None:
            # Handle no input
            dummy_image = torch.zeros(1, 64, 64, 3)
            dummy_mask = torch.zeros(1, 64, 64)
            return (dummy_image, dummy_mask, offset_x, offset_y, "No Image or Mask Input")

        is_mask_connected = mask is not None
        img_pil, mask_pil = None, None
        current_width, current_height = 0, 0

        # --- Initialize PIL images and dimensions ---
        if image is not None:
            img_mode = 'RGBA' if image.shape[-1] == 4 else 'RGB'
            img_pil = Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8), mode=img_mode)
            
            # Always convert to RGBA if a mask is connected 
            # OR if the image itself already has an alpha channel
            if img_pil.mode != 'RGBA':
                img_pil = img_pil.convert('RGBA')
            current_width, current_height = img_pil.size
        
        if is_mask_connected:
            mask_np = np.clip(mask.cpu().numpy().squeeze() * 255., 0, 255).astype(np.uint8)
            mask_pil = Image.fromarray(mask_np, mode='L')
            
            # If only a mask is provided
            if not img_pil: 
                current_width, current_height = mask_pil.size
        
        elif image is not None:
            # If no mask is connected but there's an 
            # image, create a full white mask (fully opaque)
            mask_pil = Image.new('L', (current_width, current_height), color=255)

        if img_pil and mask_pil and (img_pil.size != mask_pil.size):
            mask_pil = mask_pil.resize(img_pil.size, Image.NEAREST)


        # --- Apply Offset ---
        unified_fill_color_tuple = self._parse_color_string(fill_color)
        
        if img_pil:
            if wrap_around == "On": img_pil = ImageChops.offset(img_pil, offset_x, offset_y)
            else:
                temp_img = Image.new("RGBA", (current_width, current_height), unified_fill_color_tuple)
                
                # Paste the original image (which might 
                # have its own alpha) onto the fill_color background
                temp_img.paste(img_pil, (offset_x, offset_y), img_pil.split()[-1] if img_pil.mode == 'RGBA' else None); img_pil = temp_img
        
        if mask_pil:
            # 255 if mask is connected, 0 if it's a dummy
            mask_fill_value = 255 if is_mask_connected else 0 
            
            if wrap_around == "On": mask_pil = ImageChops.offset(mask_pil, offset_x, offset_y)
            else:
                temp_mask_img = Image.new('L', (current_width, current_height), color=mask_fill_value)
                temp_mask_img.paste(mask_pil, (offset_x, offset_y)); mask_pil = temp_mask_img

        
        # --- Apply incoming mask to image's alpha channel ---

        # If an input mask is connected, use it to control transparency.
        # We invert the mask here because a common convention 
        # is that black in mask means 'visible' while 
        # PIL's putalpha uses black for transparent.
        if img_pil and is_mask_connected:
            
            # Ensure img_pil has an alpha channel 
            # before applying the mask
            if img_pil.mode != 'RGBA':
                img_pil = img_pil.convert('RGBA')
            
            # Invert the mask for use as alpha, so black areas 
            # in the input mask become opaque in the image
            inverted_mask_for_alpha = ImageOps.invert(mask_pil)
            img_pil.putalpha(inverted_mask_for_alpha)

        
        # --- Composite image over fill_color background ---

        # This makes the fill_color appear behind the 
        # transparent areas of the image and converts 
        # the image back to 3 channels.
        if img_pil:
            
            # Ensure the fill_color tuple has an opaque alpha
            # (e.g., (R,G,B,255)) for a solid background
            
            # If user only provided RGB, ensure opaque alpha
            if len(unified_fill_color_tuple) == 3: 
                fill_color_for_composite = unified_fill_color_tuple + (255,)
            
            # Use the alpha provided by user, 
            # or default (0,0,0,0) if parsing failed
            else: 
                fill_color_for_composite = unified_fill_color_tuple

            # Create a new solid background image with the fill_color
            background_img = Image.new("RGBA", img_pil.size, fill_color_for_composite)
            
            # Composite the (potentially transparent) img_pil onto the background
            img_pil = Image.alpha_composite(background_img, img_pil)
            
            # After compositing, convert to RGB to flatten the image
            img_pil = img_pil.convert("RGB")


        # --- Invert Mask for output (if toggle is "Yes") ---

        if mask_pil and invert_mask_output == "Yes":
            mask_pil = ImageOps.invert(mask_pil)


        # --- Convert back to torch tensors ---

        # Output image is now 3-channel (RGB) 
        # because it's composited with fill_color
        output_image = torch.zeros(1, current_height, current_width, 3)
        
        if img_pil:
            img_array = np.array(img_pil).astype(np.float32) / 255.0 # No need to convert to RGB here, it's already done
            output_image = torch.from_numpy(img_array)[None,]

        output_mask = torch.zeros(1, current_height, current_width)
        if mask_pil:
            mask_np_float = np.array(mask_pil).astype(np.float32) / 255.0
            
            # If an input mask was connected, the output mask 
            # should be 1.0 - input mask (ComfyUI typically treats 
            # white as masked, black as unmasked for output)
            if is_mask_connected:
                output_mask = 1.0 - mask_np_float
            else:
                # If no mask was connected, 
                # just use the generated full mask
                output_mask = mask_np_float 
            
            output_mask = torch.from_numpy(output_mask)[None,]

        output_info = f"Offset: ({offset_x}, {offset_y}), Size: {current_width}x{current_height}, Wrapped: {wrap_around}, Mask Output Inverted: {invert_mask_output}"
        
        return (output_image, output_mask, offset_x, offset_y, output_info)