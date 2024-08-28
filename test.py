import argparse
import cv2
import glob
import numpy as np
import os
import torch

from hat.archs.hat_arch import *


# Global variablesa
MODEL_PATH = "/home/uyenv/HAT/experiments/Real_HAT_GAN_SRx4.pth"
INPUT_FOLDER = '/home/uyenv/HAT/datasets/iloveimg-resized'
OUTPUT_FOLDER = '/home/uyenv/HAT/results/64hatgan'
SCALE = 4
TILE_SIZE = 512
TILE_OVERLAP = 32

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # set up model 
    try:
        model = HAT(upscale=4, in_chans=3, img_size=64, window_size=16, compress_ratio=3, squeeze_factor=30,
                    conv_scale=0.01, overlap_ratio=0.5, img_range=1., depths=[6, 6, 6, 6, 6, 6],
                    embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], gc=32,
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        
        model.load_state_dict(torch.load(MODEL_PATH)['params'], strict=True)
        model.eval()
        model = model.to(device)
    except Exception as e:
        print(f"Error in setting up the model: {e}")
        return
    

    
    window_size = 16
    
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    for idx, path in enumerate(sorted(glob.glob(os.path.join(INPUT_FOLDER, '*')))):
        imgname = os.path.splitext(os.path.basename(path))[0]
        print('Testing', idx, imgname)

        try:
            # read image
            img = cv2.imread(path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
            img_original = img

            # Get image dimensions
            h, w = img.shape[0:2]

            # Check for too large image
            if h > 2500 or w > 2500:
                img = cv2.resize(img, (w / 2, h / 2), interpolation=cv2.INTER_LANCZOS4)
                print('too large size')
                #return img_original

            # Upscale small images
            if h < 300:
                img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)

            # Convert to tensor
            img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
            img = img.unsqueeze(0).to(device)
        except Exception as e:
            print(f"Error in reading or processing image {imgname}: {e}")
            return img_original
        
        # inference
        try:
            with torch.no_grad():
                _, _, h_old, w_old = img.size()
                h_pad = (h_old // window_size + 1) * window_size - h_old
                w_pad = (w_old // window_size + 1) * window_size - w_old
                img = torch.cat([img, torch.flip(img, [2])], 2)[:, :, :h_old + h_pad, :]
                img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w_old + w_pad]
                output = test(img, model, SCALE, TILE_SIZE, TILE_OVERLAP, window_size)
                output = output[..., :h_old * SCALE, :w_old * SCALE]
        except RuntimeError as error:
            print(f"Runtime error during inference on image {imgname}: {error}")
            return img_original
        except Exception as error:
            print(f"Unexpected error during inference on image {imgname}: {error}")
            return img_original
        
        try:
            # save image
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
            output = (output * 255.0).round().astype(np.uint8)
            if SCALE != 2:
                interpolation = cv2.INTER_AREA if SCALE < 2 else cv2.INTER_LANCZOS4
                output = cv2.resize(output, (int(w * SCALE / 2), int(h * SCALE/ 2)), interpolation=interpolation)

            cv2.imwrite(os.path.join(OUTPUT_FOLDER, f'{imgname}.jpg'), output)
        except Exception as e:
            print(f"Error in saving image {imgname}: {e}")
            return img_original



def test(img_lq, model, scale, tile_size, tile_overlap, window_size):
    if tile_size is None:
        # test the image as a whole
        output = model(img_lq)
    else:
        # test the image tile by tile
        b, c, h, w = img_lq.size()
        tile = min(tile_size, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = torch.zeros(b, c, h*scale, w*scale).type_as(img_lq)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_lq[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = model(in_patch)
                out_patch_mask = torch.ones_like(out_patch)

                E[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch)
                W[..., h_idx*scale:(h_idx+tile)*scale, w_idx*scale:(w_idx+tile)*scale].add_(out_patch_mask)
        output = E.div_(W)

    return output


if __name__ == '__main__':
    main()
