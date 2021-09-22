# ----------------------------------------------------------
# Focal Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang (jianwyan@microsoft.com) 
# ----------------------------------------------------------

import time
import torch
import torch.nn as nn
import torch.nn.functional as F

def window_partition(x, window_size):
    """
    Inputs:
        x: (B, H, W, C)
        window_size (int): window size
    Outputs:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def expand_with_roll(x, expand_size, window_size=1):
    """
    Expand with rolling the feature map
    Inputs:
        x: input feature map, B x H x W x C
        expand_size: expand size for the feature map
        window_size: expand stride for the feature map
    """
    num_heads = 3
    B, H, W, C = x.shape

    x_windows = window_partition(x, window_size).view(-1, window_size * window_size, C)

    x_tl = torch.roll(x, shifts=(-expand_size, -expand_size), dims=(1, 2))
    x_tr = torch.roll(x, shifts=(-expand_size, expand_size), dims=(1, 2))
    x_bl = torch.roll(x, shifts=(expand_size, -expand_size), dims=(1, 2))
    x_br = torch.roll(x, shifts=(expand_size, expand_size), dims=(1, 2))  

    (x_tl_windows, x_tr_windows, x_bl_windows, x_br_windows) = map(
        lambda t: window_partition(t, window_size).view(-1, window_size * window_size, C), 
        (x_tl, x_tr, x_bl, x_br)
    )            
    # NOTE: if expand_size is not equal to half of window_size, there will be either 
    # overlapped region (expand_size < window_size / 2) or missed tokens (expand_size > window_size / 2)
    x_rolled = torch.cat((x_tl_windows, x_tr_windows, x_bl_windows, x_br_windows), 1)

    # get tokens outside of windows 

    return x_rolled

def expand_with_unfold(x, expand_size, window_size=1):
    """
    Expand with unfolding the feature map
    Inputs: 
        x: input feature map -- B x H x W x C
        expand_size: expand size for the feature map
        window_size: expand stride for the feature map
    """
    kernel_size = window_size + 2*expand_size
    unfold = nn.Unfold(kernel_size=(kernel_size, kernel_size), 
        stride=window_size, padding=window_size // 2)

    B, H, W, C = x.shape
    x = x.permute(0, 3, 1, 2).contiguous()  # B x C x H x W
    x_unfolded = unfold(x).view(B, C, kernel_size*kernel_size, -1).permute(0, 3, 2, 1).contiguous().view(-1, kernel_size*kernel_size, C)
    return x_unfolded

if __name__ == '__main__':
    x = torch.rand(64, 56, 56, 64)
    expand_size = 3
    window_size = 7

    # get mask for rolled k and rolled v
    mask_tl = torch.ones(window_size, window_size); mask_tl[:-expand_size, :-expand_size] = 0
    mask_tr = torch.ones(window_size, window_size); mask_tr[:-expand_size, expand_size:] = 0
    mask_bl = torch.ones(window_size, window_size); mask_bl[expand_size:, :-expand_size] = 0
    mask_br = torch.ones(window_size, window_size); mask_br[expand_size:, expand_size:] = 0
    mask_rolled = torch.stack((mask_tl, mask_tr, mask_bl, mask_br), 0).flatten(0)
    valid_ind_rolled = (mask_rolled == 1).nonzero().flatten(0)

    tic = time.time()
    out1 = expand_with_roll(x, expand_size, window_size=7)
    out1 = out1[:, valid_ind_rolled].contiguous()
    print("out shape: ", out1.shape)
    print("time cost with rolling: ", time.time()-tic)

    tic = time.time()
    out2 = expand_with_unfold(x, expand_size, window_size=7)
    print("out shape: ", out2.shape)
    print("time cost with unfolding: ", time.time()-tic)

