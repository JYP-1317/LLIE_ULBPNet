import torch
import torch.nn as nn
import torch.nn.functional as F
"""
def rgb2hsv(rgb: torch.Tensor) -> torch.Tensor:
    cmax, cmax_idx = torch.max(rgb, dim=1, keepdim=True)
    cmin = torch.min(rgb, dim=1, keepdim=True)[0]
    delta = cmax - cmin
    hsv_h = torch.empty_like(rgb[:, 0:1, :, :])
    cmax_idx[delta == 0] = 3
    hsv_h[cmax_idx == 0] = (((rgb[:, 1:2] - rgb[:, 2:3]) / delta) % 6)[cmax_idx == 0]
    hsv_h[cmax_idx == 1] = (((rgb[:, 2:3] - rgb[:, 0:1]) / delta) + 2)[cmax_idx == 1]
    hsv_h[cmax_idx == 2] = (((rgb[:, 0:1] - rgb[:, 1:2]) / delta) + 4)[cmax_idx == 2]
    hsv_h[cmax_idx == 3] = 0.
    hsv_h /= 6.
    hsv_s = torch.where(cmax == 0, torch.tensor(0.).type_as(rgb), delta / cmax)
    hsv_v = cmax
    torch.cat([hsv_h, hsv_s, hsv_v], dim=1)

    return hsv_s
"""

'''
##Zero DCE color loss

class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):

        #b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)

        return k
'''
class RGB_variation(nn.Module):

    def __init__(self):
        super(RGB_variation, self).__init__()

    def forward(self, x, y):

        xr, xg, xb = torch.split(x, 1, dim=1)
        xvr = torch.mean(torch.pow(xr, 2)) - torch.pow(torch.mean(xr), 2)
        xvg = torch.mean(torch.pow(xg, 2)) - torch.pow(torch.mean(xg), 2)
        xvb = torch.mean(torch.pow(xb, 2)) - torch.pow(torch.mean(xb), 2)

        yr, yg, yb = torch.split(y, 1, dim=1)
        yvr = torch.mean(torch.pow(yr, 2)) - torch.pow(torch.mean(yr), 2)
        yvg = torch.mean(torch.pow(yg, 2)) - torch.pow(torch.mean(yg), 2)
        yvb = torch.mean(torch.pow(yb, 2)) - torch.pow(torch.mean(yb), 2)

        k = (torch.pow((xvr-yvr), 2) + torch.pow((xvg-yvg), 2) + torch.pow((xvb-yvb), 2))/3

        return k

class Intensity_loss(nn.Module):

    def __init__(self):
        super(Intensity_loss, self).__init__()

    def forward(self, x, y):

        xr, xg, xb = torch.split(x, 1, dim=1)
        x_intensity = (xr + xg + xb)/3
        yr, yg, yb = torch.split(y, 1, dim=1)
        y_intensity = (yr + yg + yb) / 3

        k = torch.mean(torch.pow((x_intensity-y_intensity), 2))

        return k

class RGB_pix_variation(nn.Module):

    def __init__(self):
        super(RGB_pix_variation, self).__init__()

    def forward(self, x, y):

        xr, xg, xb = torch.split(x, 1, dim=1)
        yr, yg, yb = torch.split(y, 1, dim=1)
        #print(x.shape)
        #print(xr.shape)

        xrgb_mean = (xr + xg + xb) / 3
        xrgb_mean2 = (xr ** 2 + xg ** 2 + xb ** 2) / 3
        xpixel_xvar = xrgb_mean2 - xrgb_mean

        #print(xrgb_mean.shape)
        #print(xrgb_mean2.shape)
        #print(xpixel_xvar.shape)

        #print(xrgb_mean)
        #print(xrgb_mean2)
        #print(xpixel_xvar)

        yrgb_mean = (yr + yg + yb) / 3
        yrgb_mean2 = (yr ** 2 + yg ** 2 + yb ** 2) / 3
        ypixel_yvar = yrgb_mean2 - yrgb_mean

        k2 = torch.mean(torch.pow((xpixel_xvar-ypixel_yvar), 2))

        #(k.shape)
        #print(k)
        return k2

class RGB_skew(nn.Module):

    def __init__(self):
        super(RGB_skew, self).__init__()

    def forward(self, x, y):

        xr, xg, xb = torch.split(x, 1, dim=1)
        xvr = torch.mean(torch.pow(xr, 2)) - torch.pow(torch.mean(xr), 2)
        xvg = torch.mean(torch.pow(xg, 2)) - torch.pow(torch.mean(xg), 2)
        xvb = torch.mean(torch.pow(xb, 2)) - torch.pow(torch.mean(xb), 2)

        xsr = (torch.mean(torch.pow(xr, 3)) - 3 * torch.mean(xr) * xvr - torch.pow(torch.mean(xr), 3))/torch.pow(xvr,
                                                                                                                 1.5)
        xsg = (torch.mean(torch.pow(xg, 3)) - 3 * torch.mean(xg) * xvg - torch.pow(torch.mean(xg), 3))/torch.pow(xvg,
                                                                                                                 1.5)
        xsb = (torch.mean(torch.pow(xb, 3)) - 3 * torch.mean(xb) * xvb - torch.pow(torch.mean(xb), 3))/torch.pow(xvb,
                                                                                                                 1.5)

        yr, yg, yb = torch.split(y, 1, dim=1)
        yvr = torch.mean(torch.pow(yr, 2)) - torch.pow(torch.mean(yr), 2)
        yvg = torch.mean(torch.pow(yg, 2)) - torch.pow(torch.mean(yg), 2)
        yvb = torch.mean(torch.pow(yb, 2)) - torch.pow(torch.mean(yb), 2)

        ysr = (torch.mean(torch.pow(yr, 3)) - 3 * torch.mean(yr) * yvr - torch.pow(torch.mean(yr), 3))/torch.pow(yvr,
                                                                                                                 1.5)
        ysg = (torch.mean(torch.pow(yg, 3)) - 3 * torch.mean(yg) * yvg - torch.pow(torch.mean(yg), 3))/torch.pow(yvg,
                                                                                                                 1.5)
        ysb = (torch.mean(torch.pow(yb, 3)) - 3 * torch.mean(yb) * yvb - torch.pow(torch.mean(yb), 3))/torch.pow(yvb,
                                                                                                                 1.5)

        k1 = (torch.pow((xsr-ysr), 2) + torch.pow((xsg-ysg), 2) + torch.pow((xsb-ysb), 2))/3

        return k1


class RGB_kurt(nn.Module):

    def __init__(self):
        super(RGB_kurt, self).__init__()

    def forward(self, x, y):
        xr, xg, xb = torch.split(x, 1, dim=1)
        xkr = torch.mean(torch.pow(xr - torch.mean(xr), 4)) / torch.pow(torch.mean(torch.pow(xr-torch.mean(xr), 2)),
                                                                        2)
        xkg = torch.mean(torch.pow(xr - torch.mean(xg), 4)) / torch.pow(torch.mean(torch.pow(xg - torch.mean(xg), 2)),
                                                                        2)
        xkb = torch.mean(torch.pow(xb - torch.mean(xb), 4)) / torch.pow(torch.mean(torch.pow(xb - torch.mean(xb), 2)),
                                                                        2)

        yr, yg, yb = torch.split(y, 1, dim=1)
        ykr = torch.mean(torch.pow(yr - torch.mean(yr), 4)) / torch.pow(torch.mean(torch.pow(yr - torch.mean(yr), 2)),
                                                                        2)
        ykg = torch.mean(torch.pow(yr - torch.mean(yg), 4)) / torch.pow(torch.mean(torch.pow(yg - torch.mean(yg), 2)),
                                                                        2)
        ykb = torch.mean(torch.pow(yb - torch.mean(yb), 4)) / torch.pow(torch.mean(torch.pow(yb - torch.mean(yb), 2)),
                                                                        2)

        o = (torch.pow((xkr - ykr), 2) + torch.pow((xkg - ykg), 2) + torch.pow((xkb - ykb), 2)) / 3

        return o