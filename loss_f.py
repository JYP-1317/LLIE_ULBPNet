import torch
import torch.nn as nn
import torch.nn.functional as F

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