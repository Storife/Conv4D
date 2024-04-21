import torch
import torch.nn as nn






class Conv4D(nn.Module):
    def __init__(self, cin, cout, k, s=1, pad='same', dilation=1, g=1, b=True, pmode='zeros', device=None, dtype=None,):
        super().__init__()
        # self.weight = nn.Parameter(torch.zeros(cout, cin * k * k * k, dtype=dtype, device=device))
        self.weight = nn.Parameter(torch.zeros(1, cout, cin * k * k * k, dtype=dtype, device=device))
        if b:
            self.bias = nn.Parameter(torch.zeros(cout, dtype=dtype, device=device))
        self.cout = cout
        self.k = k
        self.s = s
        self.pad = pad
        self.dilation = dilation
        #self.g = g
        #self.pmode = pmode

    def forward(self, x):
        b, c, h, w, l = x.size()
        x = unfoldNd(x, self.k, self.dilation, self.pad, self.s)#.view(b, c * k * k * k, h * w * l)
        x = torch.matmul(self.weight, x).view(b, self.cout)
        return x










