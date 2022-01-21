import torch
import torch.nn as nn
import numpy as np

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

h = 16
w =16
s = (3, h, w)
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        nbf = 8 
        self.nbf=nbf
        self.conv01 = nn.Conv2d(3, nbf, kernel_size=(3, 3), padding=1, bias=True)
        self.pool0 = nn.MaxPool2d(2,2)
        self.conv02 = nn.Conv2d(nbf, nbf, kernel_size=(3, 3), padding=1, bias=True)
        
        self.conv11 = nn.Conv2d(nbf, nbf, kernel_size=(3, 3), padding=1, bias=True)
        self.conv12 = nn.Conv2d(nbf, nbf, kernel_size=(3, 3), padding=1, bias=True)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv13 = nn.Conv2d(nbf, 2*nbf, kernel_size=(3, 3), padding=1, bias=True)
        
        self.conv21 = nn.Conv2d(2*nbf, 2*nbf, kernel_size=(3, 3), padding=1, bias=True)
        self.conv22 = nn.Conv2d(2*nbf, 2*nbf, kernel_size=(3, 3), padding=1, bias=True)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv23 = nn.Conv2d(2*nbf, 4*nbf, kernel_size=(3, 3), padding=1, bias=True)
 
        self.linear = nn.Linear(4*nbf*h//8*w//8, 2)
        self.apply(weights_init)

    def forward(self, inputs):
        input, = inputs
        x = self.conv01(input)
        x = self.pool0(x)
        x = self.conv02(x)
        
        x0 = self.conv11(x)
        x0 = self.conv12(x0)
        x0 = x0+x
        x0 = self.pool1(x0)
        x0 = self.conv13(x0)
        
        x1 = self.conv21(x0)
        x1 = self.conv22(x1)
        x1 = x1+x0
        x1 = self.pool2(x1)
        x1 = self.conv23(x1)
        
        x1= x1.reshape((1,4*self.nbf*h//8*w//8))
        y=self.linear(x1) 
        return y

model = Model()
input0 = np.linspace(-1.,1,np.prod(s)).reshape((1,h,w,3)).astype(np.float32) # in sadl, tensor are nhwc...
input0 = np.transpose(input0,(0,3,1,2)) # transpose for pytorch
inputs_torch = [ torch.from_numpy(input0)]
inputs_torch[0].requires_grad=True
output = model(inputs_torch)
print("Output",output)
torch.onnx.export(model, inputs_torch, "./pytorch.onnx")

