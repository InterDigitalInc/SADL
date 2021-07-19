import torch
import torch.nn as nn


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)

def getInput():
    return [torch.FloatTensor(1, 1, 64, 64).uniform_(-1, 1),
            torch.Tensor([[0.5]]).unsqueeze(1).unsqueeze(1).expand([1, 16, 64, 64])]

class Classifier(nn.Module):
    def __init__(self, num_classes, nbf=16):
        super().__init__()
        self.head_conv = nn.Conv2d(1, nbf, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.head_pool = nn.MaxPool2d((2, 2), (2, 2))

        self.scale0_conv1 = nn.Conv2d(nbf, nbf, kernel_size=(3, 3), padding=1, bias=True)
        self.scale0_conv2 = nn.Conv2d(nbf, nbf, kernel_size=(3, 3), padding=1, bias=True)
        self.scale0_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.scale0_conv3 = nn.Conv2d(nbf, 2*nbf, kernel_size=(1, 1), padding=0, bias=True)

        self.scale1_conv1 = nn.Conv2d(2*nbf, 2*nbf, kernel_size=(3, 3), padding=1, bias=True)
        self.scale1_conv2 = nn.Conv2d(2*nbf, 2*nbf, kernel_size=(3, 3), padding=1, bias=True)
        self.scale1_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.scale1_conv3 = nn.Conv2d(2*nbf, 4*nbf, kernel_size=(1, 1), padding=0, bias=True)

        self.scale2_conv1 = nn.Conv2d(4*nbf, 4*nbf, kernel_size=(3, 3), padding=1, bias=True)
        self.scale2_conv2 = nn.Conv2d(4*nbf, 4*nbf, kernel_size=(3, 3), padding=1, bias=True)
        self.scale2_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.scale2_conv3 = nn.Conv2d(4*nbf, 8*nbf, kernel_size=(1, 1), padding=0, bias=True)

        self.scale3_conv1 = nn.Conv2d(8*nbf, 8*nbf, kernel_size=(3, 3), padding=1, bias=True)
        self.scale3_conv2 = nn.Conv2d(8*nbf, 8*nbf, kernel_size=(3, 3), padding=1, bias=True)
        self.scale3_pool = nn.MaxPool2d((2, 2), (2, 2))
        self.scale3_conv3 = nn.Conv2d(8*nbf, 16*nbf, kernel_size=(1, 1), padding=0, bias=True)

        self.tail_linear = nn.Linear(in_features=1024, out_features=num_classes, bias=True)


    def forward(self, inputs):
        input, input_qp = inputs
        x = self.head_conv(input)
        x = nn.ReLU()(x)
        x = x * input_qp

        x = self.head_pool(x)
        x0 = self.scale0_conv1(x)
        x0 = nn.ReLU()(x0)
        x0 = self.scale0_conv2(x0)
        x0 = x0 + x
        x0 = self.scale0_pool(x0)
        x0 = self.scale0_conv3(x0)

        x1 = self.scale1_conv1(x0)
        x1 = nn.ReLU()(x1)
        x1 = self.scale1_conv2(x1)
        x1 = x1 + x0
        x1 = self.scale1_pool(x1)
        x1 = self.scale1_conv3(x1)

        x2 = self.scale2_conv1(x1)
        x2 = nn.ReLU()(x2)
        x2 = self.scale2_conv2(x2)
        x2 = x2 + x1
        x2 = self.scale2_pool(x2)
        x2 = self.scale2_conv3(x2)

        x3 = self.scale3_conv1(x2)
        x3 = nn.ReLU()(x3)
        x3 = self.scale3_conv2(x3)
        x3 = x3 + x2
        x3 = self.scale3_pool(x3)
        x3 = self.scale3_conv3(x3)

        x3 = x3.permute(0, 2, 3, 1).reshape(-1, 1024)
        o = self.tail_linear(x3)
        return o


def getModel():
    return Classifier(2, nbf=16)
