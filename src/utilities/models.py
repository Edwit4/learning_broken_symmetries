import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self, pixelization='square', layer_size=32):
        super(FCN, self).__init__()
        if pixelization == 'square':
            self.in_size = 27*3
        elif pixelization in ['rect', 'squarenoisy']:
            self.in_size = 144*3
        elif pixelization in ['fulluniform', 'fulluniformrandmask',
                              'fulluniformquadmask', 'fulllessnoiseuniform',
                              'fulllessnoiseuniformrandmask',
                              'fulllessnoiseuniformquadmask',
                              'fullmorenoiseuniform',
                              'fullmorenoiseuniformrandmask',
                              'fullmorenoiseuniformquadmask']:
            self.in_size = 172*3
        elif pixelization in ['fullrect', 'fullrectrandmask',
                              'fullrectquadmask', 'fulllessnoiserect',
                              'fulllessnoiserectrandmask',
                              'fulllessnoiserectquadmask', 'fullmorenoiserect',
                              'fullmorenoiserectrandmask',
                              'fullmorenoiserectquadmask']:
            self.in_size = 112*3
        self.network = nn.Sequential(
            nn.Linear(self.in_size, layer_size),
            nn.LeakyReLU(),
            nn.Linear(layer_size, layer_size),
            nn.LeakyReLU(),
            nn.Linear(layer_size, layer_size),
            nn.LeakyReLU(),
            nn.Linear(layer_size, 1),
            nn.Sigmoid())

    def forward(self, x):
        out = self.network(x)
        return out

class CNN(nn.Module):
    def __init__(self, cnn_layer_size=50, dense_layer_size=200):
        super(CNN, self).__init__()
        self.cnn= nn.Sequential(
            nn.Conv2d(1, cnn_layer_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_layer_size, cnn_layer_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(cnn_layer_size, cnn_layer_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_layer_size, cnn_layer_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(cnn_layer_size*8*8, dense_layer_size),
            nn.Dropout(),
            nn.Linear(dense_layer_size, dense_layer_size),
            nn.Dropout(),
            nn.Linear(dense_layer_size, 1),
            nn.Sigmoid())

    def forward(self, x):
        out = self.cnn(x)
        out = self.fc(out)
        return out
        
class PFN(nn.Module):
    def __init__(self, Phi_size=128, l_size=128, F_size=128):
        super(PFN, self).__init__()
        self.Phi_net = nn.Sequential(
            nn.Linear(3, Phi_size),
            nn.Linear(Phi_size, Phi_size),
            nn.LeakyReLU(),
            nn.Linear(Phi_size, Phi_size),
            nn.LeakyReLU(),
            nn.Linear(Phi_size, l_size))

        self.F_net = nn.Sequential(
            nn.Linear(l_size, F_size),
            nn.Linear(F_size, F_size),
            nn.LeakyReLU(),
            nn.Linear(F_size, F_size),
            nn.LeakyReLU(),
            nn.Linear(F_size, F_size),
            nn.LeakyReLU(),
            nn.Linear(F_size, 1),
            nn.Sigmoid())

    def forward(self, x):
        out = self.Phi_net(x)
        out = torch.sum(out, dim=1)
        out = self.F_net(out)
        return out

class FCN_aug(nn.Module):
    def __init__(self, pixelization='square', layer_size=32):
        super(FCN_aug, self).__init__()
        if pixelization == 'square':
            self.in_size = 27*3
        elif pixelization in ['rect', 'squarenoisy']:
            self.in_size = 144*3
        elif pixelization in ['fulluniform', 'fulluniformrandmask', 'fulluniformquadmask',
                              'fulllessnoiseuniform', 'fulllessnoiseuniformrandmask', 
                              'fulllessnoiseuniformquadmask', 'fullmorenoiseuniform',
                              'fullmorenoiseuniformrandmask', 'fullmorenoiseuniformquadmask']:
            self.in_size = 172*3
        elif pixelization in ['fullrect', 'fullrectrandmask', 'fullrectquadmask',
                              'fulllessnoiserect', 'fulllessnoiserectrandmask',
                              'fulllessnoiserectquadmask', 'fullmorenoiserect',
                              'fullmorenoiserectrandmask', 'fullmorenoiserectquadmask']:
            self.in_size = 112*3

        self.network = nn.Sequential(
            nn.Linear(self.in_size, layer_size),
            nn.LeakyReLU(),
            nn.Linear(layer_size, layer_size),
            nn.LeakyReLU(),
            nn.Linear(layer_size, layer_size),
            nn.LeakyReLU())

        self.classifier = nn.Sequential(
            nn.Linear(layer_size,1),
            nn.Sigmoid())
        
    def forward(self, x):

        if len(x.size()) < 3:
            x = x[:,None,:]

        out = self.network(x)
        out = self.classifier(out)

        if out.size()[1] > 1:
            std = torch.std(out, dim=1)
            std = std.unsqueeze(1).expand(-1, out.size()[1], -1).reshape(-1,
                                                                         std.size()[1])
        else:
            std = torch.zeros(out.size()[0], out.size()[2]).to(x.device)

        out = out.reshape(out.size()[0]*out.size()[1], out.size()[2])

        return out, std

class CNN_aug(nn.Module):
    def __init__(self, cnn_layer_size=50, dense_layer_size=200):
        super(CNN_aug, self).__init__()
        self.cnn= nn.Sequential(
            nn.Conv2d(1, cnn_layer_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_layer_size, cnn_layer_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(cnn_layer_size, cnn_layer_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_layer_size, cnn_layer_size, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(cnn_layer_size*8*8, dense_layer_size),
            nn.Dropout(),
            nn.Linear(dense_layer_size, dense_layer_size),
            nn.Dropout())

        self.classifier = nn.Sequential(
            nn.Linear(dense_layer_size,dense_layer_size),
            nn.LeakyReLU(),
            nn.Linear(dense_layer_size,dense_layer_size),
            nn.LeakyReLU(),
            nn.Linear(dense_layer_size,1),
            nn.Sigmoid())

    def forward(self, x):

        if len(x.size()) < 4:
            x = x[:,None,:,:]

        b_size = x.size()[0]
        num_augs = x.size()[1]

        x = x.reshape(x.size()[0]*x.size()[1], x.size()[2], x.size()[3])

        out = self.cnn(x)
        out = self.fc(out)

        out = out.reshape(b_size, num_augs, out.size()[1])

        if out.size()[1] > 1:
            std = torch.std(out, dim=1)
        else:
            std = torch.zeros(out.size()[0], out.size()[2]).to(x.device)

        out = torch.mean(out, dim=1)
        out = self.classifier(out)

        return out, std
        
class PFN_aug(nn.Module):
    def __init__(self, Phi_size=128, l_size=128, F_size=128):
        super(PFN_aug, self).__init__()
        self.Phi_net = nn.Sequential(
            nn.Linear(3, Phi_size),
            nn.Linear(Phi_size, Phi_size),
            nn.LeakyReLU(),
            nn.Linear(Phi_size, Phi_size),
            nn.LeakyReLU(),
            nn.Linear(Phi_size, l_size))

        self.F_net = nn.Sequential(
            nn.Linear(l_size, F_size),
            nn.Linear(F_size, F_size),
            nn.LeakyReLU(),
            nn.Linear(F_size, F_size),
            nn.LeakyReLU(),
            nn.Linear(F_size, F_size),
            nn.LeakyReLU())

        self.classifier = nn.Sequential(
            nn.Linear(F_size,1),
            nn.Sigmoid())

    def forward(self, x):

        if len(x.size()) < 4:
            x = x[:,None,:,:]

        out = self.Phi_net(x)
        out = torch.sum(out, dim=2)
        out = self.F_net(out)
        out = self.classifier(out)

        if out.size()[1] > 1:
            std = torch.std(out, dim=1)
            std = std.unsqueeze(1).expand(-1, out.size()[1], -1).reshape(-1,
                                                                         std.size()[1])
        else:
            std = torch.zeros(out.size()[0], out.size()[2]).to(x.device)

        out = out.reshape(out.size()[0]*out.size()[1], out.size()[2])

        return out, std
