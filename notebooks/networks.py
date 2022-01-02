
# DLL Project
# University of Trento
# 
# A. Leone, A. E. Piotti
# August-December 2021
#

import torch
import torch.nn          as nn
import torchvision


def ResNet50():

    model    = torchvision.models.resnet50(pretrained=True) 
    model.fc = nn.Linear(
        in_features=model.fc.in_features,
        out_features=2, bias=True
    )

    return model

class CNN(nn.Module): 
    
    def __init__(self, size):
        super().__init__()
        
        self.cnv1 = nn.Conv2d(  3,  32, kernel_size=(5,5) )
        self.cnv2 = nn.Conv2d( 32,  64, kernel_size=(4,4) )
        self.cnv3 = nn.Conv2d( 64, 128, kernel_size=(3,3) )
        
        self.bn1  = nn.BatchNorm2d( 32)
        self.bn2  = nn.BatchNorm2d( 64)
        self.bn3  = nn.BatchNorm2d(128)
        
        self.pool = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
        self.drop = nn.Dropout(p=0.25)
        
        self.fc1  = nn.Linear(size, 500)
        self.fc2  = nn.Linear( 500, 100)
        self.fc3  = nn.Linear( 100,  50)
        self.fc4  = nn.Linear(  50,  10)
        self.fc5  = nn.Linear(  10,   2)
    
    def forward(self, x):
        
        x = nn.functional.relu( self.cnv1(x) )
        x = self.bn1(x)
        x = self.pool(x)
        
        x = nn.functional.relu( self.cnv2(x) )
        x = self.bn2(x)
        x = self.pool(x)
        
        x = nn.functional.relu( self.cnv3(x) )
        x = self.bn3(x)
        x = self.pool(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.drop(x)
        
        x = nn.functional.relu( self.fc1(x) )
        x = nn.functional.relu( self.fc2(x) )
        x = nn.functional.relu( self.fc3(x) )
        x = nn.functional.relu( self.fc4(x) )
        x = nn.functional.tanh( self.fc5(x) )
        
        return x


class CNN_Age(nn.Module): 
    
    def __init__(self):
        super().__init__()
        
        self.fc1  = nn.Linear(  8, 32 )
        self.fc2  = nn.Linear( 32, 64 )
        self.fc3  = nn.Linear( 64, 32 )
        self.fc4  = nn.Linear( 32,  4 )
    
    def forward(self, x):
    
        x = x.unsqueeze(0)
        x = x.float()
        
        x = nn.functional.gelu( self.fc1(x) )
        x = nn.functional.gelu( self.fc2(x) )
        x = nn.functional.gelu( self.fc3(x) )
        x = nn.functional.gelu( self.fc4(x) )
        
        return x


class CNN_Colors_Up(nn.Module): 
    
    def __init__(self):
        super().__init__()
        
        self.fc1  = nn.Linear( 18, 72 )
        self.fc2  = nn.Linear( 72, 36 )
        self.fc3  = nn.Linear( 36,  9 )
    
    def forward(self, x):
    
        x = x.unsqueeze(0)
        x = x.float()
        
        x = nn.functional.gelu( self.fc1(x) )
        x = nn.functional.gelu( self.fc2(x) )
        x = nn.functional.gelu( self.fc3(x) )
        
        return x


class CNN_Colors_Down(nn.Module): 
    
    def __init__(self):
        super().__init__()
        
        self.fc1  = nn.Linear( 20, 80 )
        self.fc2  = nn.Linear( 80, 40 )
        self.fc3  = nn.Linear( 40, 10 )
    
    def forward(self, x):
    
        x = x.unsqueeze(0)
        x = x.float()
        
        x = nn.functional.gelu( self.fc1(x) )
        x = nn.functional.gelu( self.fc2(x) )
        x = nn.functional.gelu( self.fc3(x) )
        
        return x


class SiameseNet_v0(nn.Module): 
    
    def __init__(self, vs, es):
        super().__init__()
        
        self.fc1 = nn.Linear( vs*1, vs*3 )
        self.fc2 = nn.Linear( vs*3, vs*2 )
        self.fc3 = nn.Linear( vs*2, es*1 )
    
    def forward(self, x):
    
        x = nn.functional.gelu( self.fc1(x) )
        x = nn.functional.gelu( self.fc2(x) )
        x = nn.functional.gelu( self.fc3(x) )
        
        return x


class SiameseNet(nn.Module): 
    
    def __init__(self, vs, es):
        super().__init__()
        
        self.fc1 = nn.Linear( vs*1, vs*2 )
        self.fc2 = nn.Linear( vs*2, vs*4 )
        self.fc3 = nn.Linear( vs*4, es*1 )
    
    def forward(self, x):
    
        x = nn.functional.gelu( self.fc1(x) )
        x = nn.functional.gelu( self.fc2(x) )
        x = nn.functional.gelu( self.fc3(x) )
        
        return x


class SiameseCNN(nn.Module): 
    
    def __init__(self):
        super().__init__()
        
        self.cnv1 = nn.Conv2d( 3, 30, kernel_size=(5,5))
        self.cnv2 = nn.Conv2d(30, 20, kernel_size=(4,4))
        self.cnv3 = nn.Conv2d(20, 10, kernel_size=(3,3))
        
        self.bn1  = nn.BatchNorm2d(30)
        self.bn2  = nn.BatchNorm2d(20)
        self.bn3  = nn.BatchNorm2d(10)
        
        self.pool = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
        self.drop = nn.Dropout(p=0.25)
        
        self.fc1  = nn.Linear(480, 240)
        self.fc2  = nn.Linear(240, 120)
        self.fc3  = nn.Linear(120,  10)
        self.fc4  = nn.Linear( 10,  10)
        self.fc5  = nn.Linear( 10,   2)
    
    def forward(self, x):
        
        x = nn.functional.relu( self.cnv1(x) )
        x = self.bn1(x)
        x = self.pool(x)
        
        x = nn.functional.relu( self.cnv2(x) )
        x = self.bn2(x)
        x = self.pool(x)
        
        x = nn.functional.relu( self.cnv3(x) )
        x = self.bn3(x)
        x = self.pool(x)
        
        x = x.reshape(x.shape[0], -1)
        x = self.drop(x)
        
        x = nn.functional.relu( self.fc1(x) )
        x = nn.functional.relu( self.fc2(x) )
        x = nn.functional.relu( self.fc3(x) )
        x = nn.functional.relu( self.fc4(x) )
        x = nn.functional.tanh( self.fc5(x) )
        
        return x


def lib():
	return {
		'age':         CNN_Age(),
		'colors_up':   CNN_Colors_Up(),
		'colors_down': CNN_Colors_Down()
	}
