import json
import numpy as np
import torch
from convnet import ConvNet


class Classifier():
    def __init__(self,model, num_classes=16, device='cpu'):
        # PARAMETERS
        self.model = model
        self.num_classes = num_classes
        self.device = device
        self.OutLabels = np.zeros((num_classes,), dtype=object)
        self.OutLabels[0] = 'Alnus'
        self.OutLabels[1] = 'Betula'
        self.OutLabels[2] = 'Corylus'
        self.OutLabels[3] = 'Fraxinus'
        self.OutLabels[4] = 'Quercus'
        self.OutLabels[5] = 'Populus'
        self.OutLabels[6] = 'Salix'
        self.OutLabels[7] = 'Acer'
        self.OutLabels[8] = 'Juniperus'
        self.OutLabels[9] = 'Picea'
        self.OutLabels[10] = 'Pinus'
        self.OutLabels[11] = 'Ambrosia'
        self.OutLabels[12] = 'Artemisia'
        self.OutLabels[13] = 'Dactylis'
        self.OutLabels[14] = 'Festuca'
        self.OutLabels[15] = 'Non-pollen'

    def predict(self, data):
        with torch.no_grad():
            x = torch.from_numpy(data).to(self.device)
            y = self.model(x)
        y = y.cpu().numpy()
        class_idx = np.argmax(y, axis=1)
        pollen = self.OutLabels[class_idx]
        return pollen


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THR = 1500
num_classes = 16
num_features = 96

checkpoint_path = 'checkpoints/convnet_model.pth'
state_dict = torch.load(checkpoint_path)
model = ConvNet(num_classes=num_classes, drp0=0.0, drp1=0.0)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

clf = Classifier(model, num_classes, device)

#filepath = 'Data/2021_plants/2021_1-20_4-Festuca pratensis/D_000000045_202108120549.json'
filepath = '/media/mynewdrive/DB/2021_plants/2021_1-20_4-Festuca pratensis/D_000000045_202108120549.json'

if filepath.endswith('.json'):
    with open(filepath, 'r') as jsonfile:
        particle_dict = json.load(jsonfile)
        recog_dict = {}
        if particle_dict is not None:
            Data = particle_dict['Data']
            N = len(Data)
            print('Total numbet of particles: ', N)
            Data_flr = np.zeros( (N, 96), dtype=np.float32)
            nFl = 0
            for i in range(N):
                particle = Data[i]
                spect_image = particle['Spectrometer']
                sp_max = np.max(spect_image)
                if sp_max > THR:
                    scat_image = particle['Scattering']['Image']
                    scat_image = np.array(scat_image, dtype=np.float32)
                    scat_image = scat_image.reshape(-1,24)
                    scat_image = scat_image.T
                    n2= scat_image.shape[1]
                    if n2 <96:
                        scat_image = np.hstack( (scat_image, np.zeros( (24, 96-n2), dtype=np.float32) ) )
                    else:
                        scat_image = scat_image[:,:96]
                    scat_image = scat_image.reshape(-1,1, 24, 96)
                    pollen = clf.predict(scat_image)[0]
                    if pollen in recog_dict.keys():
                        recog_dict[pollen] += 1
                    else:
                        recog_dict[pollen] = 1
                    nFl += 1
            print('Number of fluorescencing particles: ', nFl)
            if nFl >0:
                for key in recog_dict.keys():
                    print(key, recog_dict[key])
        else:
            print('Empty file.')
else:
    print('Error. Files must be in JSON format.')

