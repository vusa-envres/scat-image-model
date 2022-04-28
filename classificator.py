import json
import numpy as np
import torch
from fluorescence_model import Net


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
        
        recog_dict = {}
        for i in range(self.num_classes):
            label = self.OutLabels[i]
            n = np.sum(class_idx==i)
            recog_dict[label] = n

        return recog_dict


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THR = 1500
num_classes = 16
num_features = 96

#fluor_model_path = os.path.join('','model_fl_lsr2.pth')
fluor_model_path = 'checkpoints/model_fl_lsr2.pth'
state_dict = torch.load(fluor_model_path)
model = Net(num_features=num_features, num_classes=num_classes)
model.load_state_dict(state_dict)
model.to(device)
model.eval()

clf = Classifier(model, num_classes, device)

filepath = 'Data/2021_plants/2021_1-20_4-Festuca pratensis/D_000000045_202108120549.json'

if filepath.endswith('.json'):
    with open(filepath, 'r') as jsonfile:
        particle_dict = json.load(jsonfile)

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
                    spc = np.array(spect_image, dtype=np.float32)
                    spc = spc.reshape(32,8)
                    offset = spc[:,7]
                    spc = spc[:,:3] - offset[:,None]
                    spc = spc.flatten()
                    L = np.sqrt(np.sum(spc*spc))
                    Data_flr[nFl] = spc/L
                    nFl += 1
            print('Number of fluorescencing particles: ', nFl)
            if nFl >0:
                Data_flr = Data_flr[:nFl]
                recog_dict = clf.predict(Data_flr)
                for key in recog_dict.keys():
                    print(key, recog_dict[key])
        else:
            print('Empty file.')
else:
    print('Error. Files must be in JSON format.')

