import numpy as np
import torch.nn as nn
import torch
import glob
from torch.utils.data import Dataset, DataLoader
import torchaudio
import sklearn.metrics as metrics # import accuracy_score, top_k_accuracy_score
import torch.optim as optim
import matplotlib.pyplot as plt
from operator import itemgetter

num_frames = 300    # 10ms shift, 25ms window, 3s speech, 300 frames
audio_len = 300*160  # 16kHz sound signal, 160 samples per 10ms
def loadWAV(filename):
    audio, samplerate = torchaudio.load(filename)
    if audio.shape[0]>1:
        audio = torch.mean(audio, dim=0, keepdim=True)
    if samplerate != 16000:
        resampler = torchaudio.transforms.Resample(samplerate, 16000)
        audio = resampler(audio)
        # audio = np.array(audio)
    if audio.shape[1]<audio_len:
        audio = np.pad(audio, (0, (audio_len-audio.shape[0])), 'wrap')
    else:
        audio = audio[:,:audio_len]        # audio shape : [1, 48000]
    return audio


class vcdataset(Dataset):
    def __init__(self, wav_path):
        self.datalist=[]
        self.datalabel=[]
        self.transform = torchaudio.transforms.Spectrogram(n_fft=1024, win_length=400, hop_length=160, normalized=True)
        names = glob.glob(wav_path)
        for name in names:
            label = int(name[42:47])
            self.datalist.append(name)
            self.datalabel.append(label)
            
        self.classids = np.unique(self.datalabel)
        
        self.datalabels = []
        for l in self.datalabel:
            lb =torch.from_numpy(np.array(int(np.flatnonzero(self.classids == l))))
            self.datalabels.append(lb)

    def __getitem__(self, index):
        audio = loadWAV(self.datalist[index])
        melspec = self.transform(audio)
        melspec = (melspec[0]+1e-6).log()
        melspec = melspec[np.newaxis, ...]
        label = self.datalabels[index]
       
        return melspec, label

    def __len__(self):
        return len(self.datalist)


class nnmodel(nn.Module):
    def __init__(self):
        super(nnmodel, self).__init__()

        # instancenorm   = nn.InstanceNorm1d(40)

        self.netcnn = nn.Sequential(
            nn.Conv2d(1, 96, kernel_size=(7,7), stride=(2,2), padding=(2,2)),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),

            nn.Conv2d(96, 256, kernel_size=(5,5), stride=(2,2), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(3,3), stride=(2,2)),

            nn.Conv2d(256, 384, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=(3,3), padding=(1,1)),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(5,3), stride=(3,2)),
        );

        self.fc6 = nn.Conv2d(256, 4096,(9,1), stride=(1,1) )
        self.dropout1 = nn.Dropout(p=0.2)

        self.avgpool = nn.AvgPool2d((1,8), stride=(1,1))
        self.fc7 = nn.Linear(in_features=4096, out_features=1024)
        self.fc8 = nn.Linear(in_features=1024, out_features=118)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x = self.instancenorm(x).unsqueeze(1)
        x = self.netcnn(x)
        x = self.fc6(x)
        x = self.dropout1(x)
        bs = x.shape[0]
        x = self.avgpool(x).view(bs, -1)
        x = self.fc7(x)
        x = self.fc8(x)
        #x = self.softmax(x)
        return x

    # torchsummary.summary(nnmodel().to(device), ())

def ComputeErrorRates(scores, labels, person_id):

      fprs=[]
      fnrs=[]

      arr = np.arange(0,len(scores))
      print(scores, labels, 'scores, labels')
      print(min(scores), max(scores), 'minmax score')
      thresholds=(sorted(np.unique(scores)))
      for t in thresholds:
          tp =0
          tn =0
          fp =0
          fn =0
          pred = np.where(scores>t)
          #print((pred))
          for p in pred[0]:
              if labels[p]==person_id:
                  tp+=1
              else:
                  fp+=1
          p_bar = np.delete(arr, pred)
          for x in p_bar:
              if labels[x]==person_id:
                  fn+=1
              else:
                  tn+=1
          fpr = fp/(fp+tn)
          fnr = fn/(tp+fn)
          fprs.append(fpr)
          fnrs.append(fnr)

      return np.array(fnrs), np.array(fprs)

def speekerVerification(test_dataloader, model, device, criterion):
    x, person_id = next(iter(test_dataloader))
    print(person_id,'person_id')
    model.eval()
    p_tar = 0.01
    c_miss = 1.0
    c_fa = 1.0
    scores = []
    lbls = []

    with torch.no_grad():
        for data in test_dataloader:
            (inputs, label) = data
            inputs = inputs.to(device)
            label = label.to(device)
            outputs = model(inputs)
            lbls.append(label.item())
            scores.append(outputs[0,person_id].item())
        
    scores = np.array(scores)
    lbls = np.array(lbls)
    print(scores, lbls)
    print('verification: ',scores.shape, lbls.shape, 'outs, labs') 
    fpr, fnr = ComputeErrorRates(scores, lbls, person_id)
    #print(labels, predictions)
    #print(fpr, fnr)

    min_c_det = float("inf")
    for i in range(0, len(fnr)):
        c_det = c_miss * fnr[i] * p_tar + c_fa * fpr[i] * (1 - p_tar)
        if c_det < min_c_det:
            min_c_det = c_det

    idx = np.nanargmin(np.absolute((fnr - fpr)))
    eer  = max(fpr[idx],fnr[idx])
    return eer, min_c_det

def train(model, device, train_dataloader, optimizer, criterion):
    model.train()
    accumulated_loss = 0.0
    for data in train_dataloader:
       # print(data.shape)
        inputs, labels = data
       # print(input.shape, labels)
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        #print('train : outputs, ls:',outputs.shape,labels.shape)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        accumulated_loss += loss.item()

    return accumulated_loss / len(train_dataloader)

def test(model, device, dataloader, criterion):
    model.eval()
    accumulated_loss = 0.0
    correct = 0

    with torch.no_grad():
        for data in dataloader:
            (inputs, labels) = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            accumulated_loss += loss.item()

            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum()
            #top_5_acc = metrics.top_k_accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy(), 5)

    num_samples = len(dataloader)
    avg_loss = accumulated_loss / num_samples
    #top_1_acc = metrics.accuracy_score(labels, predictions)
    #top_5_acc = metrics.top_k_accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy(), 5)
    return avg_loss, correct / num_samples  # avg loss, accuracy score




if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('device:', device)

    wav_path = '/ssd_scratch/cvit/neha/sindhu/vox2_test/*/*/*.wav'
    dataset = vcdataset(wav_path)
    dataset_len = len(dataset)
    traintest_dataset_split = int(0.9*dataset_len)
    print('total number of wav file samples:',dataset_len)

    train_dataset, test_dataset = torch.utils.data.random_split(dataset,
                            [traintest_dataset_split, dataset_len-traintest_dataset_split])

    train_dataset_split = int(0.9*len(train_dataset))
    train_dataset, valid_dataset = torch.utils.data.random_split(train_dataset,
                            [train_dataset_split, len(train_dataset)-train_dataset_split])

    print('train, validation test dataset lengths:',len(train_dataset), len(valid_dataset), len(test_dataset))

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset)
    test_dataloader = torch.utils.data.DataLoader(test_dataset)

    model = nnmodel().to(device) # resets parameters

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params = model.parameters(), lr=0.001, momentum=0.9)

    best_val_loss = np.Inf
    loss_history = { "train" : [], "validation" : []}
    for epoch in range(20):
        train_loss = train(model, device, train_dataloader, optimizer, criterion)  
        print(train_loss)
        loss_history["train"].append(train_loss)

        #eer, cd = speekerVerification(valid_dataloader, model, device, criterion)
        #print(eer, cd, 'eer cd')

        val_loss, val_acc = test(model, device, valid_dataloader, criterion)
        loss_history["validation"].append(val_loss)

        if val_loss < best_val_loss and (best_val_loss - val_loss) > 1e-3:
            print("new best model")
            torch.save(model.state_dict(), "best_model.pth")
            best_val_loss = val_loss

        print("[%d] training loss: %.2f" % (epoch + 1,train_loss))
        print("[%d] validation loss: %.2f, validation accuracy: %.2f%%" % (epoch + 1, val_loss, val_acc * 100))
        print()

    model.load_state_dict(torch.load("best_model.pth"))

    test_loss, test_acc = test(model, device, test_dataloader, criterion)
    eer, cdetmin = speekerVerification(test_dataloader, model, device, criterion)
    print('error rates eer, cdetmin', eer, cdetmin)
    print("test loss: %.2f, test accuracy: %.2f%%" % (test_loss, test_acc * 100))
    f = open('voxceleb_accuracies.txt', 'w')
    f.write("test loss: %.2f, test accuracy: %.2f%% \n" % (test_loss, test_acc * 100))
    #f.write("top 1 accuracy : %.2f%%, top 5 accuracy : %.2f%% \n" % (top1_acc*100, top5_acc*100))
    f.write('EER : %.2f\n' % (eer))
    f.write('C_det min : %.2f' %(cdetmin)) 
    f.close()

# # plot the learning curve
    plt.plot(np.arange(1, 21), loss_history["train"], label = "Training")
    plt.plot(np.arange(1, 21), loss_history["validation"], label = "Validation")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend()
    plt.savefig('learning_curve.png')


