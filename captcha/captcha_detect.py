'''
@Descripttion: 
@version: 
@Author: QsWen
@Date: 2020-04-22 20:31:01
@LastEditors: QsWen
@LastEditTime: 2020-04-22 20:31:59
'''
import torch.nn as nn
import numpy as np
import torch
import os
from torch.utils import data
from torchvision import transforms
from PIL import Image

'''
type:验证码识别
net:lstm
loss_fn:crossentrophy
method:output -> (-1,10), target -> (-1,)
result: about 100 epoch acc 1.0，将验证码左边留点空间在epoch 70 acc 1.0
attention:批次不宜过大，batch 8 太小，目前16效果最好

'''
data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


class Dataset(data.Dataset):
    def __init__(self, codedir=r"../datas/captcha_img_train"):
        self.datalist = []
        self.codedir = codedir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        for pic_file in os.listdir(self.codedir):
            self.datalist.append(pic_file)

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, item):
        img_name = self.datalist[item]
        img_path = os.path.join(self.codedir, img_name)
        with Image.open(img_path) as img:
            img_ = self.transform(img)
        label = img_name.split('.')[0]
        label_ = self.setlabel(label)
        # label_ = self.one_hot(label)

        return img_, label_

    def setlabel(self, x):
        '''
        以数字的形式做标签
        :param x:
        :return:
        '''
        y = torch.zeros(4)
        for i in range(4):
            y[i] = int(x[i])
        return y

    def one_hot(self, x):  # x 1235
        '''
        以onehot形式做标签
        :param x:
        :return:
        '''
        y = torch.zeros(4, 10)
        for i in range(4):
            index = int(x[i])
            y[i][index] = 1
        return y


class Encodingnet(nn.Module):
    def __init__(self):
        super(Encodingnet, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(180, 128),
            nn.BatchNorm1d(num_features=128),  # 要求数据NV结构
            nn.ReLU()
        )
        self.lstm = nn.LSTM(128, 128, 2, batch_first=True)
        # self.lstm2 = nn.LSTM(64,64,1)
        self.fc2 = nn.Linear(128, 10)
        # self.fc3 = torch.softmax(dim=2)

    def forward(self, x):
        # 合并CH，并换到外围，相当于合并三个通道的高用于训练
        x_ = x.reshape(-1, 180, 120).permute(0, 2, 1).contiguous().view(-1, 180)  # N,120,180

        y_ = self.fc1(x_)  # 180 -> 128 N,120,180

        # 以W作为步长
        y_ = y_.contiguous().view(-1, 120, 128)  # N, 120, 128

        y2, _ = self.lstm(y_)  # N, 120, 128
        y2_ = y2[:, -1, :].contiguous().view(-1, 1, 128).expand(-1, 4, 128)  # N, 4, 128

        y3, _ = self.lstm(y2_)  # N, 4, 128

        # y3_ = y3.reshape(-1, 128)
        y3_ = y3.contiguous().view(-1, 128)

        y4 = self.fc2(y3_)
        # out = y4.reshape(-1, 4, 10)
        output = y4.contiguous().view(-1, 4, 10)  # N, 4, 10

        return output

class Detector(nn.Module):
    def __init__(self, net=Encodingnet(), netpath="./captcha_net.pt", testdir=r"../datas/captcha_img_test"):
        super(Detector, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = net.to(self.device)
        self.netpath = netpath
        try:
            self.net.load_state_dict(torch.load(netpath))
            print("net param load successful")
        except FileNotFoundError as error:
            print(error)
        # self.testfile = testfile
        self.testdir = testdir
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    # def detect_file(self):


    def get_accruacy(self):
        print("test start")
        test_dataset = Dataset(self.testdir)
        test_dataloader = data.DataLoader(test_dataset, len(test_dataset), shuffle=True, drop_last=True)
        with torch.no_grad():
            self.net.eval()
            for img_, label_ in test_dataloader:
                img_ = img_.to(self.device)
                label_ = label_.view(-1, 4).long()
                output_ = self.net(img_)
                out = output_.detach().cpu()
                out_ = torch.argmax(out, dim=2)
                # print(out_.size(),label_.size())
                # print(out_ == label_)
                acc_test = torch.mean(torch.all(out_ == label_, dim=1).float())
                print("b", out_[0].detach(), "---", label_[0].detach())
                break
        print("acctest: %f" % (acc_test.item()))

    def forward(self, imgpath):
        imgname = os.path.basename(imgpath)
        # print(imgname)
        with Image.open(imgpath) as img:
            img_ = self.transform(img).to(self.device)
        label = imgname.split('.')[0]
        # print(label)
        with torch.no_grad():
            self.net.eval()
            output = self.net(img_)
            out = torch.argmax(output.detach().cpu(),dim=-1)
            # print(out.numpy().tolist()[0])
            # print(["".join(out.numpy().tolist()[0])])
            # print([d for d in out.numpy().tolist()[0]])
            # print("result: {0}{1}{2}{3}".format(*[d for d in out.numpy().tolist()[0]]))
            print("result: {0}{1}{2}{3}".format(*out.numpy().tolist()[0]))


        return 0





if __name__ == '__main__':
    torch.set_printoptions(threshold=100000, sci_mode=False)
    # dataset = Dataset()
    # dataloader = data.DataLoader(dataset,64,shuffle=True)
    # for input,target in dataloader:
    #     print(input.size())
    #     print(target.size())
    #     print(target)
    #     break
    # train()
    dataset = Dataset()
    print(len(dataset))
    net = Encodingnet()
    dataloader = data.DataLoader(dataset, batch_size=5, shuffle=True)
    print(len(dataloader))
    for i, (img, label) in enumerate(dataloader):
        print(img.shape, label.shape)
        y = net(img)
        print(y.shape)
        break
    detector = Detector()
    detector(r"D:\PycharmProjects\rnn_01\datas\captcha_img_test\0073.jpg")

