import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import copy
import torch


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(
                inchannel,
                outchannel,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(outchannel, affine=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False
            ),
            nn.BatchNorm2d(outchannel, affine=False),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    inchannel, outchannel, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(outchannel, affine=False),
            )

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10, bit_width=1):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64, 2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes, bias=False)
        self.bit_width = bit_width

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    @staticmethod
    def quantize(data, bit_width):
        if bit_width == 1:
            return torch.sign(data)
        else:
            mini = torch.min(data)
            maxi = torch.max(data)
            return torch.clip(
                torch.round(data / (maxi - mini) * (2**bit_width - 1)),
                -(2 ** (bit_width - 1)),
                2 ** (bit_width - 1) - 1,
            )

    def evaluate(self, dataset, criterion, device, eval=False, qt=False):
        net_copy = copy.deepcopy(self)
        net_copy.eval()
        scheme = "before"
        if qt:
            scheme = "after"
            for name, param in net_copy.named_parameters():
                if not name.endswith(".bias"):
                    param.data = self.quantize(param.data, self.bit_width)
        correct = 0
        total = 0
        sum_loss = 0.0

        for data in dataset:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net_copy(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum()
            loss = criterion(outputs, labels)
            sum_loss += loss.item()
        loss = sum_loss / len(dataset)

        if eval:
            print(
                "Test set's accuracy (%s quantization) is: %.3f%%"
                % (scheme, 100 * correct / total)
            )
        else:
            print(
                "Training set's accuracy (%s quantization) is: %.3f%%"
                % (scheme, 100 * correct / total)
            )
        return loss, 100 * correct / total
