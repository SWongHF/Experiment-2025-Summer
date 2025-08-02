import torch.nn as nn
import torch.nn.functional as F
import copy
import torch


class MLP(nn.Module):
    def __init__(self, input_class=784, num_classes=10, bit_width=1):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_class, 512, bias=True)
        self.fc2 = nn.Linear(512, 512, bias=True)
        self.fc3 = nn.Linear(512, num_classes, bias=True)
        self.bit_width = bit_width

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
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
                # if not name.endswith(".bias"):
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
