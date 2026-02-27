import torch.nn as nn
from torchvision.models import resnet18

class PersonAttributeModel(nn.Module):

    def __init__(self):
        super().__init__()

        # load pretrained backbone
        self.backbone = resnet18(weights="DEFAULT")

        # get feature size
        features = self.backbone.fc.in_features

        # remove original classifier
        self.backbone.fc = nn.Identity()

        # attribute heads
        self.head_beard = nn.Linear(features, 3)
        self.head_hair = nn.Linear(features, 3)
        self.head_eye = nn.Linear(features, 2)
        self.head_gender = nn.Linear(features, 2)
        self.head_hat = nn.Linear(features, 2)

    def forward(self, x):

        f = self.backbone(x)

        return {
            "beard": self.head_beard(f),
            "hair": self.head_hair(f),
            "eye": self.head_eye(f),
            "gender": self.head_gender(f),
            "hat": self.head_hat(f)
        }
