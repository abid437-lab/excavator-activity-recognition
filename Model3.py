import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple, Optional


class TwoBranchCNNLSTM(nn.Module):
    def __init__(self, input_channels: int = 3, hidden_size: int = 128, num_ops: int = 5):
        super(TwoBranchCNNLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_ops = num_ops

        # ==========================================================
        # BRANCH 1: RGB Images (ResNet18) - Expects 224x224
        # ==========================================================
        resnet = models.resnet18(pretrained=True)
        self.cnn_img = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC
        self.img_feature_dim = 512

        # ==========================================================
        # BRANCH 2: GAF Images (Custom CNN) - Expects 100x100
        # ==========================================================
        self.cnn_gaf = self._make_cnn_branch(input_channels)
        self.gaf_feature_dim = 128

        # ==========================================================
        # FUSION & LSTM
        # ==========================================================
        # 512 (Image) + 128 (GAF) = 640
        self.lstm_input_size = self.img_feature_dim + self.gaf_feature_dim

        # LSTMCell is perfect for your "for t in range(T)" loop
        self.lstm = nn.LSTMCell(input_size=self.lstm_input_size, hidden_size=hidden_size)

        self.classifier = nn.Linear(hidden_size, num_ops)

    def _make_cnn_branch(self, in_channels: int) -> nn.Sequential:
        """Custom CNN for 100x100 GAF inputs."""
        return nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
        )

    def forward(
            self,
            gaf_t: torch.Tensor,  # <--- GAF COMES FIRST to match your training script
            ext_t: torch.Tensor,  # <--- IMAGES (EXT) COME SECOND
            state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # 1. Process GAF (100x100)
        feat_gaf = self.cnn_gaf(gaf_t)  # (Batch, 128)

        # 2. Process Image (224x224)
        feat_ext = self.cnn_img(ext_t)
        feat_ext = feat_ext.view(feat_ext.size(0), -1)  # (Batch, 512)

        # 3. Concatenate
        fused = torch.cat([feat_gaf, feat_ext], dim=1)  # (Batch, 640)

        # 4. Initialize State (if first frame)
        if state is None:
            batch_size = fused.size(0)
            device = fused.device
            h_0 = torch.zeros(batch_size, self.hidden_size, device=device)
            c_0 = torch.zeros(batch_size, self.hidden_size, device=device)
            state = (h_0, c_0)

        # 5. LSTM Step
        h_t, c_t = self.lstm(fused, state)

        # 6. Classify
        logits = self.classifier(h_t)

        return logits, (h_t, c_t)