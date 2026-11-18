
import torch
import torch.nn as nn


class SignLanguageLSTM(nn.Module):
    def __init__(
        self,
        input_size=512,
        hidden_size=256,
        num_layers=2,
        num_classes=64,
        dropout=0.3,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        # Input shape: [batch, 16, 512]
        _, (hidden, _) = self.lstm(x)

        # Final hidden state of the last LSTM layer
        features = hidden[-1]

        return self.classifier(features)


if __name__ == "__main__":
    model = SignLanguageLSTM()

    dummy = torch.randn(2, 16, 512)
    output = model(dummy)

    print("Input:", dummy.shape)
    print("Output:", output.shape)

    assert output.shape == (2, 64)
    print("LSTM forward test PASSED!")
