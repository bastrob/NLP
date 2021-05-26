import torch.nn as nn
import torch.nn.functional as F


class ReviewClassifier(nn.Module):
    """
    a simple perceptron-based classifier
    """
    def __init__(self, num_features):
        """
        :param num_features: (int) the size of the input feature vector
        """
        super(ReviewClassifier, self).__init__()
        self.fc1 = nn.Linear(in_features=num_features, out_features=1)

    def forward(self, x_in, apply_sigmoid=False):
        """
        The forward pass of the classifier
        :param x_in: (torch.Tensor) an input data tensor
        :param apply_sigmoid: (bool) a flag for the sigmoid activation
        should be false if used with the cross-entropy losses
        :return: the resulting tensor, tensor shape should be (batch,)
        """
        y_out = self.fc1(x_in).squeeze()
        if apply_sigmoid:
            y_out = F.sigmoid(y_out)

        return y_out