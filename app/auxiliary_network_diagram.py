from torchviz import make_dot
import torch
from torch.utils.tensorboard import SummaryWriter

from app.auxiliary_files.model_methods.networks import DeepConvolutionalClassifierInceptionBranch3NoEncBigBottleneckComposed


OUTPUT_PATH = '/home/ferran/Documents/bsc/time_series/workloads_resource_prediction/output'


def main():
    """
    model = DeepConvolutionalClassifierInceptionBranch3NoEncBigBottleneckComposed(
        lag_size=128,
        prediction_size=1,
        number_features=8,
        device='cpu'
    )
    input = torch.zeros((5, 8, 128))
    output = model(input)
    make_dot(output, params=dict(list(model.named_parameters()))).render(f'{OUTPUT_PATH}/model', format="png")
    """

    model = DeepConvolutionalClassifierInceptionBranch3NoEncBigBottleneckComposed(
        lag_size=128,
        prediction_size=1,
        number_features=8,
        device='cpu'
    )
    input = torch.zeros((5, 8, 128))
    writer = SummaryWriter('runs/fashion_mnist_experiment_1')
    writer.add_graph(model, input)
    writer.close()


if __name__ == '__main__':
    main()
