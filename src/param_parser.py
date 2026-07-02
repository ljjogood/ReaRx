import  argparse
from texttable import Texttable


def parameter_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs_sl",
        type=int,
        default=100,
        help="The number of epochs to train FHPR model"
    )

    parser.add_argument(
        "--epochs_rl",
        type=int,
        default=100,
        help="The number of epochs to train EGHPR model"
    )

    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.2,
        help="The ratio of test data"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="The batch size to use"
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="The learning rate to use"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="The learning rate decay rate"
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0,
        help="The dropout rate to use"
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="The weight of loss function BCE"
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="The weight of loss function MSE"
    )

    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=128,
        help="The dimension of hidden layer for graph representation learning"
    )

    parser.add_argument(
        "--linear_dim1",
        type=int,
        default=32,
        help="The dimension of linear layer1"
    )

    parser.add_argument(
        "--linear_dim2",
        type=int,
        default=16,
        help="The dimension of linear layer2"
    )

    parser.add_argument(
        "--bottle_neurons",
        type=int,
        default=16,
        help="The number of bottle neurons"
    )

    parser.add_argument(
        "--head_number",
        type=int,
        default=8,
        help="The number of attention head in attention module"
    )

    parser.add_argument(
        '-patience',
        type=int,
        default=10,
    )

    return parser.parse_args()


def tab_printer(args):
    args = vars(args)
    keys = sorted(args.keys())
    t = Texttable()
    rows = [["Parameter", "Value"]]
    rows.extend([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    t.add_rows(rows)
    print(t.draw())