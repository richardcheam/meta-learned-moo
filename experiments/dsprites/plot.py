import matplotlib.pyplot as plt

import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser('plot')
    parser.add_argument('--epochs',
                        type=int,
                        default=50,
                        help='number of epochs')
    parser.add_argument('--val-step',
                        type=int,
                        default=10,
                        help='interval between validations')
    parser.add_argument('--resultspath',
                        type=str,
                        help='path to the results JSON file')
    args = parser.parse_args()

    data = {}
    with open(args.resultspath) as f:
        data = json.load(f)

    fig, ax = plt.subplots()

    ax.set_xlabel("Task 1 loss")
    ax.set_ylabel("Task 2 loss")

    for i in range(args.val_step, args.epochs, args.val_step):
        task0_loss = data[f'epoch_{i}']['task0_loss']
        task1_loss = data[f'epoch_{i}']['task1_loss']
        ax.plot(task0_loss, task1_loss, 'x-', color=(1.0 * (i / args.epochs), 0.0, 0.0), label=f'epoch {i}', linewidth=1.0)

    task0_loss = data[f'epoch_{args.epochs}']['task0_loss']
    task1_loss = data[f'epoch_{args.epochs}']['task1_loss']
    ax.plot(task0_loss, task1_loss, 'x-', color=(1.0, 0.0, 0.0), label=f'epoch {args.epochs}', linewidth=1.0)

    leg = plt.legend(loc='best')
    plt.show()