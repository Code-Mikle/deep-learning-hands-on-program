"""
    训练代码
"""
from platform import architecture

import torch
from torch import nn
from data import train_dataloader, test_dataloader
from torch.utils.tensorboard import SummaryWriter
from model import Classifier
import pandas as pd
import wandb
from config.config import conf


def train(_train_dataloader, _model, _loss_fn, _optimizer):
    size = len(_train_dataloader.dataset)
    _model.train()
    for batch_idx, (input_data, target_data) in enumerate(_train_dataloader):

        input_data, target_data = input_data.to(device), target_data.to(device)

        _optimizer.zero_grad()
        output = _model.forward(input_data)
        loss = _loss_fn(output, target_data)
        loss.backward()
        _optimizer.step()

        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * conf.hyper_parameter.batch_size + len(input_data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(_test_dataloader, _model, _loss_fn):
    size = len(_test_dataloader.dataset)
    _test_loss, _correct = 0, 0
    num_batches = len(_test_dataloader)
    _model.eval()
    with torch.no_grad():
        for batch_idx, (input_data, target_data) in enumerate(_test_dataloader):
            input_data, target_data = input_data.to(device), target_data.to(device)

            output = _model(input_data)
            _test_loss += _loss_fn(output, target_data).item()
            _correct += (output.argmax(dim=1) == target_data).sum(dtype=torch.int64).item()
    _test_loss /= num_batches
    _correct /= size
    print(f"Test Error: \n Accuracy: {(100 * _correct):>0.1f}%, Avg loss: {_test_loss:>8f} \n")
    return _test_loss, _correct


if __name__ == '__main__':
    wandb.init(
        # set the wandb project where this run will be logged
        project="deep-learning-hands-on-program",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 1e-3,
            "architecture": "FCNet",
            "dataset": "CIFAR10",
            "epochs": 5
        }

    )

    # conf = OmegaConf.load("config/config.yaml")
    # OmegaConf.to_yaml(conf, resolve=True)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device: ", device)

    writer = SummaryWriter('runs/CIFAR-10_experiment_1')

    train_dataloader = train_dataloader
    test_dataloader = test_dataloader

    model = Classifier().to(device)

    # add_graph
    writer.add_graph(model, torch.zeros([1, 3, 32, 32]).to(device))

    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=conf.hyper_parameter.learning_rate)

    epoch_list = []
    test_loss_list = []
    correct_list = []

    for epoch in range(conf.hyper_parameter.epochs):
        print(f"Epoch {epoch + 1}\n" + "--" * 20)
        train(train_dataloader, model, loss_fn, optimizer)
        test_loss, correct = test(test_dataloader, model, loss_fn)

        # log metrics to wandb
        wandb.log({"acc": correct, "loss": test_loss})

        # log metrics to tensorboard
        epoch_list.append(epoch)
        test_loss_list.append(test_loss)
        correct_list.append(correct)
        writer.add_scalar('test_loss', test_loss, epoch)

    # [optional] finish the wandb run, necessary in notebooks
    wandb.finish()

    # test_loss_numpy = np.array(test_loss_list)
    # accuracy_numpy = np.array(correct_list)
    # np.savetxt((result_path + 'test_loss.csv'), test_loss_numpy, fmt='%.6f')
    # np.savetxt(result_path + 'accuracy.csv', accuracy_numpy, fmt='%.4f')
    loss_accuracy_dict = {'epoch': epoch_list, 'loss': test_loss_list, 'accuracy': correct_list}
    df = pd.DataFrame(loss_accuracy_dict).to_csv(conf.result_path + 'loss_accuracy.csv', index=False)

    # save parameters
    torch.save(model.state_dict(), conf.result_path + 'parameters.pt')
    # save model and parameters
    torch.save(model, conf.result_path + 'model_and_parameters.pt')

    # save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, conf.result_path + 'checkpoint.tar')

    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, conf.result_path + 'checkpoint.pt')
    print("Got it!!!")
