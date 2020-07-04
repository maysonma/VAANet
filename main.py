from opts import parse_opts

from core.model import generate_model
from core.loss import get_loss
from core.optimizer import get_optim
from core.utils import local2global_path, get_spatial_transform
from core.dataset import get_training_set, get_validation_set, get_test_set, get_data_loader

from transforms.temporal import TSN
from transforms.target import ClassLabel

from train import train_epoch
from validation import val_epoch

from torch.utils.data import DataLoader
from torch.cuda import device_count

from tensorboardX import SummaryWriter


def main():
    opt = parse_opts()
    opt.device_ids = list(range(device_count()))
    local2global_path(opt)
    model, parameters = generate_model(opt)

    criterion = get_loss(opt)
    criterion = criterion.cuda()
    optimizer = get_optim(opt, parameters)

    writer = SummaryWriter(logdir=opt.log_path)

    # train
    spatial_transform = get_spatial_transform(opt, 'train')
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    target_transform = ClassLabel()
    training_data = get_training_set(opt, spatial_transform, temporal_transform, target_transform)
    train_loader = get_data_loader(opt, training_data, shuffle=True)

    # validation
    spatial_transform = get_spatial_transform(opt, 'test')
    temporal_transform = TSN(seq_len=opt.seq_len, snippet_duration=opt.snippet_duration, center=False)
    target_transform = ClassLabel()
    validation_data = get_validation_set(opt, spatial_transform, temporal_transform, target_transform)
    val_loader = get_data_loader(opt, validation_data, shuffle=False)

    for i in range(1, opt.n_epochs + 1):
        train_epoch(i, train_loader, model, criterion, optimizer, opt, training_data.class_names, writer)
        val_epoch(i, val_loader, model, criterion, opt, writer, optimizer)

    writer.close()


if __name__ == "__main__":
    main()

"""
python main.py --expr_name demo
"""