import os
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from sklearn import metrics
from data.dataloader import make_loader
from models.PAU_resnet50 import make_network
from utils.lr_scheduler import LR_Scheduler


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
writer = SummaryWriter(comment='_PAU_ResNet_train')


def main():

    best_pred = 0.0
    best_acc = 0.0
    best_macro = 0.0
    best_micro = 0.0
    lr = 0.00001
    num_epochs = 100
    train_data, val_data, trainloader, valloader = make_loader()
    model = make_network()
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)
    train_params = [{'params': model.get_1x_lr_params(), 'lr': lr},
                    {'params': model.get_10x_lr_params(), 'lr': lr * 10}]
    optimizer = optim.SGD(train_params, momentum=0.9, weight_decay=5e-4, nesterov=False)
    scheduler = LR_Scheduler(mode='step', base_lr=lr, num_epochs=num_epochs, iters_per_epoch=len(trainloader), lr_step=25)

    for epoch in range(num_epochs):
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        acc = 0.0
        micro = 0.0
        macro = 0.0
        count = 0
        model.train()
        for batch_idx, (dataA, dataB, target) in enumerate(trainloader):
            dataA, dataB, target = dataA.to(device), dataB.to(device), target.to(device)
            scheduler(optimizer, batch_idx, epoch, best_pred)
            optimizer.zero_grad()
            pred = model(dataA, dataB)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()
            predict = torch.argmax(pred, 1)
            a = metrics.accuracy_score(target.cpu(), predict.cpu())
            b = metrics.f1_score(target.cpu(), predict.cpu(), average='micro')
            c = metrics.f1_score(target.cpu(), predict.cpu(), average='macro')
            acc += a
            micro += b
            macro += c
            count += 1
            correct = torch.eq(predict, target).sum().double().item()
            running_loss += loss.item()
            running_correct += correct
            running_total += target.size(0)
        loss = running_loss * 32 / running_total
        accuracy = 100 * running_correct / running_total
        acc /= count
        micro /= count
        macro /= count
        writer.add_scalar('scalar/loss_train', loss, epoch)
        writer.add_scalar('scalar/accuracy_train', accuracy, epoch)
        writer.add_scalar('scalar/acc_train', acc, epoch)
        writer.add_scalar('scalar/micro_train', micro, epoch)
        writer.add_scalar('scalar/macro_train', macro, epoch)
        print('Training ',
              'Epoch[%d /50],loss = %.6f,accuracy=%.4f %%, acc = %.4f, micro = %.4f, macro = %.4f' %
              (epoch + 1, loss, accuracy, acc, micro, macro))
        model.eval()
        with torch.no_grad():
            running_loss = 0.0
            running_correct = 0
            running_total = 0
            acc = 0.0
            micro = 0.0
            macro = 0.0
            count = 0
            for batch_idx, (dataA, dataB, target) in enumerate(valloader):
                dataA, dataB, target = dataA.to(device), dataB.to(device), target.to(device)
                optimizer.zero_grad()
                pred = model(dataA, dataB)
                loss = criterion(pred, target)
                predict = torch.argmax(pred, 1)
                a = metrics.accuracy_score(target.cpu(), predict.cpu())
                b = metrics.f1_score(target.cpu(), predict.cpu(), average='micro')
                c = metrics.f1_score(target.cpu(), predict.cpu(), average='macro')
                correct = torch.eq(predict, target).sum().double().item()
                running_loss += loss.item()
                running_correct += correct
                running_total += target.size(0)
                acc += a
                micro += b
                macro += c
                count += 1
            loss = running_loss * 32 / running_total
            accuracy = 100 * running_correct / running_total
            acc /= count
            micro /= count
            macro /= count
            if acc > best_acc:
                best_acc = acc
            if micro > best_micro:
                best_micro = micro
            if macro > best_macro:
                best_macro = macro
            if accuracy > best_pred:
                best_pred = accuracy
            print('best results: ', 'best_acc = %.4f, best_micro = %.4f, best_macro = %.4f, best_pred = %.4f' %
                  (best_acc, best_micro, best_macro, best_pred,))
            writer.add_scalar('scalar/loss_val', loss, epoch)
            writer.add_scalar('scalar/accuracy_val', accuracy, epoch)
            writer.add_scalar('scalar/acc_val', acc, epoch)
            writer.add_scalar('scalar/micro_val', micro, epoch)
            writer.add_scalar('scalar/macro_val', macro, epoch)
            print('Valing',
                  '    Epoch[%d /50],loss = %.6f,accuracy=%.4f %%, acc = %.4f, micro = %.4f, macro = %.4f, running_total=%d,running_correct=%d' %
                  (epoch + 1, loss, accuracy, acc, micro, macro, running_total, running_correct))


if __name__ == "__main__":
    main()
    writer.close()