import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.utils.data as data
import matplotlib.pyplot as plt
import time
import pdb
import os
import sys
import copy

from i3dpt import *
from myDataset import Dataset


def save_checkpoint(epoch, model, optimizer, scheduler, path):
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, path)

save_checkpoint = False

for cluster in ['basic']: 
    for nframes in [32]:
        root_dir = '/home/Dataset/aggregorio_videos_pytorch_centercrop/{}'.format(cluster)
        dist_dir = 'stats/stats_3layer/{}/32_ds_{}_frame_center'.format(cluster, nframes)

        if not os.path.exists(dist_dir):
            os.makedirs(dist_dir)
            if save_checkpoint:
                checkpoint_path = dist_dir+"/checkpoint"
                os.makedirs(checkpoint_path)

        best_model_path = dist_dir+"/best_model.pt"
        readme_path = dist_dir+"/readme.txt"
        confusion_matrix_path = dist_dir+"/confusion_matrix.txt"
        plot_path = dist_dir+"/plot.png"

        batch_size = 32
        num_epochs = 15
        downsample = 2

        dataset_train_path = root_dir+'/train'
        dataset_train = Dataset(dataset_train_path, 32, 2, balance=True, padding=True)
        loader_train = data.DataLoader(
                                dataset_train,
                                batch_size=batch_size,
                            	shuffle=True,
                            	pin_memory=True,
                                num_workers = 4
                            )

        dataset_test_path = root_dir+'/test'
        dataset_test = Dataset(dataset_test_path, 32, 2, balance=False, padding=True)
        loader_test = data.DataLoader(
                                dataset_test,
                                batch_size=batch_size,
                            	shuffle=False,
                            	pin_memory=True,
                                num_workers = 4
                            )

        num_classes = len(dataset_train.classes)

        model_weight =  'model/{}.pt'.format(cluster)
        dropout_prob=0.
        model = I3D(num_classes=num_classes, modality='rgb', dropout_prob=dropout_prob)
        model.load_state_dict(torch.load(model_weight))


        for name,param in model.named_parameters():
            param.requires_grad = False
        for name,param in model.named_parameters():
            if name.split('.')[0] == 'mixed_5c' or name.split('.')[0] == 'conv3d_0c_1x1':
                param.requires_grad = True

        # Send the model to GPU
        model.cuda()

        print("parameters to learn:")
        params_to_update = []
        params_to_update_name = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                params_to_update_name.append(name)
                print("\t",name)

        # Observe that all parameters are being optimized
        w_decay = 0.01
        lr= 0.1
        steps= [20,60]
        optimizer = optim.SGD(params_to_update, lr=lr, momentum=0.9, weight_decay=w_decay)
        criterion = nn.CrossEntropyLoss()

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, steps, gamma=0.1, last_epoch=-1)

        since = time.time()
        train_acc_history = []
        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_epoch = 0
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            conf_matrix = torch.zeros(num_classes, num_classes)
            mean_tot = 0.

            scheduler.step()
            model.train()
            running_loss, correct = 0.0, 0

            for X, y in loader_train:
                X = X.cuda()
                y = y.cuda()

                optimizer.zero_grad()
                #i3d reutrn out and out_logits
                y_, _ = model(X)
                loss = criterion(y_, y)
                loss.backward()
                optimizer.step()

                _, y_label_ = torch.max(y_, 1)
                correct += (y_label_ == y).sum().item()
                running_loss += loss.item() * X.shape[0]

            print(f"    train accuracy: {correct/len(loader_train.dataset):0.3f}")
            print(f"    train loss: {running_loss/len(loader_train.dataset):0.3f}")

            train_acc = correct/len(loader_train.dataset)
            train_acc_history.append(train_acc)

            #validation
            model.eval()
            running_loss, correct = 0.0, 0
            with torch.no_grad():
                for X, y in loader_test:
                    X = X.cuda()
                    y = y.cuda()

                    y_, _ = model(X)
         
                    _, y_label_ = torch.max(y_, 1)
                    
                    for i in range(len(y)):
                        conf_matrix[y[i].item(), y_label_[i].item()] += 1

            _bins = dataset_test.bincount()
            bins = [_bin.item() for _bin in _bins]
            for i in range(num_classes):
                for j in range(num_classes):
                    conf_matrix[i][j] = conf_matrix[i][j]/bins[i]

            mean_tot = 0
            for i in range(num_classes):
                mean_tot += conf_matrix[i][i].item()
            mean_tot /= num_classes

            print(f"    validation accuracy: {mean_tot:0.3f}")
            time_elapsed = time.time() - since
            print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            print()

            epoch_acc = mean_tot
            val_acc_history.append(epoch_acc)

            # deep copy the model
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if save_checkpoint and epoch % 10 == 0:
                print("save model")
                save_checkpoint(epoch, model, optimizer, scheduler, checkpoint_path+"/{}_epoch.pt".format(epoch))

        model.load_state_dict(best_model_wts)
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        torch.save(model.state_dict(), best_model_path)

        #matrice confusione
        print("Calc confusion matrix")
        model.eval()
        conf_matrix = torch.zeros(num_classes, num_classes)
        with torch.no_grad():
            for X, y in loader_test:
                X = X.cuda()
                y = y.cuda()

                y_, _ = model(X)
                _, y_label_ = torch.max(y_, 1)

                for i in range(len(y)):
                    conf_matrix[y[i].item(), y_label_[i].item()] += 1

        _bins = dataset_test.bincount()
        bins = [_bin.item() for _bin in _bins]
        for i in range(num_classes):
            for j in range(num_classes):
                conf_matrix[i][j] = conf_matrix[i][j]/bins[i]

        mean_tot = 0
        for i in range(num_classes):
            mean_tot += conf_matrix[i][i].item()
        mean_tot /= num_classes

        #ßaving some stats
        
        orig_stdout = sys.stdout
        with open(readme_path, 'w+') as f:
            sys.stdout = f
            dataset_train.print()
            dataset_test.print()
            print(params_to_update_name)
            print("batch size:", batch_size)
            print("numero epoche:", num_epochs)
            print("best model at epoch:", best_epoch)
            print("drop out prob:", dropout_prob)
            print(optimizer)
            print(criterion)
            print(scheduler.state_dict())
        sys.stdout = orig_stdout


        train_hist = np.array(train_acc_history)
        val_hist = np.array(val_acc_history)

        fig = plt.figure(figsize=(19.2,10.8), dpi=100)
        plt.title("Validation vs Train")
        plt.xlabel("Training Epochs")
        plt.ylabel("Accuracy")
        plt.plot(range(1,num_epochs+1),train_hist,label="Train")
        plt.plot(range(1,num_epochs+1),val_hist,label="Test")
        plt.ylim((0,1.))
        plt.xticks(np.arange(1, num_epochs+1, 1.0))
        plt.legend()
        fig.savefig(plot_path)

              
        orig_stdout = sys.stdout
        with open(confusion_matrix_path, 'w+') as f:
            sys.stdout = f
            print(conf_matrix)
            print()
            print('accuracy:', mean_tot)
        sys.stdout = orig_stdout
