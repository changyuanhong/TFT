import torch
import numpy as np
import time
# import matplotlib.pyplot as plt
#import d2lzh_pytorch as d2l
from confusion_matrix import sit_confusion_matrix
import pandas as pd

def func_validation(iter_val, net_eval, device_=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    acc_sum, n_eval = 0.0, 0
    with torch.no_grad():
        # net_eval.eval()  # 评估模式，会关闭dropout和BatchNormalization
        for x_eval, y_eval in list(iter_val):
            net_eval.eval()           # 评估模式，会关闭dropout和BatchNormalization
            acc_sum += (net_eval(x_eval.to(device_)) == y_eval.to(device_)).float().sum().cpu().item()           
            #acc_sum += (net_eval(x_eval.to(device_)).argmax(dim=1) == y_eval.to(device_)).float().sum().cpu().item()
            # print('2:{}'.format(torch.cuda.memory_allocated(0)))
            # net_eval.train()          # 改回训练模式
            n_eval += y_eval.shape[0]  # 每计算完一代n会清零
            # del x_eval, y_eval
            # torch.cuda.empty_cache()

        net_eval.train()  # 改回训练模式
    return acc_sum / n_eval


def func_validation_confusion(iter_val, net_eval, device_=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
    acc_sum, n_eval = 0.0, 0
    label_list_, label_pre_list_ = [], []
    with torch.no_grad():
        for x_eval, y_eval in list(iter_val):
            net_eval.eval()  # 评估模式，会关闭dropout和BatchNormalization
            y_pre = net_eval(x_eval.to(device_))
            acc_sum += (y_pre.argmax(dim=1) == y_eval.to(device_)).float().sum().cpu().item()
            net_eval.train()  # 改回训练模式
            n_eval += y_eval.shape[0]  # 每计算完一代n会清零

            label_pre = y_pre.max(axis=1)[1]
            label_np = np.array(y_eval.cpu())
            label_pre_np = np.array(label_pre.cpu())
            label_list_temp = list(label_np)
            label_pre_list_temp = list(label_pre_np)
            label_list_.extend(label_list_temp)
            label_pre_list_.extend(label_pre_list_temp)

    return acc_sum / n_eval, label_list_, label_pre_list_


def func_train(net, iter_tra,iter_val, loss, optimizer, shceduler, epoch_tr, device, show_confusion_matrix=True,
               scheduler_option=True):
    net = net.to(device)  # 占用85,291,520
    print('train on', device)
    batch_count = 0
    plt_tr, plt_va = [], []
    identifier_start = 5
    identifier_end = []
    # out_features = []
    start_all = time.time()
    for num_epoch in range(epoch_tr):
        loss_batch_sum, train_acc_sum, n_batchsize_sum, start = 0.0, 0.0, 0, time.time()
        for _, (x, y) in enumerate(iter_tra):
            y_hat = net(x.to(device))  # 占用16,349,696，模型结束后模型内变量占用的显存自动释放了
            # if num_epoch != epoch_tr-1:
            #     y_hat = net(x)
            # else:
            #     y_hat, features_temp = net.forward_with_features(x)
            #     print(len(features_temp))
            #     out_features.append(features_temp)
            #     features_temp.clear()

            loss_tr = loss(y_hat, y.to(device))  # 占用2048
            optimizer.zero_grad()
            loss_tr.backward()  # 占用70,269,952，在后期每次增加1,198,080
            optimizer.step()  # 占用170,583,040，在后期每次不增加
            loss_batch_sum += loss_tr.cpu().item()
            train_acc_sum += (y_hat.to(device) == y.to(device)).sum().cpu().item()
            #train_acc_sum += (y_hat.argmax(dim=1) == y.to(device)).sum().cpu().item()
            n_batchsize_sum += y.to(device).shape[0]
            batch_count += 1

        if scheduler_option:
            print('第%d个epoch的学习率：%.8f' % (num_epoch+1, optimizer.param_groups[0]['lr']))
            shceduler.step()
        '''
        if num_epoch == epoch_tr-1 and show_confusion_matrix:
            test_acc, label_list, label_pre_list = func_validation_confusion(iter_val, net)
            sit_confusion_matrix(label_list, label_pre_list)
        else:
            test_acc = func_validation(iter_val, net)
        
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec' %
              (num_epoch + 1, loss_batch_sum / batch_count,
               train_acc_sum / n_batchsize_sum, test_acc, time.time() - start))
        plt_tr.append(train_acc_sum / n_batchsize_sum)
        plt_va.append(test_acc)
        '''
        
        list=[loss_batch_sum/batch_count]
        datalist=pd.DataFrame([list])
        datalist.to_csv('C:\\Users\\Administrator\\Desktop\\loss.csv',mode='a',header=False,index=False)
        print('epoch %d, loss %.4f time %.4f sec' %
              (num_epoch + 1, loss_batch_sum / batch_count,
               time.time() - start))
        plt_tr.append(train_acc_sum / n_batchsize_sum)

        
        

        # 人工判断是否结束训练
        # if int(loss_batch_sum / batch_count) == 0 and num_epoch + 1 >= 80:
        #     identifier_start += 1
        #     if identifier_start == 6:
        #         while True:
        #             stage = input('程序是否继续运行：', )
        #             if stage != 'stop':
        #                 identifier_start = 0
        #                 break
        #             else:
        #                 test_acc, label_list, label_pre_list = func_validation_confusion(iter_val, net)
        #                 sit_confusion_matrix(label_list, label_pre_list)
        #                 epoch_tr = num_epoch + 1
        #                 identifier_end.append(stage)
        #                 break
        # if len(identifier_end) > 0:
        #     if identifier_end[0] == 'stop':
        #         break

    print('Full training time %.1f sec' % (time.time()-start_all))
    '''
    d2l.semilogy(range(1, epoch_tr + 1), plt_tr, 'epochs', 'accuracy',
                 range(1, epoch_tr + 1), plt_va, ['train', 'test'])
    # d2l.plt.savefig('tl_loss.png', dpi=200, format='png')
    d2l.plt.close()
    # d2l.plt.show()
    '''
    return
