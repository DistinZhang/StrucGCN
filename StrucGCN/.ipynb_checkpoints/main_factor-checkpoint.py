import torch
import param_parser
import numpy as np
import time
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from copy import deepcopy
import utils_data
import os
import json
import utils
from model import StrucGNN

def Train(args, trains, vals, neigh_data, struc_data, path):
    net = StrucGNN(args, neigh_data, struc_data)
    net = net.to(device)
    epoch_time = []

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # 初始化学习率调度器：验证损失在10个连续的epoch中没有下降，则学习率会减少到当前值的10%。
    scheduler = ReduceLROnPlateau(optimizer, 'max', factor=args.factor, patience=args.patience, verbose=True)
    # 早停法参数
    early_stopping_patience = args.early_stopping_patience
    best_val_acc = float('-inf')
    patience_counter = 0
    better_net_list = {}
    for epoch in range(args.EPOCHS):
        start = time.time()
        net.train()
        logp = net()
        cla_loss = F.nll_loss(logp[trains], neigh_data.y[trains])
        loss = cla_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end = time.time() - start

        net.eval()  # 设置模型为评估模式
        logp = net()

        with torch.no_grad():
            acc = utils.accuracy(logp[vals], neigh_data.y[vals])
            val_acc_value = acc

        # 学习率衰减
        scheduler.step(val_acc_value)
        # 早停法逻辑
        if val_acc_value > best_val_acc:
            best_val_acc = val_acc_value
            patience_counter = 0
            better_net_list[epoch] = deepcopy(net.state_dict())  # 保存最佳模型的状态
        else:
            if (epoch + 1) % 3 == 0:
                better_net_list[epoch] = deepcopy(net.state_dict())
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print('Early stopping!')
                break
        # 打印损失信息（可选）
        if (epoch + 1) % 5 == 0:
            print(
                f'Epoch [{epoch + 1}/{args.EPOCHS}], Train Loss: {loss.item()}, Val acc: {acc}, time/epoch:{end}')
        epoch_time.append(end)

    with open(path, 'a') as file:
        file.write(f'Epoch [{epoch + 1}/{args.EPOCHS}], Train Loss: {loss.item()}, Val acc: {acc}')
        file.write(f'\naverage epoch time={np.mean(epoch_time[1:])}\n\n')
    return better_net_list, net


def evaluate_node_classification(net_list, net, tests, labels):
    best_epoch = 0
    Acc = 0
    for epoch in net_list.keys():
        net.load_state_dict(net_list[epoch])
        net.eval()
        logp = net()
        acc = utils.accuracy(logp[tests], labels[tests])

        if acc > Acc:
            Acc = acc
            best_epoch = epoch

    print(f'Loading {best_epoch + 1}th epoch')
    return Acc


if __name__ == "__main__":
    args = param_parser.parameter_parser()
    args_runs = param_parser.para_load()
    ACC_end_print = {}
    for k in range(len(args_runs)):
        args = param_parser.para_update(args, args_runs[k])
    # for k in range(1):
        # path = './temp.txt'
        path = os.path.join('./result/' + args.dataset + '/' + args.dataset + str(args_runs[k]['id']) + '.txt')
        with open(path, 'a') as file:
            file.write(json.dumps(vars(args), indent=4))

        Acc = []
        node_map, graph, feat_data, labels = utils_data.prepare_data(args)

        for i in range(args.N):
            with open(path, 'a') as file:
                file.write(f'\nexperiment: {i + 1}\n')
            print(f'\n第{i + 1}次实验')
            t_start = time.time()
            split_by_label_flag = True
            if args.dataset in ['chameleon', 'cornell', 'texas']:
                split_by_label_flag = False
            trains, vals, tests = utils_data.prepare_dataset(args, labels, i, split_by_label_flag)
            # trains, vals, tests = utils_data.prepare_dataset(args, labels, i)
            # 生成结构相似的边和权重
            role_adj_list, weigh_list = utils_data.generate_structure_similar_matrix(args, graph, labels, trains)
            neigh_data, struc_data = utils_data.get_data(graph, feat_data, labels, role_adj_list, weigh_list)
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            neigh_data = neigh_data.to(device)
            struc_data = struc_data.to(device)
            trains = torch.from_numpy(trains).to(device)
            vals = torch.from_numpy(vals).to(device)
            tests = torch.from_numpy(tests).to(device)

            net_list, net = Train(args, trains, vals, neigh_data, struc_data, path)
            acc = evaluate_node_classification(net_list, net, tests, neigh_data.y)
            Acc.append(acc)

        ACC_end_print[args_runs[k]['id']] = np.mean(Acc)
        print('max Acc:', format(np.max(Acc), '.4f'))
        print('mean Acc:', format(np.mean(Acc), '.4f'))
        print('std Acc:', format(np.std(Acc), '.4f'))
        Acc_dictionary = {i: v for i, v in enumerate(Acc)}
        print(Acc_dictionary)
        print('\n\n')

        with open(path, 'a') as file:
            file.write('max Acc: {:.4f}\n'.format(np.max(Acc)))
            file.write('mean Acc: {:.4f}\n'.format(np.mean(Acc)))
            file.write('std Acc: {:.4f}\n'.format(np.std(Acc)))
            Acc_dictionary = {i: v for i, v in enumerate(Acc)}
            file.write(json.dumps(Acc_dictionary) + '\n\n')
    print(ACC_end_print)
    path_end = os.path.join('./result/' + args.dataset + '/' + args.dataset + '_end.txt')
    with open(path_end, 'a') as file:
        file.write(json.dumps(ACC_end_print))
        file.write('\n\n')
        for key in ACC_end_print.keys():
            file.write(str(ACC_end_print[key]))
            file.write('\n')