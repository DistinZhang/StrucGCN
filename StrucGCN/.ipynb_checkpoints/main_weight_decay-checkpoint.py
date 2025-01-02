import torch
import param_parser
import numpy as np
import time
import torch.nn.functional as F
from copy import deepcopy
import utils_data
from torch import tensor
import os
import json
import utils
from model import StrucGNN
# from model_with_residual import StrucGNN

def Train(args, trains, vals, neigh_data, struc_data, merged_data, path):
    net = StrucGNN(args, neigh_data, struc_data, merged_data)
    net = net.to(device)
    if neigh_data is None:
        data = merged_data
    else:
        data = neigh_data
    epoch_time = []

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 早停法参数
    early_stopping_patience = args.early_stopping_patience
    best_val_acc = float('-inf')
    best_loss_val = float('inf')
    best_train_acc = 0
    patience_counter = 0
    better_net_list = {}
    val_loss_history = []
    for epoch in range(args.EPOCHS):
        start = time.time()
        net.train()
        logp = net()
        cla_loss = F.nll_loss(logp[trains], data.y[trains])
        loss = cla_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end = time.time() - start

        net.eval()  # 设置模型为评估模式
        logp = net()
        with torch.no_grad():
            acc = utils.accuracy(logp[vals], data.y[vals])
            train_acc = utils.accuracy(logp[trains], data.y[trains])
            loss_val = F.nll_loss(logp[vals], data.y[vals])

        if acc > best_val_acc and loss_val < best_loss_val:
        # if acc > best_val_acc:
            best_val_acc = acc
            best_train_acc = train_acc
            best_loss_val = loss_val
            patience_counter = 0
            better_net_list[0] = deepcopy(net.state_dict())  # 保存最佳模型的状态
        else:
            # if (epoch + 1) % 10 == 0:
            #     better_net_list[epoch] = deepcopy(net.state_dict())
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping! {epoch}')
                break
        # val_loss_history.append(loss_val)
        # if early_stopping_patience > 0 and epoch > args.EPOCHS // 2:
        #     tmp = tensor(val_loss_history[-(early_stopping_patience + 1):-1])
        #     if loss_val > tmp.mean().item():
        #         print(f'############### {epoch}')
        #         break
        # 打印损失信息（可选）
        if (epoch + 1) % 100 == 0:
            print(
                f'Epoch [{epoch + 1}/{args.EPOCHS}], Train Loss: {loss.item()}, Train acc: {train_acc}, Val acc: {acc}, time/epoch:{end}')
        epoch_time.append(end)
        
    better_net_list[epoch] = deepcopy(net.state_dict())
    with open(path, 'a') as file:
        file.write(f'Epoch [{epoch + 1}/{args.EPOCHS}], Train Loss: {loss.item()}, Val acc: {acc}')
        file.write(f'\naverage epoch time={np.mean(epoch_time[3:])}\n\n')
    return better_net_list, net, np.mean(epoch_time[3:])


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
    Time_end_print = {}
    std = {}
    edge_h = {}
    node_h = {}
    path_end = 'end.txt'
    for k in range(len(args_runs)):
        args = param_parser.para_update(args, args_runs[k])
    # for k in range(1):
        # path = './temp.txt'
        # path = os.path.join('./result/' + args.dataset + '/' + args.dataset + str(args_runs[k]['id']) + '.txt')
        path = os.path.join('./result/' + args.dataset + '_' + str(args_runs[k]['id']) + '.txt')
        with open(path, 'a') as file:
            file.write(json.dumps(vars(args), indent=4))
        utils.set_seed(args.seed)
        Acc = []
        Time = []
        node_map, graph, feat_data, labels = utils_data.prepare_data(args)

        for i in range(args.N):
            with open(path, 'a') as file:
                file.write(f'\nexperiment: {i + 1}\n')
            print(f'\n第{i + 1}次实验')
            t_start = time.time()
            split_by_label_flag = True
            if args.dataset in ['chameleon', 'cornell', 'texas']:
                split_by_label_flag = False
            # trains, vals, tests = utils_data.prepare_dataset(args, labels, i, split_by_label_flag)
            trains, vals, tests = utils_data.prepare_dataset(args, labels, i)
            # 生成结构相似的边和权重
            # role_adj_list, weigh_list = utils_data.generate_structure_similar_matrix(args, graph, labels, trains) #strucgnn
            role_adj_list, weigh_list = utils_data.generate_structure_similar_matrix(args, graph, labels, trains, i) #gcnsa\nlgnn
            neigh_data, struc_data = utils_data.get_data(graph, feat_data, labels, role_adj_list, weigh_list)
            merged_data = utils_data.get_merged_data(args, neigh_data, struc_data)
            device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
            neigh_data = neigh_data.to(device)
            struc_data = struc_data.to(device)
            merged_data = merged_data.to(device)
            trains = torch.from_numpy(trains).to(device)
            vals = torch.from_numpy(vals).to(device)
            tests = torch.from_numpy(tests).to(device)
            net_list, net, t = Train(args, trains, vals, neigh_data, struc_data, merged_data, path)
            acc = evaluate_node_classification(net_list, net, tests, neigh_data.y)
            print(f'Test acc: {acc}')
            Acc.append(acc)
            Time.append(t)

        ACC_end_print[args_runs[k]['id']] = np.mean(Acc)
        std[args_runs[k]['id']] = np.std(Acc)
        Time_end_print[args_runs[k]['id']] = np.mean(Time)
        neigh_edge_h = utils_data.calculate_edge_homophily(neigh_data)
        struc_edge_h = utils_data.calculate_edge_homophily(struc_data)
        merged_edge_h = utils_data.calculate_edge_homophily(merged_data)
        edge_h[args_runs[k]['id']] = [neigh_edge_h, struc_edge_h, merged_edge_h]
        neigh_node_h = utils_data.calculate_node_homophily(neigh_data)
        struc_node_h = utils_data.calculate_node_homophily(struc_data)
        merged_node_h = utils_data.calculate_node_homophily(merged_data)
        node_h[args_runs[k]['id']] = [neigh_node_h, struc_node_h, merged_node_h]
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
    # path_end = os.path.join('./result/' + args.dataset + '/' + args.dataset + '_end.txt')
    
    with open(path_end, 'a') as file:
        file.write(json.dumps(ACC_end_print))
        file.write('\n\n')
        for key in ACC_end_print.keys():
            file.write(str(ACC_end_print[key]))
            file.write('\n')
            
        file.write('\n')
        
        file.write(json.dumps(std))
        file.write('\n\n')
        for key in std.keys():
            file.write(str(std[key]))
            file.write('\n')
            
        file.write('\n')
        
        file.write(json.dumps(Time_end_print))
        file.write('\n\n')
        for key in Time_end_print.keys():
            file.write(str(Time_end_print[key]))
            file.write('\n')
            
        file.write('\n')
        
        file.write(json.dumps(edge_h))
        file.write('\n\n')
        for key in edge_h.keys():
            file.write(str(edge_h[key]))
            file.write('\n')
            
        file.write('\n')

        file.write(json.dumps(node_h))
        file.write('\n\n')
        for key in node_h.keys():
            file.write(str(node_h[key]))
            file.write('\n')