from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import random
import numpy as np
import sys
import time
import os
# 若某张卡报错 (如 nvidia-smi 显示 ERR!)，可去掉该卡，例如只用 5,6,7
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.cluster import DBSCAN

import torch
import torch.nn.functional as F
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader

from ptcml import datasets
from ptcml.models import resnet50part
from ptcml.loss import ACMProxy
from ptcml.trainers import PTCMLTrainerCAM
from ptcml.evaluators import Evaluator, extract_all_features
from ptcml.utils.data import IterLoader
from ptcml.utils.data import transforms as T
from ptcml.utils.data.sampler import RandomMultipleGallerySampler
from ptcml.utils.data.preprocessor import Preprocessor
from ptcml.utils.logging import Logger
from ptcml.utils.faiss_rerank import compute_ranked_list, compute_jaccard_distance
from maximum_mean_discrepancy import mmd_loss

best_mAP = 0


def get_data(name, data_dir):
    root = data_dir
    dataset = datasets.create(name, root)
    return dataset


def get_train_loader(dataset, height, width, batch_size, workers,
                     num_instances, iters, trainset=None):

    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    train_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.RandomHorizontalFlip(p=0.5),
             T.Pad(10),
             T.RandomCrop((height, width)),
             T.ToTensor(),
             normalizer,
             T.RandomErasing(probability=0.5, mean=[0.485, 0.456, 0.406])
         ])

    train_set = sorted(dataset.train) if trainset is None else sorted(trainset)
    rmgs_flag = num_instances > 0
    if rmgs_flag:
        sampler = RandomMultipleGallerySampler(train_set, num_instances)
    else:
        sampler = None
    train_loader = IterLoader(
                DataLoader(Preprocessor(train_set, root=dataset.images_dir, transform=train_transformer),
                           batch_size=batch_size, num_workers=workers, sampler=sampler,
                           shuffle=not rmgs_flag, pin_memory=True, drop_last=True), length=iters)

    return train_loader


def get_test_loader(dataset, height, width, batch_size, workers, testset=None):
    normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])

    test_transformer = T.Compose([
             T.Resize((height, width), interpolation=3),
             T.ToTensor(),
             normalizer
         ])

    if (testset is None):
        testset = list(set(dataset.query) | set(dataset.gallery))

    test_loader = DataLoader(
        Preprocessor(testset, root=dataset.images_dir, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return test_loader


def compute_pseudo_labels(features, cluster, k1):
    mat_dist = compute_jaccard_distance(features, k1=k1, k2=6)
    ids = cluster.fit_predict(mat_dist)
    num_ids = len(set(ids)) - (1 if -1 in ids else 0)

    labels = []
    outliers = 0
    for i, id in enumerate(ids):
        if id != -1:
            labels.append(id)
        else:
            labels.append(num_ids + outliers)
            outliers += 1

    return torch.Tensor(labels).long().detach(), num_ids


def compute_rts_score(features_g, features_p, k, search_option=0):
    """ Ranking Tendency Similarity: gauging distribution discrepancy of distance changes """
    print("Compute RTS score...")
    N, D, P = features_p.size()
    # N x P，存放每个样本与每个局部的“排序一致性 / 互补性”得分
    score = torch.zeros(N, P)
    end = time.time()
    ranked_list_g = compute_ranked_list(features_g, k=k, search_option=search_option, verbose=False)

    for i in range(P):
        ranked_list_p_i = compute_ranked_list(features_p[:, :, i], k=k, search_option=search_option, verbose=False)
        # 针对每个样本，既考虑集合交并（全局 / 局部邻居是否相同），
        # 也通过 MMD 度量两者“距离递增趋势”的分布差异
        for j in range(N):
            neigh_g = ranked_list_g[j]
            neigh_p = ranked_list_p_i[j]

            inter = np.intersect1d(neigh_g, neigh_p)
            union = np.union1d(neigh_g, neigh_p)

            # 集合层面的相似度（Jaccard 交并比）
            if len(union) == 0:
                jaccard = 0.0
            else:
                jaccard = float(len(inter)) / float(len(union))

            # 使用邻居之间的 L2 距离增量构造“趋势”向量，并利用 MMD 度量其分布差异
            # 为了稳定和高效，这里直接在 top-k 邻居上计算（长度约为 k-1）
            if len(neigh_g) > 1 and len(neigh_p) > 1:
                # 将索引转换为 LongTensor，以便在特征上进行索引
                idx_g = torch.LongTensor(neigh_g)
                idx_p = torch.LongTensor(neigh_p)

                # 取出对应的全局 / 局部特征序列：形状均为 k x D
                feats_g_neighbors = features_g.index_select(0, idx_g)
                feats_p_neighbors = features_p.index_select(0, idx_p)[:, :, i]

                # 相邻样本间的距离增量向量（长度 k-1）
                diff_g = feats_g_neighbors[:-1] - feats_g_neighbors[1:]
                diff_p = feats_p_neighbors[:-1] - feats_p_neighbors[1:]
                T_g = torch.norm(diff_g, p=2, dim=1)  # (k-1,)
                T_p = torch.norm(diff_p, p=2, dim=1)  # (k-1,)

                # 调用提供的 MMD 接口，度量两个一维分布的差异
                # 注意：该实现内部使用 CUDA，这里将小批量向量拷贝到 GPU 上，计算后再取回
                T_g_cuda = T_g.view(-1, 1).cuda()
                T_p_cuda = T_p.view(-1, 1).cuda()
                mmd_val = mmd_loss(T_g_cuda, T_p_cuda)
                mmd_val = float(mmd_val.detach().cpu().item())

                # 将 MMD 差异映射为 [0,1] 的“趋势相似度”：差异越大，相似度越小
                tendency_sim = 1.0 / (1.0 + mmd_val)
            else:
                tendency_sim = 0.0

            # 最终的 TMC 风格得分：集合一致性（结构一致） + 趋势相似性（分布一致）
            score[j, i] = 0.5 * jaccard + 0.5 * tendency_sim

    print("RTS score time cost: {}".format(time.time() - end))
    return score


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    main_worker(args)


def main_worker(args):
    global best_mAP

    cudnn.benchmark = True

    sys.stdout = Logger(osp.join(args.logs_dir, 'log.txt'))
    print("==========\nArgs:{}\n==========".format(args))

    # dataset
    dataset = get_data(args.dataset, args.data_dir)
    test_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers)
    cluster_loader = get_test_loader(dataset, args.height, args.width, args.batch_size, args.workers,
                                     testset=sorted(dataset.train))

    # model
    num_part = args.part
    model = resnet50part(num_parts=args.part, num_classes=3000)
    model.cuda()
    model = nn.DataParallel(model)

    # evaluator
    evaluator = Evaluator(model)

    # optimizer
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        params += [{"params": [value], "lr": args.lr, "weight_decay": args.weight_decay}]
    optimizer = torch.optim.Adam(params)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)

    score_log = torch.FloatTensor([])
    for epoch in range(args.epochs):
        features_g, features_p, _ = extract_all_features(model, cluster_loader)
        features_g = torch.cat([features_g[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)
        features_p = torch.cat([features_p[f].unsqueeze(0) for f, _, _ in sorted(dataset.train)], 0)

        if epoch == 0:
            cluster = DBSCAN(eps=args.eps, min_samples=4, metric='precomputed', n_jobs=8)

        # assign pseudo-labels
        pseudo_labels, num_class = compute_pseudo_labels(features_g, cluster, args.k1)

        # Compute RTS (Ranking Tendency Similarity) score
        rts_score = compute_rts_score(features_g, features_p, k=args.k)
        score_log = torch.cat([score_log, rts_score.unsqueeze(0)], dim=0)

        # generate new dataset with pseudo-labels
        num_outliers = 0
        new_dataset = []

        idxs, cids, pids = [], [], []
        for i, ((fname, _, cid), label) in enumerate(zip(sorted(dataset.train), pseudo_labels)):
            pid = label.item()
            if pid >= num_class:  # append data except outliers
                num_outliers += 1
            else:
                new_dataset.append((fname, pid, cid))
                idxs.append(i)
                cids.append(cid)
                pids.append(pid)

        train_loader = get_train_loader(dataset, args.height, args.width, args.batch_size,
                                        args.workers, args.num_instances, args.iters, trainset=new_dataset)

        # statistics of clusters and un-clustered instances
        print('==> Statistics for epoch {}: {} clusters, {} un-clustered instances'.format(epoch, num_class,
                                                                                           num_outliers))

        # reindex
        idxs, cids, pids = np.asarray(idxs), np.asarray(cids), np.asarray(pids)
        features_g = features_g[idxs, :]
        features_p = features_p[idxs, :, :]
        rts_score = rts_score[idxs, :]

        # compute cluster centroids and camera-aware proxies
        centroids_g, centroids_p = [], []
        cam_proxy, cam_proxy_p, cam_proxy_pids, cam_proxy_cids = [], [], [], []
        for pid in sorted(np.unique(pids)):  # loop all pids
            idxs_p = np.where(pids == pid)[0]
            centroids_g.append(features_g[idxs_p].mean(0))
            centroids_p.append(features_p[idxs_p].mean(0))

            for cid in sorted(np.unique(cids[idxs_p])):  # loop all cids for pid
                idxs_c = np.where(cids == cid)[0]
                idxs_cp = np.intersect1d(idxs_p, idxs_c)
                cam_proxy.append(features_g[idxs_cp].mean(0))
                cam_proxy_p.append(features_p[idxs_cp].mean(0))
                cam_proxy_pids.append(pid)
                cam_proxy_cids.append(cid)

        centroids_g = F.normalize(torch.stack(centroids_g), p=2, dim=1)
        model.module.classifier.weight.data[:num_class].copy_(centroids_g)
        memory = ACMProxy(centroids_g.size(1), len(cam_proxy_pids)).cuda()
        memory.proxy = F.normalize(torch.stack(cam_proxy), p=2, dim=1).cuda()
        memory.pids = torch.Tensor(cam_proxy_pids).long().cuda()
        memory.cids = torch.Tensor(cam_proxy_cids).long().cuda()

        memory_p = []
        for i in range(num_part):
            centroids_p_i = torch.stack(centroids_p)[:, :, i]
            centroids_p_i = F.normalize(centroids_p_i, p=2, dim=1)
            classifier_p_i = getattr(model.module, 'classifier' + str(i))
            classifier_p_i.weight.data[:num_class].copy_(centroids_p_i)

            memory_p_i = ACMProxy(centroids_g.size(1), len(cam_proxy_pids)).cuda()
            cam_proxy_p_i = torch.stack(cam_proxy_p)[:, :, i]
            memory_p_i.proxy = F.normalize(cam_proxy_p_i, p=2, dim=1).cuda()
            memory_p_i.pids = torch.Tensor(cam_proxy_pids).long().cuda()
            memory_p_i.cids = torch.Tensor(cam_proxy_cids).long().cuda()
            memory_p.append(memory_p_i)

        # training
        trainer = PTCMLTrainerCAM(model, rts_score, memory, memory_p, num_class=num_class, num_part=num_part,
                                 beta=args.beta, tmcls_epoch=args.tmcls_epoch, lam_cam=args.lam_cam)

        trainer.train(epoch, train_loader, optimizer, print_freq=args.print_freq, train_iters=len(train_loader))
        lr_scheduler.step()

        # evaluation
        if ((epoch+1) % args.eval_step == 0) or (epoch == args.epochs-1):
            mAP = evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=False)

            if mAP > best_mAP:
                best_mAP = mAP
                torch.save(model.state_dict(), osp.join(args.logs_dir, 'best.pth'))
            print('\n* Finished epoch {:3d}  model mAP: {:5.1%} best: {:5.1%}\n'.format(epoch, mAP, best_mAP))

    torch.save(model.state_dict(), osp.join(args.logs_dir, 'last.pth'))
    np.save(osp.join(args.logs_dir, 'scores.npy'), score_log.numpy())

    # results
    model.load_state_dict(torch.load(osp.join(args.logs_dir, 'best.pth')))
    evaluator.evaluate(test_loader, dataset.query, dataset.gallery, cmc_flag=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PTCML with ACM: Patch-based Tendency Camera Multi-Constraint Learning")
    # data
    parser.add_argument('-d', '--dataset', type=str, default='market1501')
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-n', '--num-instances', type=int, default=4,
                        help="each minibatch consist of "
                             "(batch_size // num_instances) identities, and "
                             "each identity has num_instances instances, "
                             "default: 0 (NOT USE)")
    parser.add_argument('--height', type=int, default=384, help="input height")
    parser.add_argument('--width', type=int, default=128, help="input width")

    # path
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH', default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs/test'))

    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--eval-step', type=int, default=5)

    # PTCML
    parser.add_argument('--part', type=int, default=3, help="number of part")
    parser.add_argument('--k', type=int, default=20,
                        help="hyperparameter for RTS score")
    parser.add_argument('--beta', type=float, default=0.5,
                        help="weighting parameter for TMC loss")
    parser.add_argument('--tmcls-epoch', type=int, default=5,
                        help="starting epoch for TMC label smoothing")
    parser.add_argument('--lam-cam', type=float, default=0.5,
                        help="weighting parameter of inter-camera contrastive loss")

    # optimizer
    parser.add_argument('--lr', type=float, default=0.00035, help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--iters', type=int, default=400)
    parser.add_argument('--step-size', type=int, default=20)

    # cluster
    parser.add_argument('--k1', type=int, default=30,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--k2', type=int, default=6,
                        help="hyperparameter for jaccard distance")
    parser.add_argument('--eps', type=float, default=0.5,
                        help="distance threshold for DBSCAN")

    main()
