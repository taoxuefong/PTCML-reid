import torch
import torch.nn as nn
import torch.nn.functional as F


def _mmd_rbf(x, y, sigma=None):
    """Lightweight RBF MMD for ACM distribution constraint. x,y: (n,d) tensors."""
    n, m = x.size(0), y.size(0)
    if n < 2 or m < 2:
        return torch.tensor(0., device=x.device)
    if x.dim() == 1:
        x = x.unsqueeze(1)
    if y.dim() == 1:
        y = y.unsqueeze(1)
    xx = ((x.unsqueeze(1) - x.unsqueeze(0)) ** 2).sum(2)
    yy = ((y.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(2)
    xy = ((x.unsqueeze(1) - y.unsqueeze(0)) ** 2).sum(2)
    if sigma is None:
        sigma = (xx.sum() + yy.sum()).clamp(min=1e-6) / max(1, n * n - n + m * m - m)
        sigma = sigma.clamp(min=1e-6)
    kxx = torch.exp(-xx / sigma).mean()
    kyy = torch.exp(-yy / sigma).mean()
    kxy = torch.exp(-xy / sigma).mean()
    return kxx + kyy - 2 * kxy


class TMCLS(nn.Module):
    """ TMC Label Smoothing: calibrating patch features to smooth label distributions """
    def __init__(self):
        super(TMCLS, self).__init__()
        self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

    def forward(self, logits, targets, rts_score):
        log_preds = self.logsoftmax(logits)  # B * C
        targets = torch.zeros_like(log_preds).scatter_(1, targets.unsqueeze(1), 1)
        uni = (torch.ones_like(log_preds) / log_preds.size(-1)).cuda()

        loss_ce = (- targets * log_preds).sum(1)
        loss_kld = F.kl_div(log_preds, uni, reduction='none').sum(1)
        loss = (rts_score * loss_ce + (1-rts_score) * loss_kld).mean()
        return loss


class TMC(nn.Module):
    """ Tendency-based Mutual Complementation: refine global and patch pseudo-labels via RTS """
    def __init__(self, lam=0.5):
        super(TMC, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.lam = lam

    def forward(self, logits_g, logits_p, targets, rts_score):
        targets = torch.zeros_like(logits_g).scatter_(1, targets.unsqueeze(1), 1)
        w = torch.softmax(rts_score, dim=1)  # B * P
        w = torch.unsqueeze(w, 1)  # B * 1 * P
        preds_p = self.softmax(logits_p)  # B * C * P
        ensembled_preds = (preds_p * w).sum(2).detach()  # B * class_num
        refined_targets = self.lam * targets + (1-self.lam) * ensembled_preds

        log_preds_g = self.logsoftmax(logits_g)
        loss = (-refined_targets * log_preds_g).sum(1).mean()
        return loss


class ACMProxy(nn.Module):
    """ Adaptive Camera Multi-Constraint: camera distribution + instance constraints """
    def __init__(self, num_features, num_samples, num_hards=50, temp=0.07,
                 lam_acm_dis=0.05, lam_acm_ins=0.05, acm_gamma=0.9, acm_nk=5, max_cams=8):
        super(ACMProxy, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples
        self.num_hards = num_hards
        self.logsoftmax = nn.LogSoftmax(dim=0)
        self.temp = temp
        # ACM
        self.lam_acm_dis = lam_acm_dis
        self.lam_acm_ins = lam_acm_ins
        self.acm_gamma = acm_gamma
        self.acm_nk = acm_nk
        self.max_cams = max_cams
        self.register_buffer('proxy', torch.zeros(num_samples, num_features))
        self.register_buffer('pids', torch.zeros(num_samples).long())
        self.register_buffer('cids', torch.zeros(num_samples).long())
        # D_{ci,cj}: 相机间平均匹配距离 (Eq.15)，初始 1 避免除零
        self.register_buffer('D_cam', torch.ones(max_cams, max_cams))

    def forward(self, inputs, targets, cams):
        B, D = inputs.shape
        inputs = F.normalize(inputs, dim=1)
        sims = inputs @ self.proxy.T
        sims /= self.temp
        temp_sims = sims.detach().clone()

        loss = torch.tensor(0., device=inputs.device)
        for i in range(B):
            pos_mask = (targets[i] == self.pids).float() * (cams[i] != self.cids).float()
            neg_mask = (targets[i] != self.pids).float()
            pos_idx = torch.nonzero(pos_mask > 0).squeeze(-1)
            if len(pos_idx) == 0:
                continue
            hard_neg_idx = torch.sort(temp_sims[i] + (-9999999.) * (1. - neg_mask), descending=True).indices[:self.num_hards]
            sims_i = sims[i, torch.cat([pos_idx, hard_neg_idx])]
            targets_i = torch.zeros(len(sims_i), device=inputs.device)
            targets_i[:len(pos_idx)] = 1.0 / len(pos_idx)
            loss += - (targets_i * self.logsoftmax(sims_i)).sum()

        loss = loss / B

        # ACM: 仅在 batch 内有多相机时启用
        unique_cams = cams.unique()
        if unique_cams.size(0) >= 2 and self.lam_acm_dis > 0:
            loss = loss + self.lam_acm_dis * self._acm_dis(inputs, cams)
        if unique_cams.size(0) >= 2 and self.lam_acm_ins > 0:
            loss = loss + self.lam_acm_ins * self._acm_ins(inputs, targets, cams)

        return loss

    def _acm_dis(self, inputs, cams):
        """ 分布约束: MMD( intra-camera distance dist, inter-camera distance dist ) (Eq.12-14) """
        B = inputs.size(0)
        dists_intra, dists_inter = [], []
        for c in cams.unique():
            mask_c = (cams == c)
            if mask_c.sum() < 2:
                continue
            feats_c = inputs[mask_c]
            pw = (feats_c.unsqueeze(1) - feats_c.unsqueeze(0)).norm(p=2, dim=2)
            triu = torch.triu_indices(pw.size(0), pw.size(1), 1, device=pw.device)
            d = pw[triu[0], triu[1]]
            dists_intra.append(d)
        for i in range(B):
            for j in range(i + 1, B):
                if cams[i] != cams[j]:
                    d = (inputs[i] - inputs[j]).norm(p=2)
                    dists_inter.append(d.unsqueeze(0))
        if len(dists_intra) == 0 or len(dists_inter) == 0:
            return torch.tensor(0., device=inputs.device)
        x = torch.cat(dists_intra).unsqueeze(1)
        y = torch.cat(dists_inter).unsqueeze(1)
        return _mmd_rbf(x, y)

    def _acm_ins(self, inputs, targets, cams):
        """ 实例约束: 用相机距离矩阵缩放后做 triplet-style loss (Eq.15-17) """
        B = inputs.size(0)
        uc = cams.unique()
        nc = min(uc.size(0), self.max_cams)
        cid2idx = {int(c.item()): i for i, c in enumerate(uc[:self.max_cams])}

        # 到 proxy 的距离 (1-cos 作为 proxy，归一化向量下与 L2 单调一致)
        dist_to_proxy = (1.0 - (inputs @ self.proxy.T)).clamp(min=1e-8)

        # 更新 D 矩阵 (Eq.15)：对每个 (ci,cj) 取 batch 内平均后更新
        with torch.no_grad():
            for ic, ci in enumerate(uc[:nc]):
                mask_ci = (cams == ci)
                if mask_ci.sum() == 0:
                    continue
                # D[ic,ic]: 同相机内平均距离
                feats_ci = inputs[mask_ci]
                if feats_ci.size(0) >= 2:
                    pw = (feats_ci.unsqueeze(1) - feats_ci.unsqueeze(0)).norm(p=2, dim=2)
                    triu = torch.triu_indices(pw.size(0), pw.size(1), 1, device=pw.device)
                    intra_mean = pw[triu[0], triu[1]].mean()
                    self.D_cam[ic, ic] = self.acm_gamma * self.D_cam[ic, ic] + (1 - self.acm_gamma) * intra_mean.clamp(min=1e-6)
                for jc, cj in enumerate(uc[:nc]):
                    if ci == cj:
                        continue
                    proxy_cj = (self.cids == cj)
                    if proxy_cj.sum() == 0:
                        continue
                    vals = []
                    for idx in mask_ci.nonzero(as_tuple=False).squeeze(-1):
                        idx = int(idx)
                        k = min(self.acm_nk, int(proxy_cj.sum()))
                        d, _ = torch.topk(dist_to_proxy[idx, proxy_cj], k, largest=False)
                        vals.append(d.mean())
                    if len(vals) > 0:
                        mean_d = torch.stack(vals).mean()
                        self.D_cam[ic, jc] = self.acm_gamma * self.D_cam[ic, jc] + (1 - self.acm_gamma) * mean_d.clamp(min=1e-6)
            D = self.D_cam[:nc, :nc].clamp(min=1e-6)

        # 实例损失：hardest pos/neg，缩放后 margin triplet (Eq.16-17)
        loss_ins = torch.tensor(0., device=inputs.device)
        count = 0
        dist_raw = (1.0 - (inputs @ inputs.T)).clamp(min=1e-8)
        for i in range(B):
            ic = cid2idx.get(int(cams[i].item()), -1)
            if ic < 0:
                continue
            pos_mask = (targets == targets[i]) & (cams != cams[i])
            neg_mask = (targets != targets[i])
            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue
            pos_idx = pos_mask.nonzero(as_tuple=False).squeeze(-1)
            neg_idx = neg_mask.nonzero(as_tuple=False).squeeze(-1)
            hard_pos = pos_idx[dist_raw[i, pos_idx].argmin()]
            hard_neg = neg_idx[dist_raw[i, neg_idx].argmax()]
            jc_pos = cid2idx.get(int(cams[hard_pos].item()), ic)
            jc_neg = cid2idx.get(int(cams[hard_neg].item()), ic)
            scale_pos = (D[ic, ic] / D[ic, jc_pos]).clamp(0.1, 10.0)
            scale_neg = (D[ic, ic] / D[ic, jc_neg]).clamp(0.1, 10.0)
            d_pos = dist_raw[i, hard_pos] * scale_pos
            d_neg = dist_raw[i, hard_neg] * scale_neg
            loss_ins = loss_ins + F.relu(d_pos - d_neg + 0.2)
            count += 1
        if count > 0:
            loss_ins = loss_ins / count
        return loss_ins
