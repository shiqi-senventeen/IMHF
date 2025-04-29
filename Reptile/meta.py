#!/usr/bin/env python3
import torch
from torch import nn
from torch import optim
from torch import autograd
import torch.nn.functional as F
import numpy as np

from Module.datautils import args
from utils import recall, precision, mcc, roc, prc_auc, accuracy, f1


class Learner(nn.Module):
    """

    This is a learner class, which will accept a specific network module, such as OmniNet that define the network forward
    process. Learner class will create two same network, one as theta network and the other acts as theta_pi network.
    for each episode, the theta_pi network will copy its initial parameters from theta network and update several steps
    by meta-train set and then calculate its loss on meta-test set. All loss on meta-test set will be sumed together and
    then backprop on theta network, which should be done on metalaerner class.
    For learner class, it will be responsible for update for several steps on meta-train set and return with the loss on
    meta-test set.
    """

    def __init__(self, net_cls, *args):
        """
		It will receive a class: net_cls and its parameters: args for net_cls.
		:param net_cls: class, not instance
		:param args: the parameters for net_cls
		"""
        super(Learner, self).__init__()
        # pls make sure net_cls is a class but NOT an instance of class.
        assert net_cls.__class__ == type

        # we will create two class instance meanwhile and use one as theta network and the other as theta_pi network.
        self.net = net_cls(*args)
        # you must call create_pi_net to create pi network additionally
        self.net_pi = net_cls(*args)
        # update theta_pi = theta_pi - lr * grad
        # according to the paper, here we use naive version of SGD to update theta_pi
        # 0.1 here means the learner_lr

        # self.optimizer = optim.SGD(self.net_pi.parameters(), 0.001)

        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net_pi.parameters()), lr=0.001)

        # for net in [self.net, self.net_pi]:
        #     for name, param in net.named_parameters():
        #         if 'cnn' in name:
        #             param.requires_grad = False


    def parameters(self):
        """
        		Override this function to return only net parameters for MetaLearner's optimize
        		it will ignore theta_pi network parameters.
        		:return:
        """
        return self.net.parameters()

    def update_pi(self):
        for m_from, m_to in zip(self.net.modules(), self.net_pi.modules()):
            if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

    def forward(self, support_x_atom, support_x_bond, support_x_mask, support_x_img,
                support_y,
                query_x_atom, query_x_bond, query_x_mask, query_x_img, query_y, num_updates):
        """
		learn on current episode meta-train: support_x & support_y and then calculate loss on meta-test set: query_x&y
		:param support_x: [setsz, c_, h, w]
		:param support_y: [setsz]
		:param query_x:   [querysz, c_, h, w]
		:param query_y:   [querysz]
		:param num_updates: 5
		:return:
		"""
        # now try to fine-tune from current $theta$ parameters -> $theta_pi$
        # after num_updates of fine-tune, we will get a good theta_pi parameters so that it will retain satisfying
        # performance on specific task, that's, current episode.
        # firstly, copy theta_pi from theta network

        self.update_pi()

        # update for several steps
        for i in range(num_updates):
            # forward and backward to update net_pi grad.
            loss, pred, _ = self.net_pi(support_x_atom, support_x_bond, support_x_mask, support_x_img,
                                        support_y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Compute the meta gradient and return it, the gradient is from one episode
        # in metalearner, it will merge all loss from different episode and sum over it.

        loss, pred, _ = self.net_pi(query_x_atom, query_x_bond, query_x_mask, query_x_img, query_y)

        # _, indices = torch.max(pred, dim=1)
        # correct = torch.eq(indices, query_y).sum().item()
        # acc = correct / query_y.size(0)
        # pred_ = F.softmax(pred, dim=-1).data.cpu().numpy()[:, 1]
        pred_ = pred.data.cpu().numpy()
        query_y_ = query_y.cpu().detach().numpy()
        query_y_ = [int(x) for x in query_y_]
        acc = accuracy(query_y_, pred_)
        pre_score = precision(query_y_, pred_)
        recall_score = recall(query_y_, pred_)
        mcc_score = mcc(query_y_, pred_)
        roc_score = roc(query_y_, pred_)
        f1_score = f1(query_y_, pred_)

        # gradient for validation on theta_pi
        # after call autorad.grad, you can not call backward again except for setting create_graph = True
        # as we will use the loss as dummpy loss to conduct a dummy backprop to write our gradients to theta network,
        # here we set create_graph to true to support second time backward.

        # grads_pi = autograd.grad(loss, self.net_pi.parameters(), create_graph=True,allow_unused=True)

        # grads_pi = autograd.grad(loss, self.net_pi.parameters(), create_graph=True)
        params_pi = [p for p in self.net_pi.parameters() if p.requires_grad]
        grads_pi = autograd.grad(loss, params_pi, create_graph=True)

        return loss, grads_pi, (acc, pre_score, recall_score, mcc_score, roc_score, f1_score)

    def net_forward(self, support_x_atom, support_x_bond, support_x_mask, support_x_img,
                    support_y):
        """
		This function is purely for updating net network. In metalearner, we need the get the loss op from net network
		to write our merged gradients into net network, hence will call this function to get a dummy loss op.
		:param support_x: [setsz, c, h, w]
		:param support_y: [sessz, c, h, w]
		:return: dummy loss and dummy pred
		"""
        loss, pred, _ = self.net(support_x_atom, support_x_bond, support_x_mask, support_x_img,
                                 support_y)
        return loss, pred


class MetaLearner(nn.Module):
    """
    The whole network, the training process is done in the network, should include forward propagation and back propagation
    learner is the metamodel
    optimizer Indicates the optimizer
    As we have mentioned in Learner class, the metalearner class will receive a series of loss on different tasks/episodes
    on theta_pi network, and it will merage all loss and then sum over it. The summed loss will be backproped on theta
    network to update theta parameters, which is the initialization point we want to find.
    The MetaLearner class accepts and adds the sum of the loss of the task model on different tasks, and this loss is used to optimize the metamodel
    """

    def __init__(self, net_cls, net_cls_args, n_way=args.n_way, k_shot=args.k_shot, k_query=args.k_query,
                 meta_batchsz=args.meta_batchsz, meta_lr=args.meta_lr, num_updates=args.num_updates):
        """

		:param net_cls: class, not instance. the class of specific Network for learner
		:param net_cls_args: tuple, args for net_cls, like (n_way, imgsz)

		:param n_way:
		:param k_shot:
		:param meta_batchsz:number of tasks/episode
		:param meta_lr: learning rate for meta-learner
		:param num_updates: number of updates for learner
		"""
        super(MetaLearner, self).__init__()

        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.meta_batchsz = meta_batchsz
        self.meta_lr = meta_lr

        self.num_updates = num_updates

        self.learner = Learner(net_cls, *net_cls_args)

        self.optimizer = optim.Adam(self.learner.parameters(), lr=meta_lr)

    def write_grads(self, dummy_loss, sum_grads_pi):

        # Register a hook on each parameter in the net that replaces the current dummy grad
        # with our grads accumulated across the meta-batch

        params = [p for p in self.learner.parameters() if p.requires_grad]

        hooks = []

        for i, v in enumerate(params):
            def closure():
                ii = i
                return lambda grad: sum_grads_pi[ii]

            # if you write: hooks.append( v.register_hook(lambda grad : sum_grads_pi[i]) )
            # it will pop an ERROR, i don't know why?
            hooks.append(v.register_hook(closure()))

        # use our sumed gradients_pi to update the theta/net network,
        # since our optimizer receive the self.net.parameters() only.
        self.optimizer.zero_grad()

        dummy_loss.backward()
        self.optimizer.step()

        # if you do NOT remove the hook, the GPU memory will expode!!!
        for h in hooks:
            h.remove()

    def forward(self, support_x_atom, support_x_bond, padding_mask_s, support_x_img, support_y, query_x_atom,
                query_x_bond, padding_mask_q, query_x_img, query_y):
        """
		Here we receive a series of episode, each episode will be learned by learner and get a loss on parameters theta.
		we gather the loss and sum all the loss and then update theta network.
		setsz = n_way * k_shotf
		querysz = n_way * k_shot
		:param support_x: [meta_batchsz, setsz, c_, h, w]
		:param support_y: [meta_batchsz, setsz]
		:param query_x:   [meta_batchsz, querysz, c_, h, w]
		:param query_y:   [meta_batchsz, querysz]
		:return:
		"""
        sum_grads_pi = None
        meta_batchsz = support_y.size(0)

        # s_atom_features:[batchsz,setsz,natoms,dim]
        # we do different learning task sequentially, not parallel.
        rocs = []
        losses = []

        support_x_atom = support_x_atom.to(torch.float32)
        support_x_bond = support_x_bond.to(torch.float32)
        # support_x_mask = padding_mask_s.to(torch.float32)
        support_x_img = support_x_img.to(torch.float32)
        support_y = support_y.to(torch.float32)

        # query_x_atom, query_x_bond, query_x_atom_index, query_x_bond_index, query_x_mask = query_x
        query_x_atom = query_x_atom.to(torch.float32)
        query_x_bond = query_x_bond.to(torch.float32)
        # query_x_mask = padding_mask_q.to(torch.float32)
        query_x_img = query_x_img.to(torch.float32)
        query_y = query_y.to(torch.float32)

        for i in range(meta_batchsz):

            loss, grad_pi, episode_scores = self.learner(support_x_atom[i], support_x_bond[i], padding_mask_s[i],
                                                         support_x_img[i], support_y[i],
                                                         query_x_atom[i], query_x_bond[i], padding_mask_q[i],
                                                         query_x_img[i],
                                                         query_y[i],
                                                         self.num_updates)
            #
            rocs.append(episode_scores[4])
            losses.append(loss)
            if sum_grads_pi is None:
                sum_grads_pi = grad_pi
            else:  # accumulate all gradients from different episode learner
                sum_grads_pi = [torch.add(i, j) for i, j in zip(sum_grads_pi, grad_pi)]
                # sum_grads_pi = [torch.add(i, j) for i, j in zip(sum_grads_pi, grad_pi) if
                #                 i is not None and j is not None]

        # As we already have the grads to update，
        # We use a dummy forward / backward pass to get the correct grads into self.net
        # the right grads will be updated by hook, ignoring backward.
        # use hook mechnism to write sumed gradient into network.
        # we need to update the theta/net network, we need a op from net network, so we call self.learner.net_forward
        # to get the op from net network, since the loss from self.learner.forward will return loss from net_pi network.

        dummy_loss, _ = self.learner.net_forward(support_x_atom[0], support_x_bond[0],
                                                 padding_mask_s[0], support_x_img[0], support_y[0])
        self.write_grads(dummy_loss, sum_grads_pi)

        return losses, rocs

    def pred(self, s_atom_features, s_edge_features, support_x_mask, s_img_features, support_y, q_atom_features,
             q_edge_features, query_x_mask, q_img_features, query_y):
        """
		predict for query_x
		:param support_x:
		:param support_y:
		:param query_x:
		:param query_y:
		:return:
		"""
        meta_batchsz = support_y.size(0)

        accs = []
        losses = []
        pre_scores = []
        recall_scores = []
        mcc_scores = []
        roc_scores = []
        f1_scores = []

        support_x_atom = s_atom_features.to(torch.float32)
        support_x_bond = s_edge_features.to(torch.float32)

        support_x_img = s_img_features.to(torch.float32)

        # query_x_atom, query_x_bond, query_x_atom_index, query_x_bond_index, query_x_mask = query_x
        query_x_atom = q_atom_features.to(torch.float32)
        query_x_bond = q_edge_features.to(torch.float32)

        query_x_img = q_img_features.to(torch.float32)

        for i in range(meta_batchsz):
            loss, _, episode_scores = self.learner(support_x_atom[i], support_x_bond[i], support_x_mask[i],
                                                   support_x_img[i], support_y[i], query_x_atom[i], query_x_bond[i],
                                                   query_x_mask[i], query_x_img[i], query_y[i], self.num_updates)
            episode_acc, pre_score, recall_score, mcc_score, roc_score, f1_score = episode_scores
            accs.append(episode_acc)
            losses.append(loss)
            pre_scores.append(pre_score)
            recall_scores.append(recall_score)
            mcc_scores.append(mcc_score)
            roc_scores.append(roc_score)
            f1_scores.append(f1_score)

        return losses, np.array(accs).mean(), np.array(pre_scores).mean(), np.array(recall_scores).mean(), \
            np.array(mcc_scores).mean(), np.array(roc_scores).mean(), np.array(f1_scores).mean()
