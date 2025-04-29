#!/usr/bin/env python3
import torch
from torch import nn
from torch import optim
from torch import autograd
import torch.nn.functional as F
import numpy as np

from Module.datautils import args
from utils.metrics import recall, precision, mcc, roc, prc_auc, accuracy, f1



class Learner(nn.Module):
    """
    Learner类，接受一个需要训练的模型，Learner类会创建两个网络，一个作为元学习器，一个作为任务学习器
    对于每一个episode，任务学习器会复制元学习器的参数，并采用支持集更新任务模型，计算任务学习器在查询集上的loss之和，这个loss将要反向传播回元模型，这个过程在metalearner类里面完成
    总结:返回任务模型在查询集上的Loss
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
        # 创建优化器，排除被冻结的参数
        self.optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.net_pi.parameters()), lr=0.001)

        # # 冻结 self.net 和 self.net_pi 中的 CNN 模块参数
        # for net in [self.net, self.net_pi]:
        #     for name, param in net.named_parameters():
        #         if 'cnn' in name:  # 假设 CNN 模块的名称包含 'cnn'
        #             param.requires_grad = False

    def parameters(self):
        """
		Override this function to return only net parameters for MetaLearner's optimize
		it will ignore theta_pi network parameters.
		:return:
		"""
        return self.net.parameters()

    def update_pi(self):
        """
		copy parameters from self.net -> self.net_pi
		:return:
		"""
        for m_from, m_to in zip(self.net.modules(), self.net_pi.modules()):
            if isinstance(m_to, nn.Linear) or isinstance(m_to, nn.Conv2d) or isinstance(m_to, nn.BatchNorm2d):
                m_to.weight.data = m_from.weight.data.clone()
                if m_to.bias is not None:
                    m_to.bias.data = m_from.bias.data.clone()

    def forward(self, support_x_atom, support_x_bond, support_x_mask,support_x_img,
                support_y,
                query_x_atom, query_x_bond,query_x_mask,query_x_img, query_y, num_updates):
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
        # 把元模型的参数赋值给任务模型
        self.update_pi()

        # update for several steps 对元模型进行num_updates次更新得到学习后的任务模型
        for i in range(num_updates):
            # forward and backward to update net_pi grad.
            loss, pred, _ = self.net_pi(support_x_atom, support_x_bond, support_x_mask,support_x_img,
                             support_y)
            self.optimizer.zero_grad()
            # 这里的loss backward回去了
            loss.backward()
            self.optimizer.step()

        # Compute the meta gradient and return it, the gradient is from one episode
        # in metalearner, it will merge all loss from different episode and sum over it.
        # 计算在查询集上的loss
        loss, pred, _ = self.net_pi(query_x_atom, query_x_bond,query_x_mask,query_x_img, query_y)

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
        # 这里计算任务模型的梯度
        grads_pi = autograd.grad(loss, params_pi, create_graph=True)

        return loss, grads_pi, (acc, pre_score, recall_score, mcc_score, roc_score, f1_score)

    def net_forward(self, support_x_atom, support_x_bond, support_x_mask,support_x_img,
                support_y):
        """
		This function is purely for updating net network. In metalearner, we need the get the loss op from net network
		to write our merged gradients into net network, hence will call this function to get a dummy loss op.
		:param support_x: [setsz, c, h, w]
		:param support_y: [sessz, c, h, w]
		:return: dummy loss and dummy pred
		"""
        loss, pred, _ = self.net(support_x_atom, support_x_bond, support_x_mask,support_x_img,
                support_y)
        return loss, pred


class MetaLearner(nn.Module):
    """
	整个网络，训练过程在网络里完成，应该包括正向传播和反向传播
	learner为元模型
	optimizer优化器
	As we have mentioned in Learner class, the metalearner class will receive a series of loss on different tasks/episodes
	on theta_pi network, and it will merage all loss and then sum over it. The summed loss will be backproped on theta
	network to update theta parameters, which is the initialization point we want to find.
	MetaLearner类接受不同任务上的任务模型的loss之和并累加，这个loss用来优化元模型
	"""

    def __init__(self, net_cls, net_cls_args, n_way=args.n_way, k_shot=args.k_shot, k_query=args.k_query,
                 meta_batchsz=args.meta_batchsz, meta_lr=args.meta_lr, num_updates=args.num_updates):
        """

		:param net_cls: 传入一个nn.module类，class, not instance. the class of specific Network for learner
		:param net_cls_args: 传入模型的参数，tuple, args for net_cls, like (n_way, imgsz)

		以下的参数都初始化好了
		:param n_way:
		:param k_shot:
		:param meta_batchsz:每次采样的任务数 number of tasks/episode
		:param meta_lr: 学习率，learning rate for meta-learner
		:param num_updates: 更新的次数，number of updates for learner
		"""
        super(MetaLearner, self).__init__()

        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query
        self.meta_batchsz = meta_batchsz
        self.meta_lr = meta_lr
        # self.alpha = alpha # set alpha in Learner.optimizer directly.
        self.num_updates = num_updates

        # it will contains a learner class to learn on episodes and gather the loss together.
        # 元模型，接受一个模型的类和参数
        self.learner = Learner(net_cls, *net_cls_args)
        # 这个优化器是优化元模型的参数的
        # the optimizer is to update theta parameters, not theta_pi parameters.
        # 传入的是learner的参数
        self.optimizer = optim.Adam(self.learner.parameters(), lr=meta_lr)

    def write_grads(self, dummy_loss, sum_grads_pi):
        """
		将梯度写入网络中，这些梯度来自于 sum_grads_pi
		write loss into learner.net, gradients come from sum_grads_pi.
		Since the gradients info is not calculated by general backward, we need this function to write the right gradients
		into theta network and update theta parameters as wished.
		:param dummy_loss: dummy loss, nothing but to write our gradients by hook
		:param sum_grads_pi: the summed gradients
		:return:

		"""

        # Register a hook on each parameter in the net that replaces the current dummy grad
        # with our grads accumulated across the meta-batch
        '''
        这段代码的目的是将外部计算的梯度（sum_grads_pi）应用到模型参数上。这是通过注册钩子来实现的，这些钩子在反向传播过程中被调用，并用sum_grads_pi中的梯度替换原有的梯度
        dummy_loss是一个虚拟的损失值，它仅仅是为了触发PyTorch的自动梯度计算机制（即.backward()调用），但实际上并没有用到它的具体值。通过注册的钩子，模型的每个参数在反向传播时都会使用sum_grads_pi中的梯度，而不是通过dummy_loss计算出的梯度。
        总结来说，这段代码允许你绕过常规的损失函数计算和反向传播过程，直接使用你提供的梯度来更新模型的参数。
        这段代码是在使用PyTorch的自定义钩子机制来捕获和操作梯度。这里是如何工作的：
初始化一个空列表 hooks，用于存储回调函数（闭包）。
遍历 self.learner 的所有参数 v，这里假设 self.learner 是一个 torch.nn.Module 实例，它包含了模型的参数。
对于每个参数，定义一个闭包 closure。这个闭包将捕获变量 ii（当前参数的索引），并返回一个 lambda 函数，这个 lambda 函数将接受一个梯度 grad 作为输入，并返回 sum_grads_pi[ii]。
sum_grads_pi[ii] 表示累积的梯度，这里假设 sum_grads_pi 是一个列表，其中包含了每个参数的累积梯度。
当这个闭包被调用时（例如，在反向传播过程中），它返回的 lambda 函数将被用来替换当前参数的梯度，从而在应用梯度更新之前，将累积的梯度应用于这个参数。
将这个闭包添加到 hooks 列表中，这样它就可以在反向传播过程中被调用。
这个机制的目的是为了在每个参数上应用累积的梯度，而不是直接应用当前的梯度。这可能用于实现一些特定的训练策略，比如梯度累积或者调整学习率。通过这种方式，可以在优化过程中更灵活地控制每个参数的更新。
'''
        # 获取所有需要梯度的参数
        params = [p for p in self.learner.parameters() if p.requires_grad]

        hooks = []

        # for i, v in enumerate(self.learner.parameters()):
        for i, v in enumerate(params):
            def closure():
                ii = i
                # 使用预定义的sum_grads_pi列表中的梯度来替换原始梯度。
                return lambda grad: sum_grads_pi[ii]

            # if you write: hooks.append( v.register_hook(lambda grad : sum_grads_pi[i]) )
            # it will pop an ERROR, i don't know why?
            hooks.append(v.register_hook(closure()))

        # use our sumed gradients_pi to update the theta/net network,
        # since our optimizer receive the self.net.parameters() only.
        self.optimizer.zero_grad()
        # 计算一个虚拟损失dummy_loss，并调用其backward方法来计算梯度。由于注册了钩子，所以在这个过程中，模型的参数梯度会被sum_grads_pi中的梯度所替换
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
        # sum_grads_pi为在一个batch上所有任务上的梯度之和
        sum_grads_pi = None
        # 获取batchsize
        meta_batchsz = support_y.size(0)

        # s_atom_features:[batchsz,setsz,natoms,dim]
        # we do different learning task sequentially, not parallel.
        rocs = []
        losses = []

        # 拿到support_x
        # support_x_atom, support_x_bond, support_x_atom_index, support_x_bond_index, support_x_mask = support_x
        # 统一数据
        support_x_atom = support_x_atom.to(torch.float32)
        support_x_bond = support_x_bond.to(torch.float32)
        # support_x_mask = padding_mask_s.to(torch.float32)
        support_x_img = support_x_img.to(torch.float32)
        support_y=support_y.to(torch.float32)

        # query_x_atom, query_x_bond, query_x_atom_index, query_x_bond_index, query_x_mask = query_x
        query_x_atom = query_x_atom.to(torch.float32)
        query_x_bond =  query_x_bond.to(torch.float32)
        # query_x_mask = padding_mask_q.to(torch.float32)
        query_x_img = query_x_img.to(torch.float32)
        query_y=query_y.to(torch.float32)

        # 对于每个任务
        for i in range(meta_batchsz):
            # 先通过支持集对元模型优化得到任务模型，这里得到的loss是查询集在任务模型上的loss，梯度是查询集在任务模型的梯度
            loss, grad_pi, episode_scores = self.learner(support_x_atom[i], support_x_bond[i], padding_mask_s[i],
                                                         support_x_img[i], support_y[i],
                                                         query_x_atom[i], query_x_bond[i], padding_mask_q[i], query_x_img[i],
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
        '''
        	已经拿到了需要用来更新的梯度
        	采用虚拟的loss，只是用来触反向传播过程
        	'''

        dummy_loss, _ = self.learner.net_forward(support_x_atom[0], support_x_bond[0],
                                                 padding_mask_s[0], support_x_img[0], support_y[0])
        self.write_grads(dummy_loss, sum_grads_pi)

        return losses, rocs

    # 可以在测试任务上用这个函数得到roc等结果
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

        # support_x_atom, support_x_bond, support_x_atom_index, support_x_bond_index, support_x_mask = support_x
        # support_x_atom = support_x_atom.to(torch.float32)
        # support_x_bond = support_x_bond.to(torch.float32)
        # support_x_mask = support_x_mask.to(torch.float32)
        # query_x_atom, query_x_bond, query_x_atom_index, query_x_bond_index, query_x_mask = query_x
        # query_x_atom = query_x_atom.to(torch.float32)
        # query_x_bond = query_x_bond.to(torch.float32)
        # query_x_mask = query_x_mask.to(torch.float32)
        # 统一数据
        support_x_atom = s_atom_features.to(torch.float32)
        support_x_bond = s_edge_features.to(torch.float32)

        support_x_img = s_img_features.to(torch.float32)

        # query_x_atom, query_x_bond, query_x_atom_index, query_x_bond_index, query_x_mask = query_x
        query_x_atom = q_atom_features.to(torch.float32)
        query_x_bond = q_edge_features.to(torch.float32)

        query_x_img = q_img_features.to(torch.float32)

        for i in range(meta_batchsz):
            loss, _, episode_scores = self.learner(support_x_atom[i], support_x_bond[i], support_x_mask[i],support_x_img[i],support_y[i],query_x_atom[i], query_x_bond[i],query_x_mask[i],query_x_img[i], query_y[i], self.num_updates)
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
