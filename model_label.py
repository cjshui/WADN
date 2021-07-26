import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from sklearn.metrics import confusion_matrix

from module import L2ProjFunction, GradientReversalLayer
import utils


########## Some components ##########
class MLPNet(nn.Module):
    def __init__(self, configs):
        """
        MLP network with ReLU
        """

        super().__init__()
        self.input_dim = configs["input_dim"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        # Parameters of hidden, fully-connected layers
        self.hiddens = nn.ModuleList(
            [
                nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                for i in range(self.num_hidden_layers)
            ]
        )
        self.final = nn.Linear(self.num_neurons[-1], configs["output_dim"])
        self.dropout = nn.Dropout(p=configs["drop_rate"])  # drop probability
        self.process_final = configs["process_final"]

    def forward(self, x):

        for hidden in self.hiddens:
            x = F.relu(hidden(self.dropout(x)))
        if self.process_final:
            return F.relu(self.final(self.dropout(x)))
        else:
            # no dropout or transform
            return self.final(x)


class ConvNet(nn.Module):
    def __init__(self, configs):
        """
        Feature extractor for the image (digits) datasets
        """

        super().__init__()
        self.channels = configs["channels"]  # number of channels
        self.num_conv_layers = len(configs["conv_layers"])
        self.num_channels = [self.channels] + configs["conv_layers"]
        # Parameters of hidden, cpcpcp, feature learning component.
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(self.num_channels[i], self.num_channels[i + 1], kernel_size=3)
                for i in range(self.num_conv_layers)
            ]
        )
        self.dropout = nn.Dropout(p=configs["drop_rate"])  # drop probability

    def forward(self, x):

        dropout = self.dropout
        for conv in self.convs:
            x = F.max_pool2d(F.relu(conv(dropout(x))), 2, 2, ceil_mode=True)
        x = x.view(x.size(0), -1)  # flatten
        return x


class MLPNet_digits(nn.Module):
    def __init__(self, configs):
        """
        MLP network with ReLU
        """

        super().__init__()
        self.input_dim = configs["input_dim"]
        self.num_hidden_layers = len(configs["hidden_layers"])
        self.num_neurons = [self.input_dim] + configs["hidden_layers"]
        # Parameters of hidden, fully-connected layers
        self.hiddens = nn.ModuleList(
            [
                nn.Linear(self.num_neurons[i], self.num_neurons[i + 1])
                for i in range(self.num_hidden_layers)
            ]
        )
        self.final = nn.Linear(self.num_neurons[-1], configs["output_dim"])
        self.dropout = nn.Dropout(p=configs["drop_rate"])  # drop probability
        self.process_final = configs["process_final"]

    def forward(self, x):

        for hidden in self.hiddens:
            x = F.relu(hidden(self.dropout(x)))
        latent_x = x
        if self.process_final:
            return latent_x, F.relu(self.final(self.dropout(x)))
        else:
            # no dropout or transform
            return latent_x, self.final(x)


class Label_WarnBase(nn.Module):
    def __init__(self, configs):
        """
        Domain AggRegation Network.
        """

        super().__init__()
        self.num_src_domains = configs["num_src_domains"]
        # define the classes numbers
        self.num_class = configs["num_src_classes"]
        self.fea_dim = configs["feauture_dim"]
        # Gradient reversal layer.
        self.grl = GradientReversalLayer.apply
        # self.mode = configs["mode"]
        self.mu = configs["mu"]
        self.gp_coef = configs["gp_coef"]
        self.sem_coef = configs["sem_coef"]
        self.gamma = configs["gamma"]

        # option about semantic matching
        self.semantic = True

        # define the confusion matrix for every source domain (in the presence of target label, no need to compute C)
        # self.C = np.zeros([self.num_class, self.num_class, self.num_src_domains])

        # define the label re-weights alpha (T-task times num_classes)
        self.lam = np.ones([self.num_src_domains]) / self.num_src_domains

        # defining the src_centre (self.num_src_domains X self.num_class X self. fea_dim)
        self.src_centroid = torch.zeros([self.num_src_domains, self.num_class, self.fea_dim])
        self.tar_centroid = torch.zeros([self.num_class, self.fea_dim])
        self.decay = 0.3

        # define the confusion matrix, source taget prediction output distribution (in the presence of target label, no need to compute this)
        # self.tar_pred = np.zeros([self.num_class])

    def forward(self, sinputs, soutputs, tinputs, toutput, alpha, tar_truth_label):
        """
        :param sinputs:     A list of k inputs from k source domains.
        :param soutputs:    A list of k outputs from k source domains.
        :param tinputs:     Input from the target domain.
        :param toutput:     target domain labels
        :estimated_tar_dis: Estimated target label distribution (this is different from target prediction distribution)
        :return:            tuple(aggregated loss, domain weights)
        """

        # Compute features
        s_features = []
        for dom_idx in range(self.num_src_domains):
            s_features.append(self.feature_net(sinputs[dom_idx]))
        t_features = self.feature_net(tinputs)

        # Classification probabilities on k source domains

        logprobs = []
        for dom_idx in range(self.num_src_domains):

            # with torch.no_grad():
            # source prediction error
            # src_pred = torch.argmax(self.class_net(s_features[dom_idx]),1).cpu().numpy()
            # tar_pred = torch.argmax(self.class_net(t_features), 1).cpu().numpy()
            # src_true = soutputs[dom_idx].cpu().numpy()

            logprobs.append(F.log_softmax(self.class_net(s_features[dom_idx]), dim=1))

        # weighted prediction loss
        cls_losses = torch.stack(
            [
                F.nll_loss(logprobs[dom_idx], soutputs[dom_idx], weight=alpha[dom_idx, :])
                for dom_idx in range(self.num_src_domains)
            ]
        )

        # Domain classification accuracies. (wasserstein based approach)
        sdomains, tdomains = [], []
        batch_size = tinputs.shape[0]

        src_alpha = []
        src_alpha_weights = torch.ones(
            [self.num_src_domains, batch_size],
            requires_grad=False,
            dtype=torch.float32,
            device=tinputs.device,
        )

        for dom_idx in range(self.num_src_domains):
            for cls_idx in range(self.num_class):
                src_alpha_weights[dom_idx, soutputs[dom_idx] == cls_idx] = alpha[dom_idx, cls_idx]
            src_alpha.append(src_alpha_weights)

        for dom_idx in range(self.num_src_domains):
            # weighted src adversarial loss
            sdomains.append(
                torch.mul(
                    self.domain_nets[dom_idx](self.grl(s_features[dom_idx])),
                    torch.unsqueeze(src_alpha_weights[dom_idx], -1),
                )
            )
            tdomains.append(self.domain_nets[dom_idx](self.grl(t_features)))

        # domain loss is the Wasserstein loss (current w.o gradient penality)
        domain_losses = torch.stack(
            [torch.mean(sdomains[i]) - torch.mean(tdomains[i]) for i in range(self.num_src_domains)]
        )

        # Defining domain regularization loss (gradient penality)
        domain_gradient = []

        for tsk in range(self.num_src_domains):
            src_rand = s_features[tsk]
            epsilon = np.random.rand()
            interpolated = epsilon * src_rand + (1 - epsilon) * t_features
            inter_f = self.domain_nets[tsk](interpolated)
            # The following compute the penalty of the Lipschitz constant
            penalty_coefficient = 10.0
            # torch.norm can be unstable? https://github.com/pytorch/pytorch/issues/2534
            # f_gradient_norm = torch.norm(torch.autograd.grad(torch.sum(inter_f), interpolated)[0], dim=1)
            f_gradient = torch.autograd.grad(
                torch.sum(inter_f), interpolated, create_graph=True, retain_graph=True
            )[0]
            f_gradient_norm = torch.sqrt(torch.sum(f_gradient ** 2, dim=1) + 1e-10)
            domain_gradient_penalty = penalty_coefficient * torch.mean((f_gradient_norm - 1.0) ** 2)

            domain_gradient.append(domain_gradient_penalty)
        domain_gradient = torch.stack(domain_gradient)

        # semantic loss (depending on the tar_reweighted loss)

        src_semantic = []
        # tar_pred_cuda = torch.tensor(tar_pred).to(alpha.device)
        tar_real_cuda = torch.FloatTensor(tar_truth_label).to(alpha.device)

        for tsk in range(self.num_src_domains):
            sematinc_loss = self.update_center(
                tsk, s_features[tsk], t_features, soutputs[tsk], toutput, tar_real_cuda
            )
            src_semantic.append(sematinc_loss)
        src_semantic = torch.stack(src_semantic)

        return self._aggregation(cls_losses, domain_losses, domain_gradient, src_semantic)

    def _aggregation(self, cls_losses, domain_losses, domain_gradient, src_semantic):
        """
        Aggregate the losses into a scalar
        """

        losses_tuple = (cls_losses, domain_losses, domain_gradient, src_semantic)
        mu = self.mu
        gp_coef = self.gp_coef
        sem_coef = self.sem_coef
        train_loss = cls_losses + mu * (
            domain_losses + gp_coef * domain_gradient + sem_coef * src_semantic
        )
        convex_loss = (cls_losses + 0.01 * src_semantic).detach()

        return train_loss, convex_loss, losses_tuple

    def update_center(self, tsk, src_fea, tar_fea, s_true, t_pseudo, tar_y_estimated):
        # get feature size (batch_size X dimension)
        self.src_centroid = self.src_centroid.to(src_fea.device)
        self.tar_centroid = self.tar_centroid.to(src_fea.device)
        n, d = src_fea.shape

        # get labels
        s_labels, t_labels = s_true, t_pseudo

        # image number in each class
        ones = torch.ones_like(s_labels, dtype=torch.float)
        zeros = torch.zeros(self.num_class).to(src_fea.device)

        # smaples per class
        s_n_classes = zeros.scatter_add(0, s_labels, ones)
        t_n_classes = zeros.scatter_add(0, t_labels, ones)

        # image number cannot be 0, when calculating centroids
        ones = torch.ones_like(s_n_classes)
        s_n_classes = torch.max(s_n_classes, ones)
        t_n_classes = torch.max(t_n_classes, ones)

        # calculating centroids, sum and divide
        zeros = torch.zeros(self.num_class, d).to(src_fea.device)

        s_sum_feature = zeros.scatter_add(0, torch.transpose(s_labels.repeat(d, 1), 1, 0), src_fea)
        t_sum_feature = zeros.scatter_add(0, torch.transpose(t_labels.repeat(d, 1), 1, 0), tar_fea)
        current_s_centroid = torch.div(s_sum_feature, s_n_classes.view(self.num_class, 1))
        current_t_centroid = torch.div(t_sum_feature, t_n_classes.view(self.num_class, 1))

        # Moving Centroid
        decay = self.decay
        src_centroid = (1 - decay) * self.src_centroid[tsk, :, :] + decay * current_s_centroid
        tar_centroid = (1 - decay) * self.tar_centroid + decay * current_t_centroid
        # Ver 1 finished
        s_loss = torch.mean(torch.pow(src_centroid - tar_centroid, 2), dim=1)
        semantic_loss = torch.sum(torch.mul(tar_y_estimated, s_loss))
        self.src_centroid[tsk, :, :] = src_centroid.detach()
        self.trc_centroid = tar_centroid.detach()

        return semantic_loss

    def inference(self, x):

        x = self.feature_net(x)
        x = self.class_net(x)
        return F.log_softmax(x, dim=1)


class Label_WarnBase_digits(nn.Module):
    def __init__(self, configs):
        """
        Domain AggRegation Network.
        """

        super().__init__()
        self.num_src_domains = configs["num_src_domains"]
        # define the classes numbers
        self.num_class = configs["num_src_classes"]
        self.fea_dim = configs["feauture_dim"]
        # Gradient reversal layer.
        self.grl = GradientReversalLayer.apply
        # self.mode = configs["mode"]
        self.mu = configs["mu"]
        self.gp_coef = configs["gp_coef"]
        self.sem_coef = configs["sem_coef"]
        self.gamma = configs["gamma"]

        # option about semantic matching
        self.semantic = True

        # define the confusion matrix for every source domain
        self.C = np.zeros([self.num_class, self.num_class, self.num_src_domains])

        # define the label re-weights alpha (T-task times num_classes)
        self.lam = np.ones([self.num_src_domains]) / self.num_src_domains

        # defining the src_centre (self.num_src_domains X self.num_class X self. fea_dim)
        self.src_centroid = torch.zeros([self.num_src_domains, self.num_class, self.fea_dim])
        self.tar_centroid = torch.zeros([self.num_class, self.fea_dim])
        self.decay = 0.3

        # mse loss for semantic losses
        self.MSELoss = nn.MSELoss(reduction="none")

        # define the confusion matrix, source taget prediction output distribution
        self.tar_pred = np.zeros([self.num_class])

    def forward(self, sinputs, soutputs, tinputs, toutput, alpha, tar_truth_label):
        """
        :param sinputs:     A list of k inputs from k source domains.
        :param soutputs:    A list of k outputs from k source domains.
        :param tinputs:     Input from the target domain.
        :estimated_tar_dis: Estimated target label distribution (this is different from target prediction distribution)
        :return:            tuple(aggregated loss, domain weights)
        """

        # Compute features
        s_features = []
        s_semantic = []

        for dom_idx in range(self.num_src_domains):
            s_features.append(self.feature_net(sinputs[dom_idx]))
            s_semantic.append(self.class_net(s_features[dom_idx])[0])

        t_features = self.feature_net(tinputs)
        t_semantic = self.class_net(t_features)[0]

        # Classification probabilities on k source domains

        logprobs = []
        for dom_idx in range(self.num_src_domains):

            # with torch.no_grad():
            # source prediction error
            # src_pred = torch.argmax(self.class_net(s_features[dom_idx])[1],1).cpu().numpy()
            # tar_pred = torch.argmax(self.class_net(t_features)[1], 1).cpu().numpy()
            # src_true = soutputs[dom_idx].cpu().numpy()

            # un-normalized
            # self.C[:,:,dom_idx] = confusion_matrix(src_true,src_pred,labels=list(range(self.num_class)))

            # for cls_idx in range(self.num_class):
            #   self.tar_pred[cls_idx] =  np.count_nonzero(tar_pred == cls_idx)

            logprobs.append(F.log_softmax(self.class_net(s_features[dom_idx])[1], dim=1))

        # weighted prediction loss
        cls_losses = torch.stack(
            [
                F.nll_loss(logprobs[dom_idx], soutputs[dom_idx], weight=alpha[dom_idx, :])
                for dom_idx in range(self.num_src_domains)
            ]
        )

        # Domain classification accuracies. (wasserstein based approach)

        sdomains, tdomains = [], []
        batch_size = tinputs.shape[0]

        src_alpha = []
        src_alpha_weights = torch.ones(
            [self.num_src_domains, batch_size],
            requires_grad=False,
            dtype=torch.float32,
            device=tinputs.device,
        )

        for dom_idx in range(self.num_src_domains):
            for cls_idx in range(self.num_class):
                src_alpha_weights[dom_idx, soutputs[dom_idx] == cls_idx] = alpha[dom_idx, cls_idx]
            src_alpha.append(src_alpha_weights)

        for dom_idx in range(self.num_src_domains):
            # weighted src adversarial loss
            sdomains.append(
                torch.mul(
                    self.domain_nets[dom_idx](self.grl(s_features[dom_idx])),
                    torch.unsqueeze(src_alpha_weights[dom_idx], -1),
                )
            )
            tdomains.append(self.domain_nets[dom_idx](self.grl(t_features)))

        # slabels = torch.ones([batch_size, 1], requires_grad=False,
        #                     dtype=torch.float32, device=tinputs.device)
        # tlabels = torch.zeros([batch_size, 1], requires_grad=False,
        #                      dtype=torch.float32, device=tinputs.device)

        # domain loss is the Wasserstein loss (current w.o gradient penality)
        domain_losses = torch.stack(
            [torch.mean(sdomains[i]) - torch.mean(tdomains[i]) for i in range(self.num_src_domains)]
        )

        # Defining domain regularization loss (gradient penality)
        domain_gradient = []

        for tsk in range(self.num_src_domains):
            src_rand = s_features[tsk]
            epsilon = np.random.rand()
            interpolated = epsilon * src_rand + (1 - epsilon) * t_features
            inter_f = self.domain_nets[tsk](interpolated)
            # The following compute the penalty of the Lipschitz constant
            penalty_coefficient = 10.0
            # torch.norm can be unstable? https://github.com/pytorch/pytorch/issues/2534
            # f_gradient_norm = torch.norm(torch.autograd.grad(torch.sum(inter_f), interpolated)[0], dim=1)
            f_gradient = torch.autograd.grad(
                torch.sum(inter_f), interpolated, create_graph=True, retain_graph=True
            )[0]
            f_gradient_norm = torch.sqrt(torch.sum(f_gradient ** 2, dim=1) + 1e-10)
            domain_gradient_penalty = penalty_coefficient * torch.mean((f_gradient_norm - 1.0) ** 2)

            domain_gradient.append(domain_gradient_penalty)
        domain_gradient = torch.stack(domain_gradient)

        # semantic loss (depending on the tar_reweighted loss)

        # tar_pred_cuda = torch.tensor(tar_pred).to(alpha.device)
        src_semantic = []
        tar_real_cuda = torch.FloatTensor(tar_truth_label).to(alpha.device)

        for tsk in range(self.num_src_domains):
            # tar_y_estimated = alpha[tsk,:] * src_truth_label[tsk,:]
            sematinc_loss = self.update_center(
                tsk, s_semantic[tsk], t_semantic, soutputs[tsk], toutput, tar_real_cuda
            )
            src_semantic.append(sematinc_loss)
        src_semantic = torch.stack(src_semantic)

        return self._aggregation(cls_losses, domain_losses, domain_gradient, src_semantic)

    def _aggregation(self, cls_losses, domain_losses, domain_gradient, src_semantic):
        """
        Aggregate the losses into a scalar
        """
        losses_tuple = (cls_losses, domain_losses, domain_gradient, src_semantic)
        mu = self.mu
        gp_coef = self.gp_coef
        sem_coef = self.sem_coef
        train_loss = cls_losses + mu * (
            domain_losses + gp_coef * domain_gradient + sem_coef * src_semantic
        )
        convex_loss = (cls_losses + 0.01 * src_semantic).detach()

        return train_loss, convex_loss, losses_tuple

    def update_center(self, tsk, src_fea, tar_fea, s_true, t_pseudo, tar_y_estimated):
        self.src_centroid = self.src_centroid.to(src_fea.device)
        self.tar_centroid = self.tar_centroid.to(src_fea.device)
        # get feature size (batch_size X dimension)
        n, d = src_fea.shape

        # get labels
        s_labels, t_labels = s_true, t_pseudo

        # image number in each class
        ones = torch.ones_like(s_labels, dtype=torch.float)
        zeros = torch.zeros(self.num_class).to(src_fea.device)

        # smaples per class
        s_n_classes = zeros.scatter_add(0, s_labels, ones)
        t_n_classes = zeros.scatter_add(0, t_labels, ones)

        # image number cannot be 0, when calculating centroids
        ones = torch.ones_like(s_n_classes)
        s_n_classes = torch.max(s_n_classes, ones)
        t_n_classes = torch.max(t_n_classes, ones)

        # calculating centroids, sum and divide
        zeros = torch.zeros(self.num_class, d).to(src_fea.device)

        s_sum_feature = zeros.scatter_add(0, torch.transpose(s_labels.repeat(d, 1), 1, 0), src_fea)
        t_sum_feature = zeros.scatter_add(0, torch.transpose(t_labels.repeat(d, 1), 1, 0), tar_fea)
        current_s_centroid = torch.div(s_sum_feature, s_n_classes.view(self.num_class, 1))
        current_t_centroid = torch.div(t_sum_feature, t_n_classes.view(self.num_class, 1))

        # Moving Centroid
        decay = self.decay
        src_centroid = (1 - decay) * self.src_centroid[tsk, :, :] + decay * current_s_centroid
        tar_centroid = (1 - decay) * self.tar_centroid + decay * current_t_centroid

        # *** version 1 ***
        s_loss = torch.mean(torch.pow(src_centroid - tar_centroid, 2), dim=1)
        semantic_loss = torch.sum(torch.mul(tar_y_estimated, s_loss))
        # *** version 2: code from MSTN ***
        # s_loss = self.MSELoss(src_centroid, tar_centroid)
        # semantic_loss = torch.sum(torch.mm(torch.unsqueeze(tar_y_estimated, 0), s_loss)) / n

        self.src_centroid[tsk, :, :] = src_centroid.detach()
        self.trc_centroid = tar_centroid.detach()

        return semantic_loss

    def inference(self, x):

        x = self.feature_net(x)
        x = self.class_net(x)[1]
        return F.log_softmax(x, dim=1)


class WarnMLP_Label(Label_WarnBase):
    def __init__(self, configs):
        """
        DARN with MLP
        """

        super().__init__(configs)

        fea_configs = {
            "input_dim": configs["input_dim"],
            "hidden_layers": configs["hidden_layers"][:-1],
            "output_dim": configs["hidden_layers"][-1],
            "drop_rate": configs["drop_rate"],
            "process_final": True,
        }
        self.feature_net = MLPNet(fea_configs)

        self.class_net = nn.Linear(configs["hidden_layers"][-1], configs["num_classes"])

        self.domain_nets = nn.ModuleList(
            [nn.Linear(configs["hidden_layers"][-1], 1) for _ in range(self.num_src_domains)]
        )


class WarnConv_digits_Label(Label_WarnBase_digits):
    def __init__(self, configs):
        """
        WARN with convolution feature extractor
        """

        super().__init__(configs)

        self.feature_net = ConvNet(configs)

        cls_configs = {
            "input_dim": configs["input_dim"],
            "hidden_layers": configs["cls_fc_layers"],
            "output_dim": configs["num_classes"],
            "drop_rate": configs["drop_rate"],
            "process_final": False,
        }
        self.class_net = MLPNet_digits(cls_configs)

        dom_configs = {
            "input_dim": configs["input_dim"],
            "hidden_layers": configs["dom_fc_layers"],
            "output_dim": 1,
            "drop_rate": configs["drop_rate"],
            "process_final": False,
        }
        self.domain_nets = nn.ModuleList([MLPNet(dom_configs) for _ in range(self.num_src_domains)])


class DarnBase_WoAdv(nn.Module):
    def __init__(self, configs):
        """
        Domain AggRegation Network.
        """

        super().__init__()
        self.num_src_domains = configs["num_src_domains"]
        # Gradient reversal layer.
        self.grl = GradientReversalLayer.apply
        # self.mode = configs["mode"]
        self.mu = configs["mu"]
        self.gamma = configs["gamma"]

    def forward(self, sinputs, soutputs, tinputs, toutputs):
        """
        :param sinputs:     A list of k inputs from k source domains.
        :param soutputs:    A list of k outputs from k source domains.
        :param tinputs:     Input from the target domain.
        :return:            tuple(aggregated loss, domain weights)
        """

        # Compute features
        s_features = []
        for i in range(self.num_src_domains):
            s_features.append(self.feature_net(sinputs[i]))
        t_features = self.feature_net(tinputs)

        # Classification probabilities on k source domains.
        logprobs = []
        for i in range(self.num_src_domains):
            logprobs.append(F.log_softmax(self.class_net(s_features[i]), dim=1))
        train_losses = torch.stack(
            [F.nll_loss(logprobs[i], soutputs[i]) for i in range(self.num_src_domains)]
        )

        tar_train_loss = F.nll_loss(F.log_softmax(t_features, dim=1), toutputs)

        return self._aggregation(train_losses, tar_train_loss)

    def _aggregation(self, train_losses, tar_train_loss):
        """
        Aggregate the losses into a scalar
        """

        return train_losses, tar_train_loss

    def inference(self, x):

        x = self.feature_net(x)
        x = self.class_net(x)
        return F.log_softmax(x, dim=1)