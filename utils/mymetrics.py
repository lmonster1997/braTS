import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

def cross_entropy_2D(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss

def cross_entropy_3D(input, target, weight=None, size_average=True):
    n, c, h, w, s = input.size()
    log_p = F.log_softmax(input, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).transpose(3, 4).contiguous().view(-1, c)
    target = target.view(target.numel())
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= float(target.numel())
    return loss

class SoftDice(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDice, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1e-8

        # probs = torch.sigmoid(logits)
        m1 = logits.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)
        # if (m1.sum(1) + m2.sum(1)) == 0 and intersection.sum(1)==0:
        #     score = torch.Tensor([1.0])
        # else:
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + 2.*smooth)
        score = score.sum() / num
        return score

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


class DiceMean(nn.Module):
    def __init__(self):
        super(DiceMean, self).__init__()

    def forward(self, logits, targets):
        class_num = logits.size(1)

        dice_sum = 0
        for i in range(class_num):
            inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
            union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dice_sum += dice
        return dice_sum / class_num


class DiceMeanLoss(nn.Module):
    def __init__(self):
        super(DiceMeanLoss, self).__init__()

    def forward(self, logits, targets):
        class_num = logits.size(1)

        dice_sum = 0
        for i in range(class_num):
            inter = torch.sum(logits[:, i, :, :, :] * targets[:, i, :, :, :])
            union = torch.sum(logits[:, i, :, :, :]) + torch.sum(targets[:, i, :, :, :])
            dice = (2. * inter + 1) / (union + 1)
            dice_sum += dice
        return 1 - dice_sum / class_num

class WeightDice(nn.Module):
    def __init__(self):
        super(WeightDice, self).__init__()

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        smooth = 1.
        num_sum = torch.sum(targets, dim=(0, 2, 3, 4))
        w = torch.Tensor([0, 0, 0]).cuda()
        for i in range(targets.size(1)):
            if (num_sum[i] < 1):
                w[i] = 0
            else:
                w[i] = (0.1 * num_sum[i] + num_sum[i - 1] + num_sum[i - 2] + 1) / (torch.sum(num_sum) + 1)
        print(w)
        inter = w * torch.sum(targets * probs, dim=(0, 2, 3, 4))
        inter = torch.sum(inter)

        union = w * torch.sum(targets + probs, dim=(0, 2, 3, 4))
        union = torch.sum(union)
        return 2. * (inter+smooth) / (union+smooth)

class WeightDiceLoss(nn.Module):
    def __init__(self):
        super(WeightDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 1.
        num_sum = torch.sum(targets, dim=(0, 2, 3, 4))
        w = torch.Tensor([0, 0, 0, 0]).cuda()
        for i in range(targets.size(1)):
            if (num_sum[i] < 1):
                w[i] = 0
            else:
                w[i] = (0.1 * num_sum[i] + num_sum[i - 1] + num_sum[i - 2] + 1) / (torch.sum(num_sum) + 1)
        # print(w)
        inter = w * torch.sum(targets * logits, dim=(0, 2, 3, 4))
        inter = torch.sum(inter)

        union = w * torch.sum(targets + logits, dim=(0, 2, 3, 4))
        union = torch.sum(union)
        return 1 - 2. * (inter+smooth) / (union+smooth)


class GeneralizedDice(nn.Module):
    def __init__(self):
        super(GeneralizedDice, self).__init__()

    def forward(self, logits, targets):
        smooth = 0.0000000001
        class_num = targets.shape[1]
        w = torch.zeros((class_num,1)).cuda()
        w = torch.sum(targets, dim=(0,2,3,4))
        w = 1 / (w ** 2 + smooth)
        inter = w * torch.sum(targets * logits, dim=(0, 2, 3, 4))
        union = w * torch.sum(targets + logits, dim=(0, 2, 3, 4))
        return  torch.mean(2. *(inter + smooth) / (union + smooth), dim=0)


class GeneralizedDiceLoss(nn.Module):
    def __init__(self):
        super(GeneralizedDiceLoss, self).__init__()

    def forward(self, logits, targets):
        smooth = 0.0000000001
        class_num = targets.shape[1]
        w = torch.sum(targets, dim=(0,2,3,4))
        w = 1 / (w ** 2 + smooth)
        inter = w * torch.sum(targets * logits, dim=(0, 2, 3, 4))
        union = w * torch.sum(targets + logits, dim=(0, 2, 3, 4))
        return  1. - torch.mean(2. *(inter + smooth) / (union + smooth), dim=0)

class diceloss(torch.nn.Module):
    def init(self):
        super(diceloss, self).init()
    def forward(self,pred, target):
       smooth = 1.
       iflat = pred[:, 1:, :].contiguous().view(-1)
       tflat = target[:, 1:, :].contiguous().view(-1)
       intersection = (iflat * tflat).sum()
       A_sum = torch.sum(iflat * iflat)
       B_sum = torch.sum(tflat * tflat)
       return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )

# generalized dice loss
# def generalized_dice_loss(y_true, y_pred):
#     # y_true,y_pred shape=[num_label,H,W,C]
#     smooth = 1
#     num_label = y_pred.shape[0]
#     w = torch.zeros((num_label,))
#     w = torch.sum(y_true, axis=(1, 2, 3))
#     w = 1 / (w ** 2 + 0.000001)
#     # Compute gen dice coef:
#     intersection_w = w * K.sum(y_true * y_pred, axis=[1, 2, 3])
#     union_w = w * K.sum(y_true + y_pred, axis=[1, 2, 3])
#     dice_coeff = K.mean((2. * intersection_w + smooth) / (union_w + smooth), axis=0)
#     return 1 - dice_coeff

def dice(logits, targets, class_index):
    inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
    union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
    dice = (2. * inter + 1) / (union + 1)
    return dice


def T(logits, targets):
    return torch.sum(targets[:, 2, :, :, :])


def P(logits, targets):
    return torch.sum(logits[:, 2, :, :, :])


def TP(logits, targets):
    return torch.sum(targets[:, 2, :, :, :] * logits[:, 2, :, :, :])


class KLLoss(nn.Module):
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, z_mean, z_var, input):
        m, c, H, W, D = input.shape
        n = c * H * W * D
        loss_KL = (1 / n) * torch.sum(torch.exp(z_var) + torch.square(z_mean) -1. -z_var , dim=-1)

        return torch.sum(loss_KL)/m


SUPPORTED_WEIGHTING = ['default', 'GDL']


class GeneralizedWassersteinDiceLoss(_Loss):
    """
    Generalized Wasserstein Dice Loss [1] in PyTorch.
    Optionally, one can use a weighting method for the
    class-specific sum of errors similar to the one used
    in the generalized Dice Loss [2].
    For this behaviour, please use weighting_mode='GDL'.
    The exact formula of the Wasserstein Dice loss in this case
    can be found in the Appendix of [3].
    References:
    ===========
    [1] "Generalised Wasserstein Dice Score for Imbalanced Multi-class
        Segmentation using Holistic Convolutional Networks."
        Fidon L. et al. MICCAI BrainLes (2017).
    [2] "Generalised dice overlap as a deep learning loss function
        for highly unbalanced segmentations."
        Sudre C., et al. MICCAI DLMIA (2017).
    [3] "Comparative study of deep learning methods for the automatic
        segmentation of lung, lesion and lesion type in CT scans of
        COVID-19 patients."
        Tilborghs, S. et al. arXiv preprint arXiv:2007.15546 (2020).
    """
    def __init__(self, dist_matrix, weighting_mode='default', reduction='mean'):
        """
        :param dist_matrix: 2d tensor or 2d numpy array; matrix of distances
        between the classes.
        It must have dimension C x C where C is the number of classes.
        :param: weighting_mode: str; indicates how to weight the class-specific
        sum of errors.
        'default' corresponds to the GWDL used in the original paper [1],
        'GDL' corresponds to the GWDL used in [2].
        :param reduction: str; reduction mode.
        References:
        ===========
        [1] "Generalised Wasserstein Dice Score for Imbalanced Multi-class
            Segmentation using Holistic Convolutional Networks."
            Fidon L. et al. MICCAI BrainLes (2017).
        [2] "Comparative study of deep learning methods for the automatic
            segmentation of lung, lesion and lesion type in CT scans of
            COVID-19 patients."
            Tilborghs, S. et al. arXiv preprint arXiv:2007.15546 (2020).
        """
        super(GeneralizedWassersteinDiceLoss, self).__init__(
            reduction=reduction)
        self.M = dist_matrix
        if isinstance(self.M, np.ndarray):
            self.M = torch.from_numpy(self.M)
        if torch.cuda.is_available():
            self.M = self.M.cuda()
        if torch.max(self.M) != 1:
            print('Normalize the maximum of the distance matrix '
                  'used in the Generalized Wasserstein Dice Loss to 1.')
            self.M = self.M / torch.max(self.M)
        self.num_classes = self.M.size(0)
        self.alpha_mode = weighting_mode
        assert weighting_mode in SUPPORTED_WEIGHTING, \
            "weighting_mode must be in %s" % str(SUPPORTED_WEIGHTING)
        self.reduction = reduction

    def forward(self, input, target):
        """
        Compute the Generalized Wasserstein Dice loss
        between input and target tensors.
        :param input: tensor. input is the scores maps (before softmax).
        The expected shape of input is (N, C, H, W, D) in 3d
        and (N, C, H, W) in 2d.
        :param target: target is the target segmentation.
        The expected shape of target is (N, H, W, D) or (N, 1, H, W, D) in 3d
        and (N, H, W) or (N, 1, H, W) in 2d.
        :return: scalar tensor. Loss function value.
        """
        epsilon = np.spacing(1)  # smallest number available
        # Convert the target segmentation to long if needed
        target = target.long()
        # Aggregate spatial dimensions
        flat_input = input.view(input.size(0), input.size(1), -1)  # b,c,s
        flat_target = target.view(target.size(0), -1)  # b,s
        # Apply the softmax to the input scores map
        probs = F.softmax(flat_input, dim=1)  # b,c,s
        # Compute the Wasserstein distance map
        wass_dist_map = self.wasserstein_distance_map(probs, flat_target)
        # Compute the generalised number of true positives
        alpha = self.compute_alpha_generalized_true_positives(flat_target)

        # Compute the Generalized Wasserstein Dice loss
        if self.alpha_mode == 'GDL':
            # use GDL-style alpha weights (i.e. normalize by the volume of each class)
            # contrary to [1] we also use alpha in the "generalized all error".
            true_pos = self.compute_generalized_true_positive(
                alpha, flat_target, wass_dist_map)
            denom = self.compute_denominator(alpha, flat_target, wass_dist_map)
        else:  # default: as in [1]
            # (i.e. alpha=1 for all foreground classes and 0 for the background).
            # Compute the generalised number of true positives
            true_pos = self.compute_generalized_true_positive(
                alpha, flat_target, wass_dist_map)
            all_error = torch.sum(wass_dist_map, dim=1)
            denom = 2 * true_pos + all_error
        wass_dice = (2. * true_pos + epsilon) / (denom + epsilon)
        wass_dice_loss = 1. - wass_dice

        if self.reduction == 'sum':
            return wass_dice_loss.sum()
        elif self.reduction == 'none':
            return wass_dice_loss
        else:  # default is mean reduction
            return wass_dice_loss.mean()

    def wasserstein_distance_map(self, flat_proba, flat_target):
        """
        Compute the voxel-wise Wasserstein distance (eq. 6 in [1]) for
        the flattened prediction and the flattened labels (ground_truth)
        with respect to the distance matrix on the label space M.
        References:
        ===========
        [1] "Generalised Wasserstein Dice Score for Imbalanced Multi-class
        Segmentation using Holistic Convolutional Networks",
        Fidon L. et al. MICCAI BrainLes 2017
        """
        # Turn the distance matrix to a map of identical matrix
        M_extended = torch.unsqueeze(self.M, dim=0)  # C,C -> 1,C,C
        M_extended = torch.unsqueeze(M_extended, dim=3)  # 1,C,C -> 1,C,C,1
        M_extended = M_extended.expand((
            flat_proba.size(0),
            M_extended.size(1),
            M_extended.size(2),
            flat_proba.size(2)
        ))
        # Expand the feature dimensions of the target
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)  # b,s -> b,1,s
        flat_target_extended = flat_target_extended.expand(  # b,1,s -> b,C,s
            (flat_target.size(0), M_extended.size(1), flat_target.size(1))
        )
        flat_target_extended = torch.unsqueeze(flat_target_extended, dim=1)  # b,C,s -> b,1,C,s
        # Extract the vector of class distances for the ground-truth label at each voxel
        M_extended = torch.gather(M_extended, dim=1, index=flat_target_extended)  # b,C,C,s -> b,1,C,s
        M_extended = torch.squeeze(M_extended, dim=1)  # b,1,C,s -> b,C,s
        # Compute the wasserstein distance map
        wasserstein_map = M_extended * flat_proba
        # Sum over the classes
        wasserstein_map = torch.sum(wasserstein_map, dim=1)  # b,C,s -> b,s
        return wasserstein_map

    def compute_generalized_true_positive(self, alpha, flat_target, wasserstein_distance_map):
        # Extend alpha to a map and select value at each voxel according to flat_target
        alpha_extended = torch.unsqueeze(alpha, dim=2)  # b,C -> b,C,1
        alpha_extended = alpha_extended.expand(  # b,C,1 -> b,C,s
            (flat_target.size(0), self.num_classes, flat_target.size(1))
        )
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)  # b,s -> b,1,s
        alpha_extended = torch.gather(
            alpha_extended, index=flat_target_extended, dim=1)  # b,C,s -> b,1,s

        # Compute the generalized true positive as in eq. 9 of [1]
        generalized_true_pos = torch.sum(
            alpha_extended * (1. - wasserstein_distance_map),
            dim=[1, 2],
        )
        return generalized_true_pos

    def compute_denominator(self, alpha, flat_target, wasserstein_distance_map):
        # Extend alpha to a map and select value at each voxel according to flat_target
        alpha_extended = torch.unsqueeze(alpha, dim=2)  # b,C -> b,C,1
        alpha_extended = alpha_extended.expand(  # b,C,1 -> b,C,s
            (flat_target.size(0), self.num_classes, flat_target.size(1))
        )
        flat_target_extended = torch.unsqueeze(flat_target, dim=1)  # b,s -> b,1,s
        alpha_extended = torch.gather(
            alpha_extended, index=flat_target_extended, dim=1)  # b,C,s -> b,1,s
        # Compute the generalized true positive as in eq. 9
        generalized_true_pos = torch.sum(
            alpha_extended * (2. - wasserstein_distance_map),
            dim=[1, 2],
        )
        return generalized_true_pos

    def compute_alpha_generalized_true_positives(self, flat_target):
        """
        Compute the weights \alpha_l of eq. 9 in [1].
        References:
        ===========
        [1] "Generalised Wasserstein Dice Score for Imbalanced Multi-class
        Segmentation using Holistic Convolutional Networks",
        Fidon L. et al. MICCAI BrainLes 2017.
        """
        if self.alpha_mode == 'GDL':  # GDL style
            # Define alpha like in the generalized dice loss
            # i.e. the inverse of the volume of each class.
            # Convert target to one-hot class encoding.
            one_hot = F.one_hot(  # shape: b,c,s
                flat_target, num_classes=self.num_classes).permute(0, 2, 1).float()
            volumes = torch.sum(one_hot, dim=2)  # b,c
            alpha = 1. / (volumes + 1.)
        else:  # default, i.e. as in [1]
            # alpha weights are 0 for the background and 1 otherwise
            alpha_np = np.ones((flat_target.size(0), self.num_classes))  # b,c
            alpha_np[:, 0] = 0.
            alpha = torch.from_numpy(alpha_np).float()
            if torch.cuda.is_available():
                alpha = alpha.cuda()
        return alpha

# from torch.autograd import Variable
#
# class FocalLoss(nn.Module):
#     r"""
#         This criterion is a implemenation of Focal Loss, which is proposed in
#         Focal Loss for Dense Object Detection.
#
#             Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
#
#         The losses are averaged across observations for each minibatch.
#
#         Args:
#             alpha(1D Tensor, Variable) : the scalar factor for this criterion
#             gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
#                                    putting more focus on hard, misclassiﬁed examples
#             size_average(bool): By default, the losses are averaged over observations for each minibatch.
#                                 However, if the field size_average is set to False, the losses are
#                                 instead summed for each minibatch.
#
#
#     """
#     def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
#         super(FocalLoss, self).__init__()
#         if alpha is None:
#             self.alpha = Variable(torch.ones(class_num, 1))
#         else:
#             if isinstance(alpha, Variable):
#                 self.alpha = alpha
#             else:
#                 self.alpha = Variable(alpha)
#         self.gamma = gamma
#         self.class_num = class_num
#         self.size_average = size_average
#
#     def forward(self, inputs, targets):
#         N = inputs.size(0)
#         C = inputs.size(1)
#         # P = F.softmax(inputs)
#         P = nn.Softmax(dim=1)(inputs)
#
#         class_mask = inputs.data.new(N, C).fill_(0)
#         class_mask = Variable(class_mask)
#         ids = targets.view(-1, 1)
#         class_mask.scatter_(1, ids.data, 1.)
#         #print(class_mask)
#
#
#         if inputs.is_cuda and not self.alpha.is_cuda:
#             self.alpha = self.alpha.cuda()
#         alpha = self.alpha[ids.data.view(-1)]
#
#         probs = (P*class_mask).sum(1).view(-1,1)
#
#         log_p = probs.log()
#         #print('probs size= {}'.format(probs.size()))
#         #print(probs)
#
#         batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
#         #print('-----bacth_loss------')
#         #print(batch_loss)
#
#
#         if self.size_average:
#             loss = batch_loss.mean()
#         else:
#             loss = batch_loss.sum()
#         return loss


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    r"""
    if gamma == 0, equal to CrossEntropyLoss
    default = 2, best in paper.
    """

    def __init__(self, gamma: float = 0, alpha: list = None, size_average: bool = True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        if alpha is not None:
            self.alpha = torch.Tensor(alpha)

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class SoftDiceLoss(_Loss):
    '''
    Soft_Dice = 2*|dot(A, B)| / (|dot(A, A)| + |dot(B, B)| + eps)
    eps is a small constant to avoid zero division,
    '''

    def __init__(self, *args, **kwargs):
        super(SoftDiceLoss, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-8):
        intersection = torch.sum(torch.mul(y_pred, y_true))
        union = torch.sum(torch.mul(y_pred, y_pred)) + torch.sum(torch.mul(y_true, y_true)) + eps

        dice = 2 * intersection / union
        dice_loss = 1 - dice

        return dice_loss

class CustomKLLoss(_Loss):
    '''
    KL_Loss = (|dot(mean , mean)| + |dot(std, std)| - |log(dot(std, std))| - 1) / N
    N is the total number of image voxels
    '''

    def __init__(self, *args, **kwargs):
        super(CustomKLLoss, self).__init__()

    def forward(self, mean, std):
        return torch.mean(torch.mul(mean, mean)) + torch.mean(torch.mul(std, std)) - torch.mean(
            torch.log(torch.mul(std, std))) - 1

class CombinedLoss(_Loss):
    '''
    Combined_loss = Dice_loss + k1 * L2_loss + k2 * KL_loss
    As default: k1=0.1, k2=0.1
    '''

    def __init__(self, k1=0.1, k2=0.1):
        super(CombinedLoss, self).__init__()
        self.k1 = k1
        self.k2 = k2
        self.dice_loss = SoftDiceLoss()
        self.l2_loss = nn.MSELoss()
        self.kl_loss = CustomKLLoss()

    def forward(self, y_pred, y_true, y_mid):
        est_mean, est_std = (y_mid[:, :128], y_mid[:, 128:])
        seg_pred, seg_truth = (y_pred[:, 0, :, :, :], y_true[:, 0, :, :, :])
        vae_pred, vae_truth = (y_pred[:, 1:, :, :, :], y_true[:, 1:, :, :, :])
        dice_loss = self.dice_loss(seg_pred, seg_truth)
        l2_loss = self.l2_loss(vae_pred, vae_truth)
        kl_div = self.kl_loss(est_mean, est_std)
        combined_loss = dice_loss + self.k1 * l2_loss + self.k2 * kl_div
        # print("dice_loss:%.4f, L2_loss:%.4f, KL_div:%.4f, combined_loss:%.4f"%(dice_loss,l2_loss,kl_div,combined_loss))

        return combined_loss

