import sys
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from utils.dataset import braTS_DataSet
from torch.utils.data import DataLoader
import torch.optim as optim
from models.models import *
from utils import mymetrics
from torch.autograd import Variable
from utils.common import *


# v5:Discriminator kernel:3 G:GeneratorUNet_V3  0.87 0.80 0.76
# v7:Discriminator kernel:4 G:GeneratorUNet_V3


def sample_voxel_volumes(val_loader, model):

    """Saves a generated sample from the validation set"""
    file_path = '/media/tiger/Disk0/lyu/DATA/Brain/MICCAI_BraTS2020_TrainingData_corr/'
    Dice_WT1 = 0
    Dice_TC1 = 0
    Dice_ET1 = 0
    Dice_WT2 = 0
    Dice_TC2 = 0
    Dice_ET2 = 0
    Dice = 0
    Cls = []
    Pred_c = []
    # test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=1, shuffle=True)

    model.eval()
    with torch.no_grad():
        for batch_idx, (data, target,cls,_) in enumerate(val_loader):
            if batch_idx>10:
                break
            img, mask = Variable(data.float().cuda()), Variable(target.float().cuda())
            pred_mask1, pred_mask2 = model(img)
            mask = mask2class(mask).cuda()
            pred_mask1 = mask2class(pred_mask1).cuda()
            pred_mask2 = mask2class(pred_mask2).cuda()
            fake_WT1, fake_TC1, fake_ET1 = pred_mask1[:, 0, :], pred_mask1[:, 1, :], pred_mask1[:, 2, :]
            fake_WT2, fake_TC2, fake_ET2 = pred_mask2[:, 0, :], pred_mask2[:, 1, :], pred_mask2[:, 2, :]
            real_WT, real_TC, real_ET = mask[:, 0, :], mask[:, 1, :], mask[:, 2, :]

            dice_WT1 = mymetrics.SoftDice()(fake_WT1, real_WT)
            dice_TC1 = mymetrics.SoftDice()(fake_TC1, real_TC)
            dice_ET1 = mymetrics.SoftDice()(fake_ET1, real_ET)
            Dice_WT1 = Dice_WT1 + dice_WT1.item()
            Dice_TC1 = Dice_TC1 + dice_TC1.item()
            Dice_ET1 = Dice_ET1 + dice_ET1.item()

            dice_WT2 = mymetrics.SoftDice()(fake_WT2, real_WT)
            dice_TC2 = mymetrics.SoftDice()(fake_TC2, real_TC)
            dice_ET2 = mymetrics.SoftDice()(fake_ET2, real_ET)
            Dice_WT2 = Dice_WT2 + dice_WT2.item()
            Dice_TC2 = Dice_TC2 + dice_TC2.item()
            Dice_ET2 = Dice_ET2 + dice_ET2.item()

    # test_num = len(val_loader)
    test_num = 11
    Dice_WT1 = Dice_WT1/test_num
    Dice_TC1 = Dice_TC1/test_num
    Dice_ET1 = Dice_ET1/test_num
    Dice_WT2 = Dice_WT2/test_num
    Dice_TC2 = Dice_TC2/test_num
    Dice_ET2 = Dice_ET2/test_num
    # Dice = Dice/test_num
    print('Test~~~ Dice_WT1:', Dice_WT1, 'Dice_TC1:', Dice_TC1, 'Dice_ET1:', Dice_ET1,
          'Dice_WT2:', Dice_WT2, 'Dice_TC2:', Dice_TC2, 'Dice_ET2:', Dice_ET2,)
    torch.cuda.empty_cache()
    return (Dice_WT2+Dice_TC2+Dice_ET2)/3.0

def train():
    torch.backends.cudnn.benchmark = True
    # version = 'ru_v1'
    version = 'w_withoutvae_v2_f4'

    dice_max = 0
    args = config.args
    data_path = args.dataset_path
    os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'

    # data info
    train_set = braTS_DataSet(args.crop_size, data_path, mode='train', fold='4')
    val_set = braTS_DataSet(args.crop_size, data_path, mode='val', fold='4')
    train_loader = DataLoader(dataset=train_set, batch_size=2, num_workers=8, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_set, batch_size=1, num_workers=1, shuffle=False)

    save_path = '/media/tiger/Disk0/lyu/DATA/Brain/output_2020/{}/model.pkl'.format(version)

    # model info
    model = WNet_withoutVAE()

    epoch_initial = 0
    if os.path.exists(save_path):
        # gpu1->gpu0
        # torch.load(save_path_g, map_location={'cuda:1':'cuda:0'})
        checkpoint = torch.load(save_path,map_location=lambda storage, loc:storage)

        model.load_state_dict(checkpoint['model'])
        if len(os.environ['CUDA_VISIBLE_DEVICES']) == 1:
            model.cuda()
        else:
            model =  nn.DataParallel(model).cuda()

        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999),weight_decay=1e-5)
        optimizer.load_state_dict(checkpoint['optimizer'])

        epoch_initial = checkpoint['epoch']
        # avg_cost = checkpoint['avg_cost']
        print('load epoch {}~'.format(epoch_initial))
    else:
        if len(os.environ['CUDA_VISIBLE_DEVICES']) == 1:
            model.cuda()
        else:
            model = nn.DataParallel(model).cuda()

        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999),weight_decay=1e-5)
    #     avg_cost = np.zeros([400, 4], dtype=np.float32)
    #
    # lambda_weight = np.ones([2, 400])


    for epoch in range(epoch_initial, args.epochs + 1):
        # time_start = time.time()
        adjust_learning_rate(optimizer, epoch, args, args.lr)

        for batch_idx, (data, target, _, name) in enumerate(train_loader):
            img, mask = Variable(data.float().cuda()), Variable(target.float().cuda())
            # loss_mse = torch.nn.MSELoss().cuda()
            optimizer.zero_grad()

            pred_mask1, pred_mask2 = model(img)

            # losses
            # S1
            loss_dice1 = mymetrics.GeneralizedDiceLoss().cuda()(pred_mask1, mask)
            loss_focal1 = mymetrics.FocalLoss(gamma=2).cuda()(pred_mask1, torch.argmax(mask, dim=1))
            loss_segment1 = (2 * loss_dice1 + loss_focal1) / 3.0

            # S2
            loss_dice2 = mymetrics.GeneralizedDiceLoss().cuda()(pred_mask2, mask)
            loss_focal2 = mymetrics.FocalLoss(gamma=2).cuda()(pred_mask2, torch.argmax(mask, dim=1))
            loss_segment2 = (2 * loss_dice2 + loss_focal2) / 3.0

            loss = loss_segment1+loss_segment2
            # loss = loss_segment1
            loss.backward()
            optimizer.step()
            # batches_done = epoch * len(train_loader) + batch_idx

            # avg_cost[epoch,0] = avg_cost[epoch,0]+loss_dice.detach()
            # avg_cost[epoch, 1] = avg_cost[epoch, 1] + loss_focal.detach()
            # avg_cost[epoch, 2] = avg_cost[epoch, 2] + loss_rebuild.detach()
            # avg_cost[epoch, 3] = avg_cost[epoch, 3] + loss_KL.detach()

            # Print log
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d][Dice Loss1: %f,Dice Loss2: %f]"
                % (
                    epoch,
                    args.epochs,
                    batch_idx+1,
                    len(train_loader),
                    loss_dice1.detach(),
                    loss_dice2.detach(),
                )
            )

        # avg_cost[epoch, 0] = avg_cost[epoch, 0]/len(train_loader)
        # avg_cost[epoch, 1] = avg_cost[epoch, 1]/len(train_loader)
        # avg_cost[epoch, 2] = avg_cost[epoch, 2] / len(train_loader)
        # avg_cost[epoch, 3] = avg_cost[epoch, 3] / len(train_loader)
        #
        # summary_path = '/media/tiger/Disk0/lyu/DATA/Brain/output_2020/runs/{}'.format(version)
        # with SummaryWriter(pathexists(summary_path)) as w:
        #     w.add_scalar('Dice_loss', avg_cost[epoch,0], epoch)
        #     w.add_scalar('Focal_loss', avg_cost[epoch, 1], epoch)
        #     w.add_scalar('Rebuild_loss', avg_cost[epoch, 2], epoch)
        #     w.add_scalar('KL_loss', avg_cost[epoch, 3], epoch)

        # If at sample interval save image
        save_path2 = '/media/tiger/Disk0/lyu/DATA/Brain/output_2020/{}/'.format(version)
        #
        if epoch>=0 :
            print('\n*****volumes sampled*****')
            dice = sample_voxel_volumes(val_loader, model)
            if dice >= dice_max:
                dice_max = dice
                if len(os.environ['CUDA_VISIBLE_DEVICES']) > 1:
                    state = {'model': model.module.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                else:
                    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
                torch.save(state, pathexists(save_path2)+'model.pkl')



if __name__ == '__main__':
    train()

