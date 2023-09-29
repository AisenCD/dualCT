import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import os
from progressBar import printProgressBar
import scipy.io as sio
from scipy import ndimage
from Metrics import SegmenationMetrics
from hausdorff import hausdorff_distance    


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


class computeDiceOneHotBinary(nn.Module):
    def __init__(self):
        super(computeDiceOneHotBinary, self).__init__()

    def dice(self, input, target):
        inter = (input * target).float().sum()
        sum = input.sum() + target.sum()
        if (sum == 0).all():
            return (2 * inter + 1e-8) / (sum + 1e-8)

        return 2 * (input * target).float().sum() / (input.sum() + target.sum())

    def inter(self, input, target):
        return (input * target).float().sum()

    def sum(self, input, target):
        return input.sum() + target.sum()

    def forward(self, pred, GT):
        # pred is converted to 0 and 1
        batchsize = GT.size(0)
        DiceB = to_var(torch.zeros(batchsize, 2))
        DiceF = to_var(torch.zeros(batchsize, 2))
        
        for i in range(batchsize):
            DiceB[i, 0] = self.inter(pred[i, 0], GT[i, 0])
            DiceF[i, 0] = self.inter(pred[i, 1], GT[i, 1])
           
            DiceB[i, 1] = self.sum(pred[i, 0], GT[i, 0])
            DiceF[i, 1] = self.sum(pred[i, 1], GT[i, 1])
           
        return DiceB, DiceF 
        
        
def DicesToDice(Dices):
    sums = Dices.sum(dim=0)
    return (2 * sums[0] + 1e-8) / (sums[1] + 1e-8)


def getSingleImageBin(pred):
    # input is a 2-channels image corresponding to the predictions of the net
    # output is a gray level image (1 channel) of the segmentation with "discrete" values
    n_channels = 2
    Val = to_var(torch.zeros(2))
    Val[1] = 1.0
    x = predToSegmentation(pred)
    out = x * Val.view(1, n_channels, 1, 1)
    return out.sum(dim=1, keepdim=True)
    

def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return (x == 1).float()



def getOneHotSegmentation(batch):
    backgroundVal = 0
    # IVD
    label1 = 1.0
    oneHotLabels = torch.cat((batch == backgroundVal, batch == label1), 
                             dim=1)
                             
    return oneHotLabels.float()


def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    spineLabel = 1.0
    return (batch / spineLabel).round().long().squeeze()


def saveImages(model, val_loader_save_images, batch_size_val_save, epoch, modelName,record_time,mode_list):
    # record_time = '2022-4-12-3-50_'
    # epoch = 7
    path = os.path.join(r'D:\workspace\data\result\dual_ct\result_images',record_time + '_epoch_'+ str(epoch)) 
    if not os.path.exists(path):
        os.makedirs(path)
        
    total = len(val_loader_save_images)
    model.eval()
    softMax = nn.Softmax()
    
    for i, data in enumerate(val_loader_save_images):
        printProgressBar(i, total, prefix="Saving images.....", length=30)
        image_ze,image_iod,image_ionw,image_mono70,image_mono40,image_mono100, labels, img_names = data

        # Be sure here your image is betwen [0,1]
        image_ze=image_ze.type(torch.FloatTensor)
        image_iod=image_iod.type(torch.FloatTensor)
        image_ionw=image_ionw.type(torch.FloatTensor)
        image_mono70=image_mono70.type(torch.FloatTensor)
        image_mono70=image_mono70.type(torch.FloatTensor)
        image_mono70=image_mono70.type(torch.FloatTensor)

        labels = labels.numpy()
        idx=np.where(labels>0.0)
        labels[idx]=1.0
        labels = torch.from_numpy(labels)
        labels = labels.type(torch.FloatTensor)

        mode_list_name = ''
        if len(mode_list)==1:
            # mode_list = ['ze']
            mode_list_name = mode_list[0]+'_'
            if mode_list[0] == 'ze':
                MRI = to_var(image_ze)
            elif mode_list[0] == 'iod':
                MRI = to_var(image_iod)                   
            elif mode_list[0] == 'ionw':
                MRI = to_var(image_ionw)
            elif mode_list[0] == 'mono40':
                MRI = to_var(image_mono40)
            elif mode_list[0] == 'mono70':
                MRI = to_var(image_mono70)
            elif mode_list[0] == 'mono100':
                MRI = to_var(image_mono100)

        elif len(mode_list)>1:
            # mode_list = ['ze','iod']
            t = ()
            for mode_1 in mode_list:
                # print(mode_1)
                mode_list_name = mode_list_name+mode_1+'_'
                if mode_1 == 'ze':
                    t = t+(image_ze,)
                elif mode_1 == 'iod':
                    t = t+(image_iod,)                    
                elif mode_1 == 'ionw':
                    t = t+(image_ionw,)
                elif mode_1 == 'mono40':
                    t = t+(image_mono40,)
                elif mode_1 == 'mono70':
                    t = t+(image_mono70,)
                elif mode_1 == 'mono100':
                    t = t+(image_mono100,)
            MRI = to_var(torch.cat(t,dim=1))

        # images = torch.cat((image_f,image_i,image_o,image_w),dim=1)
        # MRI = to_var(images)
        image_1_var = to_var(image_ze)
        image_2_var = to_var(image_iod)
        Segmentation = to_var(labels)
            
        segmentation_prediction = model(MRI)

        pred_y = softMax(segmentation_prediction)
        segmentation = getSingleImageBin(pred_y)
        segmentation_prediction_ones = predToSegmentation(segmentation_prediction)
        # Segmentation_planes = getOneHotSegmentation(Segmentation)
        # Dice_ = computeDiceOneHotBinary()
        # DicesB, DicesF = Dice_(segmentation_prediction_ones, Segmentation_planes)
        # DiceB = DicesToDice(DicesB)
        # DiceF = DicesToDice(DicesF)
        
        
        imgname = os.path.split(img_names[0])[1].split('.')[0] 
        #img_names[0].split('/Fat/')
        # imgname = imgname[1].split('_fat.png')
        # imgname = r'test_file'
        
        # out = torch.cat((image_f_var, segmentation, Segmentation*255))
        out = torch.cat((image_1_var, image_2_var, segmentation_prediction_ones[:,1:2,:,:], Segmentation*255))
        # out_add_1 = image_1_var + Segmentation*255
        # out_add_4 = image_2_var + Segmentation*0.2 + segmentation_prediction_ones[:,1:2,:,:]*0.5
        
        torchvision.utils.save_image(out.data, os.path.join(path,imgname + '.png'),
                                     nrow=batch_size_val_save,
                                     padding=2,
                                     normalize=False,
                                     range=None,
                                     scale_each=False)

        torchvision.utils.save_image(image_1_var.data, os.path.join(path,imgname + '_image_1.png'),
                                     nrow=batch_size_val_save,
                                     padding=2,
                                     normalize=False,
                                     range=None,
                                     scale_each=False)

        torchvision.utils.save_image(image_2_var.data, os.path.join(path,imgname + '_image_2.png'),
                                     nrow=batch_size_val_save,
                                     padding=2,
                                     normalize=False,
                                     range=None,
                                     scale_each=False)

        torchvision.utils.save_image(segmentation_prediction_ones[:,1:2,:,:].data, os.path.join(path,imgname + '_pred_1.png'),
                                     nrow=batch_size_val_save,
                                     padding=2,
                                     normalize=False,
                                     range=None,
                                     scale_each=False)

        torchvision.utils.save_image((Segmentation*255).data, os.path.join(path,imgname + '_label_1.png'),
                                     nrow=batch_size_val_save,
                                     padding=2,
                                     normalize=False,
                                     range=None,
                                     scale_each=False)

        # torchvision.utils.save_image(out_add_1.data, os.path.join(path,imgname + '_out_add_1.png'),
        #                              nrow=batch_size_val_save,
        #                              padding=2,
        #                              normalize=False,
        #                              range=None,
        #                              scale_each=False)

        # torchvision.utils.save_image(out_add_2.data, os.path.join(path,imgname + '_out_add_2.png'),
        #                              nrow=batch_size_val_save,
        #                              padding=2,
        #                              normalize=False,
        #                              range=None,
        #                              scale_each=False)

        # torchvision.utils.save_image(out_add_4.data, os.path.join(path,imgname + '_out_add_4.png'),
        #                              nrow=batch_size_val_save,
        #                              padding=2,
        #                              normalize=False,
        #                              range=None,
        #                              scale_each=False)

        # torchvision.utils.save_image(out_add_4.data, os.path.join(path,imgname + '_out_add_4.png'),
        #                              nrow=batch_size_val_save,
        #                              padding=2,
        #                              normalize=False,
        #                              range=None,
        #                              scale_each=False)


                                     
    printProgressBar(total, total, done="Images saved !")
   

def dice_coef(output, target):#output为预测结果 target为真实结果
    smooth = 1e-5 #防止0除
     
    intersection = (output * target).sum()
 
    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)


def iou_score(output, target):
    smooth = 1e-5
 
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
 
    return (intersection + smooth) / (union + smooth)



def sensitivity(output, target):
    smooth = 1e-5
  
    intersection = (output * target).sum()
 
    return (intersection + smooth) / \
        (target.sum() + smooth)



def ppv(output, target):
    smooth = 1e-5
  
    intersection = (output * target).sum()
 
    return (intersection + smooth) / \
        (output.sum() + smooth)



   
#import copy
#img_batch = copy.copy(val_loader)    
   
def inference(model, img_batch, batch_size, epoch, mode_list):
    total = len(img_batch)
    # total = len(val_loader)
    metric = SegmenationMetrics()

    Dice1 = torch.zeros(total, 2)
    
    list_metric_Dice1_1 = []
    list_metric_dice_1 = []
    list_metric_iou_1 = []
    list_metric_sensitivity_1 = []
    list_metric_ppv_1 = []
    list_metric_hausdorff_1 = []    

    
    model.eval()
    
    dice = computeDiceOneHotBinary().cuda()
    softMax = nn.Softmax().cuda()

    img_names_ALL = []
    for i, data in enumerate(img_batch):
        print(i)
        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        image_ze,image_iod,image_ionw,image_mono70,image_mono40,image_mono100, labels, img_names = data

        # # Be sure here your image is betwen [0,1]
        image_ze=image_ze.type(torch.FloatTensor)
        image_iod=image_iod.type(torch.FloatTensor)
        image_ionw=image_ionw.type(torch.FloatTensor)
        image_mono70=image_mono70.type(torch.FloatTensor)
        image_mono70=image_mono70.type(torch.FloatTensor)
        image_mono70=image_mono70.type(torch.FloatTensor)

        # 配置转移和未转移是否归为同一类
        labels = labels.numpy()
        idx=np.where(labels>0.0)
        labels[idx]=1.0
        labels = torch.from_numpy(labels)
        labels = labels.type(torch.FloatTensor)


        if len(mode_list)==1:
            if mode_list[0] == 'ze':
                MRI = to_var(image_ze)
            elif mode_list[0] == 'iod':
                MRI = to_var(image_iod)                   
            elif mode_list[0] == 'ionw':
                MRI = to_var(image_ionw)
            elif mode_list[0] == 'mono40':
                MRI = to_var(image_mono40)
            elif mode_list[0] == 'mono70':
                MRI = to_var(image_mono70)
            elif mode_list[0] == 'mono100':
                MRI = to_var(image_mono100)

        elif len(mode_list)>1:
            t = ()
            for mode_1 in mode_list:
                if mode_1 == 'ze':
                    t = t+(image_ze,)
                elif mode_1 == 'iod':
                    t = t+(image_iod,)                    
                elif mode_1 == 'ionw':
                    t = t+(image_ionw,)
                elif mode_1 == 'mono40':
                    t = t+(image_mono40,)
                elif mode_1 == 'mono70':
                    t = t+(image_mono70,)
                elif mode_1 == 'mono100':
                    t = t+(image_mono100,)
            MRI = to_var(torch.cat(t,dim=1))

        # MRI = to_var(image_f)
        # MRI = to_var(torch.cat((image_f,image_i),dim=1))
        Segmentation = to_var(labels)
        segmentation_prediction = model(MRI)

        Segmentation_planes = getOneHotSegmentation(Segmentation)

        pred_y = softMax(segmentation_prediction)
        segmentation_prediction_ones = predToSegmentation(pred_y)


        # segmentation_prediction_ones = predToSegmentation(segmentation_prediction)
        

        DicesN, Dices1 = dice(segmentation_prediction_ones, Segmentation_planes)
        metric_Dice1_1 = (2*Dices1.data[0][0]/Dices1.data[0][1]).cpu().numpy()
        
        # Dice1[i] = Dices1.data        
        # accuracy = metric.ClassifyMetric(Segmentation_planes, segmentation_prediction_ones)
        # print('accuracy: ',accuracy)

        # sums = Dices1.sum(dim=0)

        # intersection = (value_output * value_target).sum()     
        # return (2. * intersection + smooth) / \
        #     (output.sum() + target.sum() + smooth)        
        value_output = segmentation_prediction_ones[0,1,:,:].data.cpu().numpy()
        value_target = Segmentation_planes[0,1,:,:].data.cpu().numpy()

        metric_dice_1 = dice_coef(value_output, value_target)
        metric_iou_1 = iou_score(value_output, value_target)
        metric_sensitivity_1 = sensitivity(value_output, value_target)
        metric_ppv_1 = ppv(value_output, value_target)
        metric_hausdorff_1 = \
            hausdorff_distance(value_output, value_target, distance="euclidean")

        list_metric_Dice1_1.append(metric_Dice1_1)
        list_metric_dice_1.append(metric_dice_1)
        list_metric_iou_1.append(metric_iou_1)
        list_metric_sensitivity_1.append(metric_sensitivity_1)
        list_metric_ppv_1.append(metric_ppv_1)
        list_metric_hausdorff_1.append(metric_hausdorff_1)
        

    printProgressBar(total, total, done="[Inference] Segmentation Done !")
    
    # ValDice1 = DicesToDice(Dice1)

    ValDice1 = np.mean(list_metric_Dice1_1)
    val_dice_1 = np.mean(list_metric_dice_1)
    val_iou_1 = np.mean(list_metric_iou_1)
    val_sensitivity_1 = np.mean(list_metric_sensitivity_1)
    val_ppv_1 = np.mean(list_metric_ppv_1)
    val_hausdorff_1 = np.mean(list_metric_hausdorff_1)
    

    return [ValDice1], val_dice_1, val_iou_1, val_sensitivity_1, val_ppv_1, val_hausdorff_1



# # t_1 = torch.randn(1,2,192,192).cuda()
# t_1 = torch.zeros(1,2,192,192).cuda()
# t_1_seg = predToSegmentation(t_1)

# t_2 = torch.rand(1,2,192,192).cuda()
# t_2_seg = predToSegmentation(t_2)

# dice_coef(t_1_seg, t_2_seg)
# dice_coef(t_1_seg[0,1,:,:], t_2_seg[0,1,:,:])
# iou_score(t_1_seg[0,1,:,:], t_2_seg[0,1,:,:])
# hausdorff_distance(t_1_seg[0,1,:,:].data.cpu().numpy(), t_2_seg[0,1,:,:].data.cpu().numpy(), distance="euclidean")

# # t_1_seg[0,1,:,:].data.cpu().numpy()
# # torch.sigmoid(t_1_seg[0,1,:,:]).data.cpu().numpy()


 
# # two random 2D arrays (second dimension must match)
# np.random.seed(0)
# X = np.random.random((1000,100))
# Y = np.random.random((5000,100))
 
# # Test computation of Hausdorff distance with different base distances
# print("Hausdorff distance test: {0}".format( hausdorff_distance(X, Y, distance="manhattan") ))
# print("Hausdorff distance test: {0}".format( hausdorff_distance(X, Y, distance="euclidean") ))
# print("Hausdorff distance test: {0}".format( hausdorff_distance(X, Y, distance="chebyshev") ))
# print("Hausdorff distance test: {0}".format( hausdorff_distance(X, Y, distance="cosine") ))
 
# # For haversine, use 2D lat, lng coordinates
# def rand_lat_lng(N):
#     lats = np.random.uniform(-90, 90, N)
#     lngs = np.random.uniform(-180, 180, N)
#     return np.stack([lats, lngs], axis=-1)
        
# X = rand_lat_lng(100)
# Y = rand_lat_lng(250)
# print("Hausdorff haversine test: {0}".format( hausdorff_distance(X, Y, distance="haversine") ))

