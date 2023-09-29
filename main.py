# -*- coding: utf-8 -*-

from torch.utils.data import DataLoader
from torchvision import transforms
from progressBar import printProgressBar
import medicalDataLoader
from utils import *
from network_unet_bigmb import *
from unet_postfusion_resnet_2modality_v2_RBU18_dwc import *

import time
from optimizer import Adam
import pandas as pd
from Metrics import SegmenationMetrics
import datetime


def runTraining(model_name, mode_list, batch_size, epoch, result_path, root_dir_train, root_dir_val):
    print('-' * 40)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 40)

    # batch_size = 2
    batch_size_val = 1
    batch_size_val_save = 1
 
    lr = 0.0001
    # lr = 0.01
    # epoch = 200
    num_classes = 2
    # initial_kernels = 32
    
        
    modelName = 'DualCT_Net'
    metric = SegmenationMetrics()
    
    
    img_names_ALL = []
    print('.'*40)
    print(" ....Model name: {} ........".format(modelName))
    
    print(' - Num. classes: {}'.format(num_classes))
    # print(' - Num. initial kernels: {}'.format(initial_kernels))
    print(' - Batch size: {}'.format(batch_size))
    print(' - Learning rate: {}'.format(lr))
    print(' - Num. epochs: {}'.format(epoch))

    print('.'*40)
    # model_dir = 'IVD_Net'


    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = medicalDataLoader.MedicalImageDataset('train',
                                                      root_dir_train,
                                                      transform=transform,
                                                      mask_transform=mask_transform)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              #num_workers=5,
                              shuffle=True)

    val_set = medicalDataLoader.MedicalImageDataset('val',
                                                    root_dir_val,
                                                    transform=transform,
                                                    mask_transform=mask_transform)
    
    val_loader = DataLoader(val_set,
                            batch_size=batch_size_val,
                            #num_workers=5,
                            shuffle=False)
    
    val_loader_save_images = DataLoader(val_set,
                                        batch_size=batch_size_val_save,
                                        #num_workers=5,
                                        shuffle=False)
    
    # Initialize
    print("~~~~~~~~~~~ Creating the model ~~~~~~~~~~")


    if model_name=='unet_v1':
        # in_ch_num=1
        in_ch_num=len(mode_list)
        model = Unet_v1(in_ch=in_ch_num)
        print(in_ch_num)
    elif model_name=='att_unet': 
        in_ch_num=len(mode_list)
        model = AttU_Net(img_ch=in_ch_num, output_ch=2) 
    elif model_name=='unet_pp': 
        in_ch_num=len(mode_list)
        model = NestedUNet(in_ch=in_ch_num, out_ch=2)
    elif model_name=='rbu18_2m': 
        in_ch_num=len(mode_list)
        model = unet_postfusion_resnet_2modality_v2_rbu18_dwc(in_ch=in_ch_num, out_ch=2)






    softMax = nn.Softmax()
    CE_loss = nn.CrossEntropyLoss()
    Dice_ = computeDiceOneHotBinary()

    if torch.cuda.is_available():
        model.cuda()
        softMax.cuda()
        CE_loss.cuda()
        Dice_.cuda()
        
    optimizer = Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), amsgrad=False)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode='max',
                                                           patience=4,
                                                           verbose=True,
                                                           factor=10 ** -0.5)
 
    BestDice, BestEpoch = 0, 0

    d1Train = []
    d1Val = []
    Losses = []
    result_summmary = pd.DataFrame()
    record_time = str(datetime.datetime.now().year)+'-'+str(datetime.datetime.now().month)+\
        '-'+str(datetime.datetime.now().day)+'-'+str(datetime.datetime.now().hour)+\
            '-'+str(datetime.datetime.now().minute)
        
    print("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    for i in range(epoch):
        model.train()
        lossTrain = []
        d1TrainTemp = []
        
        totalImages = len(train_loader)
        
        for j, data in enumerate(train_loader):
            print(j)
            image_ze,image_iod,image_ionw,image_mono70,image_mono40,image_mono100, labels, img_names = data
            
            # Be sure your data here is between [0,1]
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
          
            optimizer.zero_grad()
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

            Segmentation = to_var(labels)
            
            #target_dice = to_var(torch.ones(1))
            #model.zero_grad()            
            segmentation_prediction = model(MRI)
            # It needs the logits, not the softmax
            Segmentation_class = getTargetSegmentation(Segmentation)
            if len(Segmentation_class.shape)==2:
                Segmentation_class = torch.unsqueeze(Segmentation_class,0)
            CE_loss_ = CE_loss(segmentation_prediction, Segmentation_class)

            loss = CE_loss_ 
            #print(loss)
            loss.backward()
            optimizer.step()

            segmentation_prediction_ones = predToSegmentation(segmentation_prediction)
            
            Segmentation_planes = getOneHotSegmentation(Segmentation)
            DicesB, DicesF = Dice_(segmentation_prediction_ones, Segmentation_planes)
            DiceB = DicesToDice(DicesB)
            DiceF = DicesToDice(DicesF)
            
            lossTrain.append(loss.data.item())
 
            printProgressBar(j + 1, totalImages, prefix="[Training] Epoch: {} ".format(i), length=15, suffix=" Mean Dice: {:.4f},".format(DiceF.data.item()))

        printProgressBar(totalImages, totalImages,
                             done="[Training] Epoch: {}, LossG: {:.4f}".format(i,np.mean(lossTrain)))
        # Save statistics
        Losses.append(np.mean(lossTrain))
        
        print('optimizer.param_groups[0]:', optimizer.param_groups[0]['lr'])
        lr_current = optimizer.param_groups[0]['lr']
        
        d1, val_dice_1, val_iou_1, val_sensitivity_1, val_ppv_1, val_hausdorff_1 = inference(model, val_loader, batch_size, i, mode_list)
        print(d1)
        d1Val.append(d1)
        # currentDice = d1[0].numpy()
        currentDice = d1[0]
        print("[val] DSC: {:.4f} ".format(d1[0]))

        result_summmary = result_summmary.append([[np.mean(lossTrain),currentDice,lr_current,\
                                                   val_dice_1, val_iou_1, val_sensitivity_1, val_ppv_1, val_hausdorff_1]])
        record = '('+model_name+')_'+'('+mode_list_name+str(len(mode_list))+'m)_'+'(batchsize'+str(batch_size)+')_'+\
            '(epoch'+str(epoch)+')_'+'('+record_time+')'
        result_path_csv = os.path.join(result_path, record+'.csv')
        result_summmary.to_csv(result_path_csv, index=False)

        # Two ways of decay the learning rate:      
        if i % (BestEpoch + 10):
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
        scheduler.step(currentDice)

        # if currentDice > BestDice:
        #     BestDice = currentDice
        #     model_path = os.path.join(result_path, record+".pth")
        #     torch.save(model, model_path)

    

if __name__ == '__main__':


    result_path = r'.\result\'
    root_dir_train = r'.\train\'
    root_dir_val = r'.\val\'


    epoch_num_1 = 10

    test_dict = {'1': {'model_name':'rbu18_2m', 'mode_list':['ze','ionw'], 'batch_size':2, 'epoch':10, 
                        'result_path':result_path, 'root_dir_train':root_dir_train, 'root_dir_val':root_dir_val},
                  '2': {'model_name':'rbu18_2m', 'mode_list':['ze','mono40'], 'batch_size':2, 'epoch':10, 
                        'result_path':result_path, 'root_dir_train':root_dir_train, 'root_dir_val':root_dir_val},
                  '3': {'model_name':'rbu18_2m', 'mode_list':['ze','mono100'], 'batch_size':2, 'epoch':10, 
                        'result_path':result_path, 'root_dir_train':root_dir_train, 'root_dir_val':root_dir_val},
                  '4': {'model_name':'rbu18_2m', 'mode_list':['ionw','mono40'], 'batch_size':2, 'epoch':10, 
                        'result_path':result_path, 'root_dir_train':root_dir_train, 'root_dir_val':root_dir_val},
                  '5': {'model_name':'rbu18_2m', 'mode_list':['ionw','mono100'], 'batch_size':2, 'epoch':10, 
                        'result_path':result_path, 'root_dir_train':root_dir_train, 'root_dir_val':root_dir_val}}


    for i in test_dict.keys():
        print(i)
        # i='1'
        runTraining(model_name=test_dict[i]['model_name'],
                    mode_list=test_dict[i]['mode_list'],
                    batch_size=test_dict[i]['batch_size'],
                    epoch=test_dict[i]['epoch'],
                    result_path=test_dict[i]['result_path'],
                    root_dir_train=test_dict[i]['root_dir_train'],
                    root_dir_val=test_dict[i]['root_dir_val'])


