import os
import argparse
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import utils
from utils import PointLoss
from utils import distance_squre
import data_utils as d_utils
import timber_axis_loader
from tqdm import tqdm
import shutil

from model_PFNet import _netlocalD,_netG



parser = argparse.ArgumentParser()
parser.add_argument('--dataroot',  default='dataset/train', help='path to dataset')
parser.add_argument('--workers', type=int,default=0, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
parser.add_argument('--psample', type=int, default=2048, help='the point number of a sample')
parser.add_argument('--ptruth', type=int, default=512, help='the point number of a sample')
parser.add_argument('--crop_point_num',type=int,default=512,help='0 means do not use else use with this weight')
parser.add_argument('--niter', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', type = bool, default = False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--D_choose',type=int, default=1, help='0 not use D-net,1 use D-net')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--drop',type=float,default=0.2)
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
parser.add_argument('--point_scales_list',type=list,default=[2048, 1024, 512],help='number of points in each scales')
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
parser.add_argument('--wtl2',type=float,default=0.95,help='0 means do not use else use with this weight')
opt = parser.parse_args()
print(opt)

#   resume 设置

# opt.netG = r"H:\irregular construct\PF-Net-Point-Fractal-Network-master\Trained_Model\netG\point_netG0200.pth"
# opt.netD = r"H:\irregular construct\PF-Net-Point-Fractal-Network-master\Trained_Model\netD\point_netD0200.pth"
resume_epoch=0

#

blue = lambda x: '\033[94m' + x + '\033[0m'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USE_CUDA = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
point_netG = _netG(opt.num_scales,opt.each_scales_size,opt.point_scales_list,opt.crop_point_num)
point_netD = _netlocalD(opt.crop_point_num)
cudnn.benchmark = True

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Conv1d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0) 

if USE_CUDA:       
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    point_netG = torch.nn.DataParallel(point_netG)
    point_netD = torch.nn.DataParallel(point_netD)
    point_netG.to(device) 
    point_netG.apply(weights_init_normal)
    point_netD.to(device)
    point_netD.apply(weights_init_normal)
if opt.netG != '' :
    point_netG.load_state_dict(torch.load(opt.netG,map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netG)['epoch']
if opt.netD != '' :
    point_netD.load_state_dict(torch.load(opt.netD,map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netD)['epoch']

        
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)


transforms = transforms.Compose(
    [
        d_utils.PointcloudToTensor(),
    ]
)
dset = timber_axis_loader.timberDataset( root='./dataset/timber_axis/train', nsample=opt.psample, ntruth=opt.ptruth)
assert dset
# print(len(dset.datapath))
dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize,
                                         shuffle=True,num_workers = int(opt.workers))



test_dset = timber_axis_loader.timberDataset( root='./dataset/timber_axis/test', nsample=opt.psample, ntruth=opt.ptruth)
test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
                                         shuffle=True,num_workers = int(opt.workers))

#dset = ModelNet40Loader.ModelNet40Cls(opt.pnum, train=True, transforms=transforms, download = False)
#assert dset
#dataloader = torch.utils.data.DataLoader(dset, batch_size=opt.batchSize,
#                                         shuffle=True,num_workers = int(opt.workers))
#
#
#test_dset = ModelNet40Loader.ModelNet40Cls(opt.pnum, train=False, transforms=transforms, download = False)
#test_dataloader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
#                                         shuffle=True,num_workers = int(opt.workers))

#pointcls_net.apply(weights_init)
# print(point_netG)
# print(point_netD)

criterion = torch.nn.BCEWithLogitsLoss().to(device)
criterion_PointLoss = PointLoss().to(device)

# setup optimizer
optimizerD = torch.optim.Adam(point_netD.parameters(), lr=0.0001,betas=(0.9, 0.999),eps=1e-05,weight_decay=opt.weight_decay)
optimizerG = torch.optim.Adam(point_netG.parameters(), lr=0.0001,betas=(0.9, 0.999),eps=1e-05 ,weight_decay=opt.weight_decay)
schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=40, gamma=0.2)
schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=40, gamma=0.2)

real_label = 1
fake_label = 0

crop_point_num = int(opt.crop_point_num)
sample_point1 = torch.FloatTensor(opt.batchSize, opt.psample, 3)
label = torch.FloatTensor(opt.batchSize)


num_batch = len(dset) / opt.batchSize
###########################
#  G-NET and T-NET
##########################  
if opt.D_choose == 1:
    for epoch in range(resume_epoch,opt.niter):
        if epoch<30:
            alpha1 = 0.01
            alpha2 = 0.02
        elif epoch<80:
            alpha1 = 0.05
            alpha2 = 0.1
        else:
            alpha1 = 0.1
            alpha2 = 0.2
        
        loss_D = []
        loss_G = []
        loss_CD = []
        loop = tqdm(enumerate(dataloader, 0), total=len(dataloader))

        for i, data in loop:
            sample_point, truth_point = data       
            # print(sample_point.shape, truth_point.shape) #torch.Size([8, 4096, 3]) torch.Size([8, 1024, 3])

            batch_size = sample_point.size()[0]

            label.resize_([batch_size,1]).fill_(real_label)
            sample_point = sample_point.to(device)
            truth_point = truth_point.to(device)
            label = label.to(device)

            p_origin = [0,0,0]

            truth_point = Variable(truth_point, requires_grad=True)

            #truth_point准备
            truth_point_key1_idx = utils.farthest_point_sample(truth_point,128,RAN = False)
            truth_point_key1 = utils.index_points(truth_point, truth_point_key1_idx)
            truth_point_key1 =Variable(truth_point_key1,requires_grad=True)
            # print(truth_point_key1.shape) #torch.Size([8, 128, 3])

            truth_point_key2_idx = utils.farthest_point_sample(truth_point,256,RAN = False)
            truth_point_key2 = utils.index_points(truth_point, truth_point_key2_idx)
            truth_point_key2 =Variable(truth_point_key2,requires_grad=True)
            # print(truth_point_key2.shape) #torch.Size([8, 256, 3])

            #多尺度采样
            sample_point1 = sample_point
            sample_point2_idx = utils.farthest_point_sample(sample_point1,opt.point_scales_list[1],RAN = True)
            sample_point2     = utils.index_points(sample_point1,sample_point2_idx)
            sample_point3_idx = utils.farthest_point_sample(sample_point1,opt.point_scales_list[2],RAN = False)
            sample_point3     = utils.index_points(sample_point1,sample_point3_idx)
            sample_point1 = Variable(sample_point1,requires_grad=True)
            sample_point2 = Variable(sample_point2,requires_grad=True)
            sample_point3 = Variable(sample_point3,requires_grad=True)
            sample_point2 = sample_point2.to(device)
            sample_point3 = sample_point3.to(device)      
            input_cropped  = [sample_point1,sample_point2,sample_point3]
            # print(sample_point1.shape, sample_point2.shape, sample_point3.shape) #torch.Size([8, 4096, 3]) torch.Size([8, 2048, 3]) torch.Size([8, 1024, 3])
            point_netG = point_netG.train()
            point_netD = point_netD.train()
            ############################
            # (2) Update D network
            ###########################        
            point_netD.zero_grad()
            truth_point = torch.unsqueeze(truth_point,1)   
            output = point_netD(truth_point)
            errD_real = criterion(output,label)
            errD_real.backward()
            fake_center1,fake_center2,fake  =point_netG(input_cropped)
            fake = torch.unsqueeze(fake,1)
            label.data.fill_(fake_label)
            output = point_netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()
            ############################
            # (3) Update G network: maximize log(D(G(z)))
            ###########################
            point_netG.zero_grad()
            label.data.fill_(real_label)
            output = point_netD(fake)
            errG_D = criterion(output, label)
            errG_l2 = 0
            CD_LOSS = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(truth_point,1))
    
            errG_l2 = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(truth_point,1))\
            +alpha1*criterion_PointLoss(fake_center1,truth_point_key1)\
            +alpha2*criterion_PointLoss(fake_center2,truth_point_key2)
            
            errG = (1-opt.wtl2) * errG_D + opt.wtl2 * errG_l2
            errG.backward()
            optimizerG.step()

            # print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f / %.4f/ %.4f'
            #     % (epoch, opt.niter, i, len(dataloader), 
            #         errD.data, errG_D.data,errG_l2,errG,CD_LOSS))
            f=open('loss_PFNet.txt','a')
            f.write('\n'+'[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f / %.4f / %.4f /%.4f'
                % (epoch, opt.niter, i, len(dataloader), 
                    errD.data, errG_D.data,errG_l2,errG,CD_LOSS))
            
            if i % 10 ==0:
                # print('After, ',i,'-th batch')
                f.write('\n'+'After, '+str(i)+'-th batch')
                for i, data in enumerate(test_dataloader, 0):
                    sample_point1, truth_point = data
                    
                    batch_size = sample_point1.size()[0]

                    sample_point1 = sample_point1.to(device)
                    truth_point = truth_point.to(device)
                    p_origin = [0,0,0]

                    sample_point2_idx = utils.farthest_point_sample(sample_point1,opt.point_scales_list[1],RAN = True)
                    sample_point2     = utils.index_points(sample_point1,sample_point2_idx)
                    sample_point3_idx = utils.farthest_point_sample(sample_point1,opt.point_scales_list[2],RAN = False)
                    sample_point3     = utils.index_points(sample_point1,sample_point3_idx)
                    sample_point1 = Variable(sample_point1,requires_grad=False)
                    sample_point2 = Variable(sample_point2,requires_grad=False)
                    sample_point3 = Variable(sample_point3,requires_grad=False)
                    sample_point2 = sample_point2.to(device)
                    sample_point3 = sample_point3.to(device)      
                    input_cropped  = [sample_point1,sample_point2,sample_point3]
                    point_netG.eval()
                    fake_center1,fake_center2,fake  =point_netG(input_cropped)
                    CD_loss = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(truth_point,1))
                    # print('test result:',CD_loss)
                    f.write('\n'+'test result:  %.4f'%(CD_loss))
                    break
            f.close()
            
            loop.set_description(f'Epoch [{epoch}/{opt.niter}]')
            loop.set_postfix(loss_D = errD.item(), loss_G = errG.item(), loss_CD = CD_loss.item())

        schedulerD.step()
        schedulerG.step()
        if (epoch + resume_epoch) % 1 == 0:   
            torch.save({'epoch':epoch+1,
                        'state_dict':point_netG.state_dict()},
                        'Trained_Model/netG/point_netG'+str(epoch).zfill(len(str(opt.niter)) + 1)+'.pth' )
            torch.save({'epoch':epoch+1,
                        'state_dict':point_netD.state_dict()},
                        'Trained_Model/netD/point_netD'+str(epoch).zfill(len(str(opt.niter)) + 1)+'.pth' )  
            if (epoch + resume_epoch) % 1 == 0:  
                shutil.move('Trained_Model/netD/point_netD'+str(epoch).zfill(len(str(opt.niter)) + 1)+'.pth', 
                            'Trained_Model/netD/save/point_netD'+str(epoch).zfill(len(str(opt.niter)) + 1)+'.pth')
                shutil.move('Trained_Model/netG/point_netG'+str(epoch).zfill(len(str(opt.niter)) + 1)+'.pth', 
                            'Trained_Model/netG/save/point_netG'+str(epoch).zfill(len(str(opt.niter)) + 1)+'.pth')
            while len(os.listdir(os.path.join(BASE_DIR, 'Trained_Model\\netG'))) > 5:
                os.remove(
                    os.path.join(BASE_DIR, 'Trained_Model/netG', 
                                os.listdir('./Trained_Model/netG')[0]))
                os.remove(
                    os.path.join(BASE_DIR, 'Trained_Model/netD', 
                                os.listdir('./Trained_Model/netD')[0]))

#
#############################
## ONLY G-NET
############################ 
else:
    for epoch in range(resume_epoch,opt.niter):
        if epoch<30:
            alpha1 = 0.01
            alpha2 = 0.02
        elif epoch<80:
            alpha1 = 0.05
            alpha2 = 0.1
        else:
            alpha1 = 0.1
            alpha2 = 0.2
        
        for i, data in enumerate(dataloader, 0):
            
            real_point, target = data
            
            batch_size = real_point.size()[0]
            truth_point = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)       
            sample_point1 = torch.FloatTensor(batch_size, opt.pnum, 3)
            sample_point1 = sample_point1.data.copy_(real_point)
            real_point = torch.unsqueeze(real_point, 1)
            sample_point1 = torch.unsqueeze(sample_point1,1)
            p_origin = [0,0,0]
            if opt.cropmethod == 'random_center':
                choice = [torch.Tensor([1,0,0]),torch.Tensor([0,0,1]),torch.Tensor([1,0,1]),torch.Tensor([-1,0,0]),torch.Tensor([-1,1,0])]
                for m in range(batch_size):
                    index = random.sample(choice,1)
                    distance_list = []
                    p_center = index[0]
                    for n in range(opt.pnum):
                        distance_list.append(distance_squre(real_point[m,0,n],p_center))
                    distance_order = sorted(enumerate(distance_list), key  = lambda x:x[1])
                    
                    for sp in range(opt.crop_point_num):
                        sample_point1.data[m,0,distance_order[sp][0]] = torch.FloatTensor([0,0,0])
                        truth_point.data[m,0,sp] = real_point[m,0,distance_order[sp][0]]
            real_point = real_point.to(device)
            truth_point = truth_point.to(device)
            sample_point1 = sample_point1.to(device)
            ############################
            # (1) data prepare
            ###########################      
            truth_point = Variable(truth_point,requires_grad=True)
            truth_point = torch.squeeze(truth_point,1)
            truth_point_key1_idx = utils.farthest_point_sample(truth_point,64,RAN = False)
            truth_point_key1 = utils.index_points(truth_point,truth_point_key1_idx)
            truth_point_key1 =Variable(truth_point_key1,requires_grad=True)

            truth_point_key2_idx = utils.farthest_point_sample(truth_point,128,RAN = True)
            truth_point_key2 = utils.index_points(truth_point,truth_point_key2_idx)
            truth_point_key2 =Variable(truth_point_key2,requires_grad=True)
            
            sample_point1 = torch.squeeze(sample_point1,1)
            sample_point2_idx = utils.farthest_point_sample(sample_point1,opt.point_scales_list[1],RAN = True)
            sample_point2     = utils.index_points(sample_point1,sample_point2_idx)
            sample_point3_idx = utils.farthest_point_sample(sample_point1,opt.point_scales_list[2],RAN = False)
            sample_point3     = utils.index_points(sample_point1,sample_point3_idx)
            sample_point1 = Variable(sample_point1,requires_grad=True)
            sample_point2 = Variable(sample_point2,requires_grad=True)
            sample_point3 = Variable(sample_point3,requires_grad=True)
            sample_point2 = sample_point2.to(device)
            sample_point3 = sample_point3.to(device)      
            input_cropped  = [sample_point1,sample_point2,sample_point3]
            point_netG = point_netG.train()
            point_netG.zero_grad()
            fake_center1,fake_center2,fake  =point_netG(input_cropped)
            fake = torch.unsqueeze(fake,1)
            ############################
            # (3) Update G network: maximize log(D(G(z)))
            ###########################
            
            CD_LOSS = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(truth_point,1))
            
            errG_l2 = criterion_PointLoss(torch.squeeze(fake,1),torch.squeeze(truth_point,1))\
            +alpha1*criterion_PointLoss(fake_center1,truth_point_key1)\
            +alpha2*criterion_PointLoss(fake_center2,truth_point_key2)

            errG_l2.backward()
            optimizerG.step()
            print('[%d/%d][%d/%d] Loss_G: %.4f / %.4f '
                % (epoch, opt.niter, i, len(dataloader), 
                    errG_l2,CD_LOSS))
            f=open('loss_PFNet.txt','a')
            f.write('\n'+'[%d/%d][%d/%d] Loss_G: %.4f / %.4f '
                % (epoch, opt.niter, i, len(dataloader), 
                    errG_l2,CD_LOSS))
            f.close()

        schedulerD.step()
        schedulerG.step()
        
        if epoch% 10 == 0:   
            torch.save({'epoch':epoch+1,
                        'state_dict':point_netG.state_dict()},
                        'Checkpoint/point_netG'+str(epoch)+'.pth' )
 

    
        
