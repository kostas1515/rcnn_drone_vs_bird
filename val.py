import torch.optim as optim
import torch
import torchvision
import pandas as pd
import time
import sys
import timeit
from dataset import *
import torchvision.ops.boxes as nms_box



df = pd.read_csv('../test_annotations.csv')

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# For training
PATH = './faster_rcnn.pth'
weights = torch.load(PATH)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)
model.to(device)

model.load_state_dict(weights)
model.eval()




df['pred_xmin']=0.0
df['pred_ymin']=0.0
df['pred_xmax']=0.0
df['pred_ymax']=0.0
df['iou']=0.0
drone_size='large+medium'
print('testing for '+ drone_size+'\n')
transformed_dataset=DroneDatasetCSV(csv_file='../test_annotations.csv',
                                           root_dir='../test_images/',
                                           drone_size=drone_size,
                                           transform=transforms.Compose([
                                               ResizeToTensor(800)
                                           ]))


dataset_len=(len(transformed_dataset))
print('Length of dataset is '+ str(dataset_len)+'\n')
batch_size=1

dataloader = DataLoader(transformed_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=0)

true_pos=0
false_pos=0
counter=0
iou_threshold=0.1
confidence=0.9

for images, target in dataloader:
    inp=images.cuda()
    raw_pred = model(inp)
    target=target['boxes']

    boxes=raw_pred[0]['boxes'].to(device='cuda')
    pred_mask=raw_pred[0]['scores']>confidence
    boxes=boxes[pred_mask]
    scores=raw_pred[0]['scores'][pred_mask]
#     pred_mask=true_pred[0,:,4].max() == true_pred[0,:,4]
#       df.pred_xmin[counter]=round(pred_final[:,:,0].item())
#       df.pred_ymin[counter]=round(pred_final[:,:,1].item())
#       df.pred_xmax[counter]=round(pred_final[:,:,2].item())
#       df.pred_ymax[counter]=round(pred_final[:,:,3].item())
    
    indices=nms_box.nms(boxes,scores,iou_threshold)
    target=target.to('cuda')

    iou=nms_box.box_iou(target,boxes[indices])
    true_pos=true_pos+(iou>=0.5).sum().item()
    false_pos=false_pos+(iou<0.5).sum().item()
    #         df.iou[counter]=iou.item()
    counter=counter+1
print('precision')
precision=true_pos/(true_pos+false_pos)
print(precision)

print('recall')
recall=true_pos/(counter)
print(recall)
f1=2*(precision*recall)/(precision+recall)
print(f1)


# df.to_csv('test+pred_annotations.csv')
