from torch.utils.tensorboard import SummaryWriter
from collections import defaultdict
import numpy as np
import pandas as pd
import os
import torch
import yaml


class Logger(object):
    def __init__(self, data, logscalars=True, save_embed = True):


        exp = 0
        if data["LAI"]:
            experience =  './logs/' + data['dataset'] +'/'+ data['model_arch'] + '_LAI/' + str(exp)
        else:
            experience =  './logs/' + data['dataset'] +'/'+ data['model_arch'] + '/' + str(exp)
        while os.path.isdir(experience) == True:
            exp+=1
            if data["LAI"]:
                experience = './logs/' + data['dataset'] +'/'+ data['model_arch'] + '_LAI/' + str(exp)
            else:
                experience = './logs/' + data['dataset'] +'/'+ data['model_arch'] + '/' + str(exp)
        # Explicit, idempotent directory creation. Previously this relied
        # entirely on SummaryWriter's own implicit directory creation; if
        # ./logs/.../<exp>/ is later removed by something outside this
        # process (disk cleanup, quota, NFS issue, etc.), every subsequent
        # write — TensorBoard's background thread AND save_model/save_log
        # below — fails with FileNotFoundError. This doesn't prevent an
        # external process from deleting the directory again, but it means
        # save_model/save_log (which run in the main thread, unlike
        # TensorBoard's writer) will recover instead of crashing.
        os.makedirs(experience, exist_ok=True)
        self.logscalars = defaultdict(list)
        self.writer = SummaryWriter(experience)
        self.save_embed = save_embed
        self.savepath = experience
        self.log =logscalars
        with open(self.savepath + '/config.yaml', "w", encoding = "utf-8") as yaml_file:       
            dump = yaml.dump(data, default_flow_style = False, allow_unicode = True, encoding = None)
            yaml_file.write(dump) 

    def write_scalars(self, scalars, step, write_epoch=False, step_key='epoch'):
        if write_epoch:
            self.logscalars[step_key].append(step)
        for k, v in scalars.items():
            if self.log:
                self.logscalars[k].append(v)
            self.writer.add_scalar(k, v, step)

    def save_log(self):
        # Defensive re-creation — see note in __init__. Cheap and a no-op
        # when self.savepath already exists.
        os.makedirs(self.savepath, exist_ok=True)
        acc_dict = {k: v for k, v in self.logscalars.items() if k.startswith("Acc") or k == "epoch"}
        rest = {k: v for k, v in self.logscalars.items() if (not k.startswith("Acc") and k != "epoch") or k == "batch_step"}
        df = pd.DataFrame(acc_dict)
        df.to_csv(self.savepath + "/log_acc.txt")
        df = pd.DataFrame(rest)
        df.to_csv(self.savepath + "/log_train.txt")
        self.writer.flush()
        self.writer.close()

    # def save_model(self, model):
    #     maps = self.logscalars['Accuraccy/mAP']
    #     cmcs = self.logscalars['Accuraccy/CMC1']
    #     if len(maps) >= 1 and maps[-1] >= max(maps):
    #         torch.save(model.state_dict(), self.savepath + '/' + 'best_mAP' + '.pt')
    #     if len(cmcs) >= 1 and cmcs[-1] >= max(cmcs):
    #         torch.save(model.state_dict(), self.savepath + '/' + 'best_CMC' + '.pt')
    #     torch.save(model.state_dict(), self.savepath + '/' + 'last' + '.pt')
    
    def save_model(self, model, epoch=None):
        # Defensive re-creation — see note in __init__. Cheap and a no-op
        # when self.savepath already exists.
        os.makedirs(self.savepath, exist_ok=True)
        maps = self.logscalars.get('Accuraccy/mAP_vehiclex', self.logscalars['Accuraccy/mAP'])  # falls back to VeRi if not VehicleX run
        cmcs = self.logscalars.get('Accuraccy/CMC1_vehiclex', self.logscalars['Accuraccy/CMC1'])
        if len(maps) >= 1 and maps[-1] >= max(maps):
            torch.save(model.state_dict(), self.savepath + '/' + 'best_mAP' + '.pt')
        if len(cmcs) >= 1 and cmcs[-1] >= max(cmcs):
            torch.save(model.state_dict(), self.savepath + '/' + 'best_CMC' + '.pt')
        torch.save(model.state_dict(), self.savepath + '/' + 'last' + '.pt')
        if epoch is not None:
            torch.save(model.state_dict(), self.savepath + f'/epoch_{epoch}.pt')

    def write_embeddings(self, features, labels, images, epoch, tag='default'):
        #if self.logscalars['Accuraccy/mAP'][-1] > np.max(self.logscalars['Accuraccy/mAP']):
        self.writer.add_embedding(features, metadata=labels, label_img=images, global_step=epoch, tag=tag)
        
    
    

if __name__ == "__main__":
    import torch
    import torchvision
    with open("./config/config.yaml", "r") as stream:
        print('debug')
        data = yaml.safe_load(stream)
    loggerwriter = Logger(data)

    dicttest = {"Loss/train_total": np.mean([4,2,1]), "Loss/train_crossentropy": np.mean([12,32,3]), "Loss/train_triplet": np.mean([223,11]), "Loss/ce_loss_weight": np.mean([[2,8,6,4,4]]), "Loss/triplet_loss_weight": np.mean([1,1])} 
    diccacc = {"Accuraccy/CMC1": int(100), "Accuraccy/mAP": 99}
    loggerwriter.write_scalars(diccacc, 1)
    loggerwriter.write_scalars(dicttest, 1)
    loggerwriter.write_scalars(dicttest, 2)
    loggerwriter.write_scalars(dicttest, 3)
    loggerwriter.write_scalars(dicttest, 4)
    loggerwriter.write_scalars(diccacc, 4)
    loggerwriter.write_scalars(dicttest, 5)
    loggerwriter.write_scalars(dicttest, 6)
    loggerwriter.write_scalars({"Accuraccy/CMC1": int(101), "Accuraccy/mAP": 109}, 6)
    model = torchvision.models.resnet18(pretrained=False)
    loggerwriter.save_model(model, 'test.pt')

    loggerwriter.save_log()