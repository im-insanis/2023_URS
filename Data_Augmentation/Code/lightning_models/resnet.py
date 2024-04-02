from typing import Any,List
import lightning as L
import torch,torchvision
from pytorch_models.cifar import resnet
from utils.eval import accuracy


class ResNet(L.LightningModule):
    def __init__(self,args):
        super().__init__()
        self.save_hyperparameters()
        self.args= args
        self.model = self._create_model()
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.example_input_array = torch.rand(1, 3, 32, 32)
        self.log_image =True
    def forward(self, x):
        out = self.model(x)
        return out
    
    def training_step(self, batch, batch_idx) :
        x,y = batch

        logits = self.model(x)
        loss = self.criterion(logits,y)
        acc = accuracy(logits, y)[0]
        self.sample = x[:10]
        self.log("train_loss", loss,prog_bar=True)
        self.log("train_acc",acc, prog_bar=True )

        return loss
    def on_train_epoch_end(self):
        if self.log_image:
            self.logger.experiment.add_image("sample",torchvision.utils.make_grid(self.sample,nrow=self.sample.shape[0],normalize=True),self.current_epoch)

    def _create_model(self):
        if self.args['dataset'] == 'cifar10':
            num_classes = 10
        if self.args['dataset'] == 'cifar100':
            num_classes = 100
        return resnet(depth=self.args['depth'], num_classes=num_classes)
    def validation_step(self, batch,batch_idx):
        self._shared_eval(batch,batch_idx,"val")
    def test_step(self, batch,batch_idx):
        self._shared_eval(batch,batch_idx,"test")
    def _shared_eval(self, batch, batch_idx, prefix):
        x,  y = batch
        logits = self(x)
        loss = self.criterion(logits,y)
        top1,top5 = accuracy(logits, y,topk=(1,5))
        self.log(f"{prefix}_loss", loss,prog_bar=True,on_epoch=True)
        self.log(f"{prefix}_acc", top1, prog_bar=True,on_epoch=True)
        self.log(f"{prefix}_error_rate", (100-top1), prog_bar=True,on_epoch=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.args['lr'],
            momentum=self.args['momentum'],
            weight_decay=self.args['weight_decay'],
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.args['schedule'], gamma=self.args['gamma'])
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
