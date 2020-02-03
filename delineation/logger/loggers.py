import os
from torch.utils.tensorboard import SummaryWriter

class TensorboardLogger:

    def __init__(self, path):
        self.writer = SummaryWriter(path)

    def add_loss_to_tensorboard(self, type, loss, iter):
        self.writer.add_scalar(type+'/Loss', loss, iter)

    def add_scalars_to_tensorboard(self, type, epoch, iter, total_loss, values):
        pix1_err_m, pix3_err_m, pix5_err_m, epe_m = values

        self.writer.add_scalar(type+'/Loss', total_loss, iter)
        self.writer.add_scalar(type+'/EPE', epe_m, iter)
        self.writer.add_scalar(type+'/Pix1Err', pix1_err_m, iter)
        self.writer.add_scalar(type+'/Pix3Err', pix3_err_m, iter)
        self.writer.add_scalar(type+'/Pix5Err', pix5_err_m, iter)

    def add_images_to_tensorboard(self, img, name):

        self.writer.add_image('images/'+name, img, 0)

class FileLogger:
    "Log text in file."
    def __init__(self, path):
        self.path = os.path.join(path, 'log.txt')
        self.init_logs()

    def init_logs(self):

        text_file = open(self.path, "w")
        text_file.close()

    def log_string(self, string):
        """Stores log string in log file."""
        text_file = open(self.path, "a")
        text_file.write(str(string)+'\n')
        text_file.close()