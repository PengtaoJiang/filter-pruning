import os
import torch
import torch.distributed as dist
from os.path import join
try:
    import nvidia.dali.plugin.pytorch as plugin_pytorch
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

# DALI data loader
class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, record_dir, crop, num_gpus, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.MXNetReader(path = [record_dir+"/train.rec"], 
        index_path=[join(record_dir, "train.idx")], random_shuffle = True, shard_id = device_id, num_shards = num_gpus)
        
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)

        self.random_resize_crop = ops.RandomResizedCrop(device="gpu", size =(crop, crop), interp_type=types.INTERP_CUBIC, random_area=[0.2, 1])
        self.crop_mirror_norm = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])
        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format("gpu"))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.random_resize_crop(images)
        output = self.crop_mirror_norm(images.gpu(), mirror=rng)
        return [output, self.labels]

class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, record_dir, crop, size, num_gpus, dali_cpu=False):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.MXNetReader(path = [join(record_dir,"val.rec")], index_path=[join(record_dir,"val.idx")],
                                     random_shuffle = False, shard_id = device_id, num_shards = num_gpus)

        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_CUBIC)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255,0.456 * 255,0.406 * 255],
                                            std=[0.229 * 255,0.224 * 255,0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]
    
def dali_ilsvrc_loader(mx_record_dir, num_gpus, batch_size, crop_size=224, resize=256,  num_threads_per_gpu=1):

    pipes = [HybridTrainPipe(batch_size=int(batch_size/num_gpus), num_threads=num_threads_per_gpu, device_id=device_id,\
             record_dir=mx_record_dir, crop=crop_size, num_gpus=num_gpus, dali_cpu=False) for device_id in range(num_gpus)]
    pipes[0].build()
    train_loader = plugin_pytorch.DALIGenericIterator(pipes, ["data", "label"], size=int(pipes[0].epoch_size("Reader")))

    pipes = [HybridValPipe(batch_size=int(100/num_gpus), num_threads=num_threads_per_gpu, device_id=device_id,\
            record_dir=mx_record_dir, crop=crop_size, size=resize, num_gpus=num_gpus, dali_cpu=False) for device_id in range(num_gpus)]
    pipes[0].build()
    val_loader = plugin_pytorch.DALIGenericIterator(pipes, ["data", "label"], size=int(pipes[0].epoch_size("Reader")))

    return train_loader, val_loader

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    from vltools.image import norm255
    import time

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    print(torch.distributed.get_world_size())

    train_loader, val_loader = dali_ilsvrc_loader("/media/ssd0/ilsvrc12/rec", num_gpus=2, batch_size=256, num_threads_per_gpu=1)

    end = time.time()
    for data in train_loader:
        print("Data loading time: %f"%(time.time()-end))
        # fig, axes = plt.subplots(1, 2)
        # label0 = data[0]["label"]
        # data0 = data[0]["data"]
        # end = time.time()
        print(data[0]["data"])

        # label1 = data[1]["label"]
        # data1 = data[1]["data"]

        # axes[0].imshow(norm255(data0[0].cpu().squeeze().numpy().transpose((1,2,0))))
        # axes[1].imshow(norm255(data0[0].cpu().squeeze().numpy().transpose((1,2,0))))
        # plt.show()

        #print(data0.shape, data0.device)
