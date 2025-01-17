# Copyright (c) 2020 Uber Technologies, Inc.
# See the License for the specific language governing permissions and
# limitations under the License.

# import debugpy
# debugpy.listen(address = ('0.0.0.0', 5678))
# debugpy.wait_for_client() 
# breakpoint()


import os
from tkinter.messagebox import NO

os.umask(0)
import argparse
import numpy as np
import random
import sys
import time
import shutil
from importlib import import_module
from numbers import Number

from tqdm import tqdm
import torch
from torch.utils.data import Sampler, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import subprocess
from utils import Logger, load_pretrain,viz_sequence

from torch.distributed.elastic.multiprocessing.errors import record
from lanegcn import Optimizer
from data import ArgoTestDataset

#vis
from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader
root_dir = "/mnt/lustre/tangxiaqiang/Code/LaneGCN/dataset/val/data"
afl = ArgoverseForecastingLoader(root_dir)
import matplotlib.pyplot as plt

MY_TIME=time.strftime('%mmounth%dday%Hhour%Mminit%Ss')
DEBUG=True

root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)


parser = argparse.ArgumentParser(description="Fuse Detection in Pytorch")
parser.add_argument(
    "-m", "--model", default="lanegcn", type=str, metavar="MODEL", help="model name"
)
parser.add_argument("--eval", action="store_true")
parser.add_argument(
    "--resume", default="", type=str, metavar="RESUME", help="checkpoint path"
)
parser.add_argument(
    "--weight", default="", type=str, metavar="WEIGHT", help="checkpoint path"
)

device = torch.device("cuda")




def cleanup():
    dist.destroy_process_group()

@record
def main(rank, world_size):


    # initialize the process group
    dist.init_process_group(backend="nccl",world_size=world_size,rank=rank)
    print("inited rank",rank)
    torch.distributed.barrier()

    seed=7
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Import all settings for experiment.
    args = parser.parse_args()
    model = import_module(args.model)
    config, Dataset, collate_fn, net, loss, post_process, _ = model.get_model()

    #ddp
    net.to("cuda")
    net = DDP(net, device_ids=[rank])
    opt = Optimizer(net.parameters(), config)


    if args.resume or args.weight:
        ckpt_path = args.resume or args.weight
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.join(config["save_dir"], ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
        load_pretrain(net, ckpt["state_dict"])
        if args.resume:
            config["epoch"] = ckpt["epoch"]
            opt.load_state_dict(ckpt["opt_state"])

    if args.eval:
        # Data loader for evaluation
        dataset = Dataset(config["val_split"], config, train=False)
        
        val_loader = DataLoader(
            dataset,
            batch_size=config["val_batch_size"],
            num_workers=config["val_workers"],
            #sampler=val_sampler,
            collate_fn=collate_fn,
            pin_memory=True,
        )

       
        val(config, val_loader, net, loss, post_process, 999)
        return
    print(config)
    print(MY_TIME)
    # Create log and copy all code
    save_dir = config["save_dir"]
    # log = os.path.join(save_dir, "log")

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # sys.stdout = Logger(log)

    src_dirs = [root_path]
    dst_dirs = [os.path.join(save_dir, "files")]
    for src_dir, dst_dir in zip(src_dirs, dst_dirs):
        files = [f for f in os.listdir(src_dir) if f.endswith(".py")]
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        for f in files:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))

    # Data loader for training
    dataset = Dataset(config["train_split"], config, train=True)
    
    train_sampler =DistributedSampler(dataset) #DDP
    
    train_loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        num_workers=config["workers"],
        sampler=train_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    # Data loader for evaluation
    dataset  = Dataset(config["val_split"], config, train=False)
    val_sampler=DistributedSampler(dataset) #DDP
    val_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        sampler=val_sampler,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    import pdb;pdb.set_trace()
    
    dataset = ArgoTestDataset(config["test_split"], config, train=False)
    test_loader = DataLoader(
        dataset,
        batch_size=config["val_batch_size"],
        num_workers=config["val_workers"],
        collate_fn=collate_fn,
        shuffle=False,
        pin_memory=True,
    )


    epoch = config["epoch"]
    remaining_epochs = int(np.ceil(config["num_epochs"] - epoch))
    if rank == 0:
        print("----------------config---------------------")
        #print(f"num_batches:{num_batches}  epoch_per_batch:{epoch_per_batch}  save_iters:{save_iters} display_iters:{display_iters} val_iters:{val_iters} ")
        print(config)
    for i in range(remaining_epochs):
        train(epoch + i, config, train_loader, net, loss, post_process, opt, val_loader)
    # if rank == 0:    
    #     test(test_loader,net,config)

# def worker_init_fn(pid):
#     np_seed = hvd.rank() * 1024 + int(pid)
#     np.random.seed(np_seed)
#     random_seed = np.random.randint(2 ** 32 - 1)
#     random.seed(random_seed)


def train(epoch, config, train_loader, net, loss, post_process, opt, val_loader=None):
    #train_loader.sampler.set_epoch(int(epoch))

    net.train()
    
    num_batches = len(train_loader) #1608*128个样本
    epoch_per_batch = 1.0 / num_batches
    save_iters = int(np.ceil(config["save_freq"] * num_batches))
    display_iters = int(config["display_iters"]  )#/(torch.cuda.device_count() * config["batch_size"]) )
    val_iters = int(config["val_iters"] )# /(torch.cuda.device_count() * config["batch_size"]))

    start_time = time.time()
    metrics = dict()
    
    rank=dist.get_rank()

    for i, data in tqdm(enumerate(train_loader),disable=rank):
        epoch += epoch_per_batch
        data = dict(data)



        output = net(data)
        loss_out = loss(output, data)  #output: list len4, output[0]:23,6,30,2 data['gt_preds'][0]:23,30,2
        post_out = post_process(output, data)
        post_process.append(metrics, loss_out, post_out)

        opt.zero_grad()
        loss_out["loss"].backward()
        lr = opt.step(epoch)

        num_iters = int(np.round(epoch * num_batches))
        if rank == 0 and (
            num_iters % save_iters == 0 or epoch >= config["num_epochs"]
        ):
            print(f"saving at num_iters:{num_iters} epoch: {epoch}")
            save_ckpt(net, opt, config["save_dir"]+MY_TIME, epoch)

        if num_iters % save_iters == 0 and rank == 0 :#display_iters == 0 and rank==0:
            print(f"display---num_iters:{num_iters} iter:{i}")
            dt = time.time() - start_time
            # metrics = sync(metrics)
            # if hvd.rank() == 0:
            post_process.display(metrics, dt, epoch, lr)
            start_time = time.time()
            metrics = dict()

        if num_iters % save_iters == 0 :#val_iters == 0:
            val(config, val_loader, net, loss, post_process, epoch)

        if epoch >= config["num_epochs"]:
            val(config, val_loader, net, loss, post_process, epoch)
            return


def val(config, data_loader, net, loss, post_process, epoch):
    net.eval()
    print("---------------VAL---------------------")
    start_time = time.time()
    metrics = dict()
    for i, data in enumerate(data_loader):
        data = dict(data)
        with torch.no_grad():
            output = net(data)
            if dist.get_rank() == 0:
                import ipdb;ipdb.set_trace()
            if True and (i+1)%1==0:

                trajs_list = [x[0:1].detach().cpu().numpy() for x in output["reg"]]
                probs_list=[x[0:1].detach().cpu().numpy() for x in output["cls"]]
                
                for trajs,probs,key in zip(trajs_list,probs_list,data["idx"]):
                    seq_path = f"{root_dir}/"+str(key)+".csv"
                    fig,ax = plt.subplots(figsize=(16, 14),dpi=100)
                    ax=viz_sequence(afl.get(seq_path).seq_df,ax=ax)
                    
                    for traj,prob in zip(trajs.squeeze(),probs.squeeze()):
                        ax.plot(traj[:,0],traj[:,1])
                        ax.text(traj[-1,0],traj[-1,1],str(round(prob,2)))
                print("saving at:"+config["save_dir"]+str(epoch)+"epoch/")
                plt.savefig(config["save_dir"]+str(epoch)+"epoch/"+str(key)+".jpg")
            
            loss_out = loss(output, data)
            post_out = post_process(output, data)
            post_process.append(metrics, loss_out, post_out)
            
    
    dt = time.time() - start_time
    post_process.display(metrics, dt, epoch, 777)
    print("------------------END-VAL---------------------")
    # if epoch >10 and DEBUG and dist.get_rank()==0:
    #     import ipdb;ipdb.set_trace()
    # metrics = sync(metrics)
    # if hvd.rank() == 0:
    #     post_process.display(metrics, dt, epoch)
    net.train()

def test(data_loader,net,config):
    print("--------------start-test--------------------")
    # begin inference
    preds = {}
    gts = {}
    cities = {}
    for ii, data in tqdm(enumerate(data_loader)):
        data = dict(data)
        with torch.no_grad():
            output = net(data)
            results = [x[0:1].detach().cpu().numpy() for x in output["reg"]]
            
            for idx, pred_traj in zip(data["argo_id"], results):
                preds[idx] = pred_traj.squeeze()
    import csv
    with open(f"{config['save_dir']}/predictions.csv", 'w', newline='') as csvfile:
        for _, pred in preds.items():
            for i, mode in enumerate(pred):
                writer = csv.writer(csvfile)
                writer.writerow(['X', 'Y'])
                for row in mode:
                    writer.writerow(row)
        csvfile.close()
        # for i, (argo_idx, pred_traj) in enumerate(zip(data["argo_id"], results)):
        #     preds[argo_idx] = pred_traj.squeeze()
        #     cities[argo_idx] = data["city"][i]
        #     gts[argo_idx] = data["gt_preds"][i][0] if "gt_preds" in data else None

    # # save for further visualization
    # res = dict(
    #     preds = preds,
    #     gts = gts,
    #     cities = cities,
    # )
    # torch.save(res,f"{config['save_dir']}/results.pkl")
    

    # # for test set: save as h5 for submission in evaluation server
    # from argoverse.evaluation.competition_util import generate_forecasting_h5
    # generate_forecasting_h5(preds, f"{config['save_dir']}/submit.h5")  # this might take awhile
    # print("finish generation")
        
def save_ckpt(net, opt, save_dir, epoch):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    state_dict = net.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()

    save_name = "%3.3f.ckpt" % epoch
    torch.save(
        {"epoch": epoch, "state_dict": state_dict, "opt_state": opt.opt.state_dict()},
        os.path.join(save_dir, save_name),
    )


# def sync(data):
#     data_list = comm.allgather(data)
#     data = dict()
#     for key in data_list[0]:
#         if isinstance(data_list[0][key], list):
#             data[key] = []
#         else:
#             data[key] = 0
#         for i in range(len(data_list)):
#             data[key] += data_list[i][key]
#     return data


if __name__ == "__main__":
    num_gpus = torch.cuda.device_count()

    assert "SLURM_JOB_ID" in os.environ
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    node_list = os.environ["SLURM_NODELIST"]
    addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
    # specify master port
    port=None
    if port is not None:
        os.environ["MASTER_PORT"] = str(port)
    elif "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29610"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = addr
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank % num_gpus)
    os.environ["RANK"] = str(rank)
    
    print(f"world_size: {world_size},LOCAL_RANK{rank}")
    
    torch.cuda.set_device(rank % num_gpus)
    main(rank,world_size)