import hydra
import time
import torch
import torch.multiprocessing as multiprocessing
from train import main 

@hydra.main(config_path='conf/config.yaml')
def launch_runs(args):
    torch.multiprocessing.set_start_method('spawn')
    # print all parameters
    print(args.pretty())
    if type(args.gpu_ids) == int:
        gpu_ids = [args.gpu_ids]
    else:
        gpu_ids = [int(i) for i in args.gpu_ids.split(',')]

    # run experiments in parallel
    jobs = []
    for idx in gpu_ids:
        p = multiprocessing.Process(target=launch_run, args=(idx,args))
        jobs.append(p)
        p.start()


def launch_run(gpu_id,args):
    print('launching ' + args.domain_name + ', ' + args.task_name +  ' experiment on GPU:',gpu_id)
    ts = time.gmtime() 
    ts = time.strftime("%m-%d--%I-%M-%S-%p", ts)    
    env_name = args.domain_name + '-' + args.task_name
    exp_name = '/'+ env_name + '-im' + str(args.image_size) +'-b'  + str(args.batch_size)+ '-gpu' + str(gpu_id) + '-' + ts 
    args.work_dir = args.root_dir + args.work_dir  + exp_name
    main(args,gpu_id)




if __name__ == "__main__":
    launch_runs()