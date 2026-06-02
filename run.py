# -*- coding: utf-8 -*
import random
import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from utils.opts import parse_opt
from utils.baseTrainer import main_worker


if __name__ == '__main__':
    args = parse_opt()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False
        warnings.warn(
            'You have chosen to seed training. '
            'This will turn on the CUDNN deterministic setting, '
            'which can slow down your training considerably! '
            'You may see unexpected behavior when restarting '
            'from checkpoints.'
        )

    if args.gpu is None:
        args.gpu = 0

    main_worker(args.gpu, args)
