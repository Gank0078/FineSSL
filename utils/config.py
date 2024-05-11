from yacs.config import CfgNode as CN

_C = CN()

_C.dataset = ""  # Dataset name
_C.root = ""  # Directory where datasets are stored
_C.backbone = ""
_C.resolution = 224
_C.stride = 16

_C.output_dir = None  # Directory to save the output files (like log.txt and model weights)
_C.resume = None  # Path to a directory where the files were saved previously
_C.checkpoint_freq = 0  # How often (epoch) to save model during training. Set to 0 or negative value to only save the last one
_C.print_freq = 10  # How often (batch) to print training information

_C.seed = None
_C.deterministic = False
_C.gpu = None
_C.num_workers = 8
_C.prec = "fp16"  # fp16, fp32, amp

_C.num_epochs = 10
_C.batch_size = 256
_C.lr = 0.03
# _C.lr2 = 0.03
_C.weight_decay = 5e-4
_C.momentum = 0.9

_C.tau = 0.0
_C.downsampling = False
_C.n_max = 100
_C.test_ensemble = False

_C.finetune = False
_C.bias_tuning = False
_C.bn_tuning = False  # only for resnet
_C.vpt_shallow = False
_C.vpt_deep = False
_C.vpt_last = False

_C.adaptformer = False
_C.ffn_num = 64
_C.ln_opt = "in"
_C.adapter_scalar = 0.1
_C.scalar_learnable = False
_C.ffn_opt = "parallel"

_C.lam_tokens = 1.0

_C.vpt_len = 0
_C.adapter = False
_C.adapter_dim = 0
_C.lora = False
_C.lora_dim = 0
_C.ssf = False
_C.partial = None

_C.zero_shot = False

_C.eval_only = False
_C.model_dir = None
_C.load_epoch = None
_C.template = None

_C.DATA = CN()
_C.DATA.NAME = ""
_C.DATA.DATAPATH = ""
_C.DATA.NUMBER_CLASSES = -1
_C.DATA.BATCH_SIZE = 32
# Number of data loader workers per training process
_C.DATA.NUM_WORKERS = 4
# Setting about LTSSL
_C.DATA.NUM_L = 50
_C.DATA.NUM_U = 400
_C.DATA.IMB_L = 1.0
_C.DATA.IMB_U = 1.0
_C.DATA.MU_U = 2
_C.DATA.LABEL_RATIO = 0.01
_C.DATA.out_ulab = False
_C.block_num = None

_C.th = 0.7
_C.patch_size = 16
_C.mode = 0
_C.alpha = 8.0
_C.th_min = 0.0
_C.w_con = 1.0
_C.total_steps = 500
_C.rand_init = True
_C.rand_init1 = True
_C.smoothing = 0.1
