from argparse import ArgumentParser
import importlib
from pathlib import Path
import sys
import logging
import yaml

import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from loader import load_model_path_by_args

ROOT = Path(__file__).parent
sys.path.append(str(ROOT.absolute()))

with open(ROOT / 'config' / 'model_config.yaml') as f:
    config = yaml.safe_load(f)
MODEL_NAME = config['model_name']
DATA_MODULE_NAME = config['dm_name']


def load_model():
    camel_name = "".join([i.capitalize() for i in MODEL_NAME.split("_")])
    try:
        model = getattr(
            importlib.import_module(
                "model." + MODEL_NAME, package=__package__),
            camel_name,
        )
    except Exception:
        raise ValueError(
            f"Invalid Module File Name or Invalid Class Name {MODEL_NAME}.{camel_name}!"
        )
    return model


def load_dm():
    camel_name = "".join([i.capitalize() for i in DATA_MODULE_NAME.split("_")])
    try:
        dm = getattr(
            importlib.import_module(
                "data." + DATA_MODULE_NAME, package=__package__),
            camel_name,
        )
    except Exception:
        raise ValueError(
            f"Invalid Module File Name or Invalid Class Name {DATA_MODULE_NAME}.{camel_name}!"
        )
    return dm

def infer_metric_best(metric: str) -> str:
    metric = metric.lower()
    higher_is_better = ['acc', 'precision', 'f1', 'recall']
    for i in higher_is_better:
        if i in metric:
            return 'max'
    else:
        return 'min'

def load_callbacks(args):
    callbacks = [plc.LearningRateMonitor(logging_interval="step"),]
    if args.save_top_k > 0:
        callbacks.append(
            plc.ModelCheckpoint(
                monitor=args.monitor,
                filename="-".join(["best", "{epoch:02d}", "{val_loss:.4f}", "{" + args.monitor + ":.4f}"]),
                save_top_k=args.save_top_k,
                mode=infer_metric_best(args.monitor),
                save_last=True,
                verbose=True,
            )
        )
    if args.min_epochs > 0:
        callbacks.append(
            plc.EarlyStopping(
                monitor=args.monitor, mode="min", patience=args.patience, min_delta=0.001, verbose=True
            ),
        )
    return callbacks

def save_config(params: dict):
    save_path = Path(__file__).parent / params.get('log_dir', './') / params.get('project_name', '')
    save_file = save_path / 'latest_config.yaml'
    with open(save_file, 'w') as f:
        yaml.dump(params, f)

def main(parent_parser):
    # load model specific
    model = load_model()
    if hasattr(model, "add_model_args"):
        parent_parser = model.add_model_args(parent_parser)
        logging.info('Added model specific args')
    dm = load_dm()
    if hasattr(dm, "add_data_args"):
        parent_parser = dm.add_data_args(parent_parser)
        logging.info('Added dm specific args')
    
    args = parent_parser.parse_args()
    
    # use nni to tune hyperparameters
    params = vars(args)
    if args.use_nni:
        import nni
        tuner_params = nni.get_next_parameter()
        logging.info('Got nni parameters')
        params.update(tuner_params)
    
    # load and save config
    if args.config_file is not None:
        with open(args.config_file, 'r') as f:
            config = yaml.safe_load(f)
        params.update(config)
    
    save_config(params)
    
    # global seed
    pl.seed_everything(params['seed'])
    
    # initilize data module
    dm = dm(**params)
    if args.stage == "test" or args.stage == "predict":
        dm.setup('test')
        test_dataloader = dm.test_dataloader()
    else:
        dm.setup('fit')
        train_dataloader = dm.train_dataloader()
        val_dataloader = dm.val_dataloader()
    # add some info from dataset
    ex_params = dm.add_data_info() if hasattr(dm, "add_data_info") else {}
    params.update(ex_params)
    
    # initilize model
    if args.load_model is not None:
        model = model.load_from_checkpoint(args.load_model, **params)
    else:
        model = model(**params)
    
    # restart setting
    if args.load_dir is not None or args.load_ver is not None or args.load_v_num is not None:
        load_path = load_model_path_by_args(args)
        logging.info(f'Loading model from {load_path}...')
    else:
        load_path = None
    
    # initilize logger
    if args.stage == "debug" or args.stage == "test" or args.stage == "predict":
        logger = None
    else:
        if args.wandb:
            logger = WandbLogger(
                project=args.project_name, save_dir=args.log_dir, log_model=False,
                version=args.load_ver,
            )
            logger.watch(model)
        else:
            logger = TensorBoardLogger(
                save_dir=args.log_dir, name=args.project_name, version=args.load_ver,)
    
    # initilize callbacks
    args.callbacks = load_callbacks(args)
    
    # mutli GPU strategy
    if args.multi_gpu_strategy == 'ddp':
        args.multi_gpu_strategy = DDPStrategy(find_unused_parameters=args.find_unused_parameters)
    
    # initilize trainer
    # trainer = pl.Trainer.from_argparse_args(args)
    trainer = pl.Trainer(
        default_root_dir=args.log_dir,
        strategy=args.multi_gpu_strategy,
        logger=logger,
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        min_epochs=None if args.min_epochs == 0 else args.min_epochs,
        precision=args.precision,
        fast_dev_run=args.stage == "debug",
        callbacks=None if args.stage == "debug" else args.callbacks,
        gradient_clip_val=args.gradient_clip_val,
    )
    
    if args.stage == "predict":
        trainer.predict(model, test_dataloader, ckpt_path=load_path)
    elif args.stage == "test":
        trainer.test(model, test_dataloader, ckpt_path=load_path)
    else:
        # start training
        trainer.fit(model, train_dataloader, val_dataloader, ckpt_path=load_path)
        # end of training
        if args.use_nni:
            logging.info(f'{trainer.global_rank}: Running final report for nni')
            metrics = trainer.validate(model, dataloaders=val_dataloader)
            if trainer.is_global_zero:
                nni.report_final_result(metrics[0][args.monitor])
        

def get_params():
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument("--log_dir", type=str, default="pl_log", help="log directory")
    parser.add_argument("--project_name", type=str, default="default_run", help="project name")
    parser.add_argument("--batch_size_per_gpu", default=64, type=int, help='batch size per gpu')
    parser.add_argument("--num_workers", default=2, type=int, help='num workers in dataloader')
    parser.add_argument("--seed", default=42, type=int, help='global random seed')
    parser.add_argument("--learning_rate", default=1e-3, type=float, help='global learning rate')
    parser.add_argument("--weight_decay", default=0.05, type=float, help='global weight decay')
    parser.add_argument("--max_epochs", default=30, type=int, help='max epochs')
    parser.add_argument("--min_epochs", default=0, type=int, help='min epochs for early stop. 0 for no early stop')
    parser.add_argument("--patience", default=3, type=int, help='patience for early stop')
    parser.add_argument("--gpus", default="0", type=str, help='gpus to use')
    parser.add_argument("--multi_gpu_strategy", default=None, help='strategy for multi gpu')
    parser.add_argument("--precision", type=int, default=32, help='training precision', choices=[32, 16])
    parser.add_argument("--gradient_clip_val", type=float, default=0., help='gradient clip val. 0 means no clip')
    parser.add_argument("--save_top_k", type=int, default=0, help='save top k best models, 0 means no save')
    parser.add_argument("--stage", type=str, choices=["debug", "fit", "test", "predict"], default="fit", help='running stage')
    parser.add_argument("--find_unused_parameters", action='store_true', help='find unused parameters')

    # Restart Control
    parser.add_argument("--config_file", type=str, default=None, help='config file')
    parser.add_argument("--load_model", default=None, type=str, help='load model for fine-tuning')
    parser.add_argument("--load_best", action="store_true", help='load best model')
    parser.add_argument("--load_dir", default=None, type=str, help='load model directory')
    parser.add_argument("--load_ver", default=None, type=str, help='load model version')
    parser.add_argument("--load_v_num", default=None, type=int, help='load model version number')

    # Training Info
    parser.add_argument("--wandb", action="store_true", help='use wandb')
    # parser.add_argument("--no_early_stop", action="store_true")
    parser.add_argument("--use_nni", action="store_true", help='use nni')
    parser.add_argument("--monitor", type=str, default="val_acc", help='monitor to use for early stop')

    # parser = pl.Trainer.add_argparse_args(parser)

    # Reset Some Default Trainer Arguments' Default Values
    # parser.set_defaults(max_epochs=40)
    # parser.set_defaults(gpus=1)
    return parser


if __name__ == "__main__":
    try:
        main(get_params())
        
    except Exception as exception:
        raise exception
