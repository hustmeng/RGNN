#Reference:

#1. C.W. Park, M. Kornbluth, J. Vandermause, C. Wolverton, B. Kozinsky, J.P. Mailoa, Accurate and scalable graph neural network force field and molecular dynamics with direct force architecture, npj Comput. Mater. 7(1) (2021) 73. 

#This code is extended from https://github.com/ken2403/gnnff.git, which has the MIT License.

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.

import os
import pathlib
import logging
import pickle
import torch
import torch.nn as nn

from RGNN.data.celldata import CellData
from RGNN.data.process import Datagraph
from RGNN.model.rgnn import rgnn
from RGNN.utils.evaluation import evaluate
from RGNN.utils.data import get_loader
from RGNN.utils.training import get_metrics, get_trainer
from RGNN.utils.script_utils import ScriptError, set_random_seed, count_params
from RGNN.utils.parsing import make_parser, read_from_json


logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def main(args):
    # set up learning environment
    device = torch.device("cuda:1" if args.cuda else "cpu")

    # set random seed
    if args.mode == "train":
        set_random_seed(args.seed, logging=logging)

    # get dataset
    logging.info("loading the dataset...")
    graphpath = args.graphpath
    if os.path.exists(graphpath):                #read graph data
       with open(graphpath,"rb") as f:
           data_list = pickle.load(f)
    else:
       graph = Datagraph(db_path=args.datapath, cutoff=args.cutoff, available_properties = ["forces"])
       data_list = graph()                       #process graph data
       with open(graphpath,"wb") as f:
            pickle.dump(data_list, f)
            
    
    dataset = CellData(db_path=args.datapath,data_list = data_list, cutoff=args.cutoff)
    # get dataloaders
    split_path = os.path.join(args.modelpath, "split.npz")
    conductance = torch.load(args.conductance_path,map_location=device)

    train_loader, val_loader, test_loader = get_loader(
        dataset, args, split_path=split_path, logging=logging
    )
    # setup property metrics
    metrics = get_metrics(args)
    #del data_list      #release large list 
    # train or eval
    if args.mode == "train":
        # build model
        logging.info("building the rgnn model...")
        model = rgnn(
            n_node_feature=args.n_node_feature,
            n_edge_feature=args.n_edge_feature,
            n_message_passing=args.n_message_passing,
            cutoff=args.cutoff,
            gaussian_filter_end=args.gaussian_filter_end,
            trainable_gaussian=args.trainable_gaussian,
            share_weights=args.share_weights,
            return_intermid=False,
            properties=args.predict_property,
            n_output_layers=args.n_output_layers,
        )
        if args.parallel:
            model = nn.DataParallel(model)
        logging.info(
            "The model you built has: {} parameters".format(count_params(model))
        )
        # training
        trainer = get_trainer(args, model, train_loader, val_loader, metrics)
        logging.info("training ...")
        torch.backends.cudnn.benchmark = True
        trainer.train(
            device, n_epochs=args.n_epochs, lambda_=args.regularization_lambda, conductance = conductance
        )
        logging.info("... training done!")

    # train or eval
    elif args.mode == "eval":
        evaluation_fp = os.path.join(args.modelpath, "evaluation.txt")
        if os.path.exists(evaluation_fp):
            os.remove(evaluation_fp)

        # load model
        logging.info("loading trained model...")
        
        model = rgnn(
            n_node_feature=args.n_node_feature,
            n_edge_feature=args.n_edge_feature,
            n_message_passing=args.n_message_passing,
            cutoff=args.cutoff,
            gaussian_filter_end=args.gaussian_filter_end,
            trainable_gaussian=args.trainable_gaussian,
            share_weights=args.share_weights,
            return_intermid=True,
            properties=args.predict_property,
            n_output_layers=args.n_output_layers,
        ).to(device)
        
        checkpoint = torch.load(args.checkpoint_path,map_location=device)
        model.load_state_dict(checkpoint["model"])
        # run evaluation
        logging.info("evaluating...")
        with torch.no_grad():
            evaluate(
                args,
                model,
                train_loader,
                val_loader,
                test_loader,
                device,
                metrics=metrics,
                conductance = conductance,
            )
        logging.info("... evaluation done!")

    else:
        raise ScriptError("Unknown mode: {}".format(args.mode))


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()
    if args.input == "from_json":
        args = read_from_json(args.json_path)
        main(args)

