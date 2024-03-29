#Reference:

#1. C.W. Park, M. Kornbluth, J. Vandermause, C. Wolverton, B. Kozinsky, J.P. Mailoa, Accurate and scalable graph neural network force field and molecular dynamics with direct force architecture, npj Comput. Mater. 7(1) (2021) 73. 
#This code is extended from https://github.com/ken2403/gnnff.git, which has the MIT License.

# Thanks to the original authors for their generous sharing. If you need to use the code, please refer to the original version of the code and literature.


import json
from argparse import ArgumentParser
from argparse import Namespace


__all__ = ["make_parser", "read_from_json"]


def make_parser():
    """
    Returns
    -------
    main_parser : argparse.ArgumentParser
        parser object
    """
    main_parser = ArgumentParser(
        description="Command to run the training of SchNetTriple"
    )

    # subparser structure
    input_subparsers = main_parser.add_subparsers(
        dest="input", help="Input file arguments", required=True
    )

    # from_json arguments
    json_subparser = input_subparsers.add_parser(
        "from_json",
        help="load from json help",
    )
    json_subparser.add_argument("json_path", help="argument json file path")

    # from_poscar arguments
    poscar_subparser = input_subparsers.add_parser(
        "from_poscar",
        help="load from POSCAR help",
    )
    poscar_subparser.add_argument("poscar_path", help="input poscar file path")
    poscar_subparser.add_argument("model_path", help="learned model path")
    poscar_subparser.add_argument(
        "--cuda", help="compute device flag", action="store_true"
    )
    poscar_subparser.add_argument(
        "--cutoff", type=float, help="cutoff radious", required=True
    )

    return main_parser


def read_from_json(jsonpath: str) -> Namespace:
    """
    This function reads args from the .json file and returns the content as a namespace dict.

    Parameters
    ----------
    jsonpath : str
        path to the .json file

    Returns
    -------
    namespace_dict : Namespace
        namespace object build from the dict stored into the given .json file.
    """
    with open(jsonpath) as handle:
        dict = json.loads(handle.read())
        namespace_dict = Namespace(**dict)
    return namespace_dict
