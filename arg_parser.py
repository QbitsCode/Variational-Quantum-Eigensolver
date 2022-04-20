import argparse


def build_default_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', help='path of local output directory', type=str, required=True)
    parser.add_argument('--jobID', help='computation job id', type=str, required=True)
    parser.add_argument('--algo', help='VQE algorithm', required=True)
    parser.add_argument('--molecule', help='molecule chemical formula', type=str, required=True)
    parser.add_argument('--nlayer', help='number of layers in VQE-UCC', type=int, required=True)
    parser.add_argument('--omega', help='energy shift for FS method', type=float, required=False)
    parser.add_argument('--basis', help='Basis set name', type=str, required=True)
    parser.add_argument('--bondlen', help='Bond Length for diatomic molecules ; input 0 for ground stable length', type=float, required=False)
    parser.add_argument('--refstate', help='Reference state for initial guess (HF...)', type=str, required=False)
    parser.add_argument('--device', help='CPU or GPU', type=str, required=True)
    parser.add_argument('--lb', help='for pec - lower bound of length range in Å ', type=float, required=False)
    parser.add_argument('--ub', help='for pec - upper bound of length range in Å ', type=float, required=False)
    parser.add_argument('--step', help='for pec - step between 2 measures in Å ', type=float, required=False)
    return parser