"""Run VDN-H3's single-GPU infer.py with the torch.accelerator shim."""

import sys

sys.path.insert(0, "/xs-fd-nas/workspace_zhaoxinrui/vdn-h3/vdn-minimax-h3")

import vdn_shim  # noqa: F401

import src.inference.infer as infer

if __name__ == "__main__":
    infer.main()
