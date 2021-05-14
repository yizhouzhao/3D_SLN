import subprocess
import os

def run_blender(args):
    subprocess.run(['/home/yizhou/blender-2.92.0-linux64/blender', '-b', '-P', './render/render_caller.py', '--', os.path.abspath(args.test_dir)])

def run_blender_mask_depth(args):
    subprocess.run(['/home/yizhou/blender-2.92.0-linux64/blender', '-b', '-P', './render/semantic_depth_caller.py', '--', os.path.abspath(args.test_dir)])