import os

def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)

def make_folder(path, version):
     if not os.path.exists(os.path.join(path, version)):
         os.makedirs(os.path.join(path, version))