from easydict import EasyDict as edict

def HParams(**kwargs):
    ed = edict()
    for name, value in kwargs.items():
        ed[name] = value
    return ed
