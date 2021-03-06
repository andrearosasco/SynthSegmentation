

class Config:
    class General:
        device = 'cuda'

    class Train:
        lr = 0.01
        momentum = 0.9
        weight_decay = 0.0001
        log_every = 10
        epoch = 100

    class Eval:
        wandb = True

    class Data:
        class Eval:
            mb_size = 64
            paths = None

        class Train:
            mb_size = 64

        num_worker = 30

    @classmethod
    def to_dict(cls, target=None):
        if target is None:
            target = cls

        res = {}
        for k in dir(target):
            if not k.startswith('__') and k != 'to_dict':
                attr = getattr(target, k)
                if type(attr) == type:
                    res[k] = cls.to_dict(attr)
                else:
                    res[k] = attr
        return res

