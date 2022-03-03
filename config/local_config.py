class Config:
    class General:
        device = 'cpu'

    class Train:
        lr = 0.01
        momentum = 0.9
        weight_decay = 0.0
        log_every = 10
        epoch = 10

    class Eval:
        wandb = True

    class Data:
        class Eval:
            mb_size = 1
            paths = {'ycb': '../../Downloads/YCB_Video_Dataset'}

        class Train:
            mb_size = 1

        num_worker = 0

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
