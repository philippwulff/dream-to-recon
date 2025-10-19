from .mlp import ImplicitNet
from .resnetfc import ResnetFC


def make_head(conf, d_in, d_out, allow_empty=True, **kwargs):       # TODO fix allow empty. Should be false usually
    if conf["type"] == "mlp":
        head = ImplicitNet.from_conf({"d_out": d_out, **conf}, d_in)
    elif conf["type"] == "resnet":
        head = ResnetFC.from_conf({"d_out": d_out, **conf}, d_in)
    elif conf["type"] == "empty" and allow_empty:
        head = None
    else:
        raise NotImplementedError("Unsupported MLP type")
    return head



def make_mlp(conf, d_in, d_latent=0, allow_empty=False, **kwargs):
    mlp_type = conf.get("type", "mlp")  # mlp | resnet
    if mlp_type == "mlp":
        net = ImplicitNet.from_conf(conf, d_in + d_latent, **kwargs)
    elif mlp_type == "resnet":
        net = ResnetFC.from_conf(conf, d_in, d_latent=d_latent, **kwargs)
    elif mlp_type == "empty" and allow_empty:
        net = None
    else:
        raise NotImplementedError("Unsupported MLP type")
    return net