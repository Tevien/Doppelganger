from dpplgngr.models.vanilla_ae import VanillaLightning

def get_model(name, args):
    if name == "vanilla_ae":
        return VanillaLightning(args["input_size"], args["mid_size"], args["latent_size"])
    else:
        raise ValueError("Model name not recognized")