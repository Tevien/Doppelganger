from dpplgngr.datasets.tabular import Tabular

def get_dataset(name, args):
    if name == "tabular":
        train = Tabular(filename=args["inputfile"], tr_test="train")
        test = Tabular(filename=args["inputfile"], tr_test="test")
        return train, test
    else:
        raise ValueError("Dataset name not recognized")