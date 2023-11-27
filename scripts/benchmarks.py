import torch
import argparse
from torch.profiler import profile, record_function, ProfilerActivity

def main():
    
    # read input arguments and potentially overwrite default settings
    parser = argparse.ArgumentParser(
        description="Benchmarking settings"
        )
    parser.add_argument("-t", "--tensor_size", required=True)
    parser.add_argument("-d", "--device", required=True)
    args = parser.parse_args()
    
    # set device
    device = torch.device(args.device)
    
    # initialise tensor
    sz = int(args.tensor_size)
    x = torch.rand((sz, sz)).to(device)
    print(f"{x.shape=}")
    
    # benchmark
    with profile(
        activities=[ProfilerActivity.CPU], 
        with_stack=True,
        record_shapes=True
    ) as prof:
        with record_function("benchmark"):
            # expand a dimension
            # x = torch.unsqueeze(x, -1)
            x = x[:, :, None]
    print(f"{x.shape=}")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
if __name__ == "__main__":
    main()