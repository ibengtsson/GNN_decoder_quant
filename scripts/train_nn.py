import argparse
import sys

sys.path.append("..")
from src.decoder import Decoder

def main():
    
    # command line parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--configuration", required=True)
    args = parser.parse_args()  
    
    # create decoder object
    config_file = args.configuration
    decoder = Decoder(config_file)
    decoder.train()
    
if __name__ == "__main__":
    main()