from ACJ import ACJ
import numpy as np
import pickle
import random

if __name__ == "__main__":
    	with open(r"ACJAssignment1.pkl", "rb") as input_file:
        	acj = pickle.load(input_file)
	acj.maxRounds = 1000;
    	with open(r"ACJAssignment1.pkl", "wb") as output_file:
        	pickle.dump(acj, output_file)
	
