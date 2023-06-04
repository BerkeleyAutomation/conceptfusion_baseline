import argparse
import json
import numpy as np
parser = argparse.ArgumentParser(description='Transforms a set of poses from one frame to another')
parser.add_argument('--transform', type=str, required=True)
parser.add_argument('--savedir', type=str, required=True)
args = parser.parse_args()

transform_file = open(args.transform)
data = json.load(transform_file)
data = data['frames']
f = open(args.savedir, 'w')

for frame in data:
    m = np.array(frame['transform_matrix'])
    m = m[:3]
    f.write(str(m[0])[1:-1])
    f.write('\n')
    f.write(str(m[1])[1:-1])
    f.write('\n')
    f.write(str(m[2])[1:-1])
    f.write('\n')
    f.write('\n')

f.close()
