#!python3
#
# Copyright 2020 Georgia Tech Research Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# Author(s): Tony C. Pan
#

import argparse
import random

import numpy as np

import matrix_io

from timeit import default_timer as timer


# add noise based on row std.   This adds a gaussian noise to the data.
def row_perturb(input, noise_level = 0., method='normal'):
    if noise_level == 0.:
        return input

    # perturb global noise (per data set.).
    n = np.random.uniform(noise_level * 0.8, noise_level * 1.2)
    ns = np.random.uniform(n * 0.8, n * 1.2, input.shape[0])
    # get data's per row std (computed "using" column, axis = 1).
    sds = np.std(input, axis = 1, dtype=np.float64)
    # replace 0.
    ns2 = np.random.uniform(0.01, 0.15, input.shape[0])
    sds = np.where( sds == 0, ns2, sds)
    sds = sds * ns

    if method == 'normal':
        noise = np.random.normal(loc=0., scale=sds, size=(input.shape[1], input.shape[0])).T
    elif method == 'lognormal':
        noise = np.random.lognormal(mean=0., sigma=sds, size=(input.shape[1], input.shape[0])).T

    return input + noise

# this increases the standard deviation of the data. 
# def row_perturb2(input, noise_level = 0., method='normal'):
#     if noise_level == 0:
#         return input

#     # perturb global noise (per data set.).
#     random.seed()  # based on system time.
#     n = random.uniform(noise_level * 0.8, noise_level * 1.2)
    
#     ns = np.random.uniform(n * 0.8, n * 1.2, input.shape[0])
#     # get data's per row std (computed "using" column, axis = 1).

#     return input * ns

# add noise based on matrix std. This adds a gaussian noise to the data.
def matrix_perturb(input, noise_level = 0, method='normal'):
    if noise_level == 0.:
        return input

    # perturb global noise (per data set.).
    n = np.random.uniform(noise_level * 0.8, noise_level * 1.2)
    
    # get data's per row std (computed "using" column, axis = 1).
    sd = np.mean(np.std(input, axis = 1, dtype=np.float64))
    sd = random.uniform(0.01, 0.15) if sd == 0 else sd
    sd = sd * n

    if method == 'normal':
        noise = np.random.normal(loc=0., scale=sd, size=input.shape)
    elif method == 'lognormal':
        noise = np.random.lognormal(mean=0., sigma=sd, size=input.shape)

    return input + noise


def main():
    parser = argparse.ArgumentParser(description='perturbation - add per-gene and per-dataset noise.')
    parser.add_argument('--local-method', action='store', default='normal',
                        choices=['normal'],
                        help='local perturbation method, default = normal')
    parser.add_argument('--local-noise-level', type=float, action='store', default=0.,
                        help='local noise level.  in range (0, 1).  default 0 (false)')
    parser.add_argument('--global-method', action='store', default='lognormal',
                        choices=['normal', 'lognormal'],
                        help='global perturbation method, default = lognormal')
    parser.add_argument('--global-noise-level', type=float, action='store', default=0.,
                        help='global noise level.  in range (0, 1).  default 0 (false)')
    parser.add_argument('-i', '--input', action='store', default=None,
                        help='input EXP formatted file [random data if not given]', required=True)
    parser.add_argument('-o', '--output', action='store', default=None,
                        help='input EXP formatted file', required=True)
    parser.add_argument('--seed', type=int, action='store', default=None,
                        help='random generator seed.  default to system time')
    

    args = parser.parse_args()

    input_fn = args.input
    output_fn = args.output
    local_method = args.local_method
    local_noise = args.local_noise_level
    global_method = args.global_method
    global_noise = args.global_noise_level
    seed = args.seed

    if seed is not None:
        np.random.seed(seed)  # based on system time, or supplied seed.


    start = timer()
    ipd = matrix_io.read(input_fn, dtype=np.float64)
    di = ipd.to_numpy()
    end = timer()
    print('input {} s'.format(end - start))

    start = timer()
    ld = row_perturb(di, noise_level = local_noise, method=local_method)
    end = timer()
    print('add local noise: {} s'.format(end - start))

    start = timer()
    gd = matrix_perturb(ld, noise_level = global_noise, method=global_method)
    end = timer()
    print('add global noise: {} s'.format(end - start))

    start = timer()
    ipd[:] = gd
    matrix_io.write(ipd, output_fn)
    end = timer()
    print('output {} s'.format(end - start))
    pass
    
if __name__ == '__main__':
    main()