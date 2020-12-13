#!/usr/bin/python

import os, sys
import json
import numpy as np
import re

### YOUR CODE HERE: write at least three functions which solve
### specific tasks by transforming the input x and returning the
### result. Name them according to the task ID as in the three
### examples below. Delete the three examples. The tasks you choose
### must be in the data/training directory, not data/evaluation.
def solve_0a938d79(x):
    dim1,dim2 = x.shape
    print(dim1,dim2)
    x_ = x.copy()
    result = np.argwhere(x_ > 0)
    values = x_[x_>0]
    if dim1 > dim2:
        index0 = result[0][0]
        index1 = result[1][0]
        intervals = index1-index0
        i = index0
        j = index1
        while i < dim1:
            x_[i] = values[0]
            i+=2*intervals
        while j < dim1:
            x_[j] = values[1]    
            j +=2*intervals
    else:
        result = np.argwhere(x_ > 0)
        values = x_[x_>0]
        index0 = result[0][1]
        index1 = result[1][1]
        intervals = index1-index0
        i = index0
        j = index1
        while i < dim2:
            x_[:,i] = values[0]
            i +=2*intervals
        while j < dim2:
            x_[:,j] = values[1]    
            j +=2*intervals
    return x_

def solve_68b16354(x):
    """
    Solves the task 68b16354
   
    Description:
       
    Correctness:
        All the given cases are solved.
   
    Arguments:
        x : Input Numpy array of dimension 2 and equal shape
            values for both axes
    Returns:
        A copy of x with required transformations applied
    """
    x_ = x.copy()          # make a copy of x
    x_ = np.flip(x_,0)     # flip the array horizontally
    return x_              # return the solution

def solve_dc0a314f(x):
    """
    Solves task dc0a314f
    
    Description:
       
    Correctness:
        All the given cases are solved.
   
    Arguments:
        x : Input Numpy array of dimension 2 and equal shape
            values for both axes
    Returns:
        A copy of x with required transformations applied
    """
    
    dim1,dim2 = x.shape              # get the dimensions(rows,columns respectively) from the array
    half = int(dim1/2)               # calculate the middle for the rows
    x_ = x.copy()                    # make a copy of x
    result = np.argwhere(x_ == 3)    # find the locations of the green squares
    x3 = x_[half:dim2,0:half]        # get the lower most left part of the array
    x4 = np.flip(x3,1)               # flip the sliced array from above vertically to get the right half
    x2 = np.concatenate([x3,x4],axis=1)  # concatenate the arrays together to form the lower half of the original array
    x1 = np.flip(x2,0)                   # flip the lower half horizontally to get the upper half  
    x__ = np.concatenate([x1,x2],axis=0) # put the lower half and upper half together to create the array
    x5 = x__[result[0][0]:result[-1][0]+1,result[0][1]:result[-1][1]+1]  # slice the created array by the location of the green squares
    return x5                                                            # return the solution

def main():
    # Find all the functions defined in this file whose names are
    # like solve_abcd1234(), and run them.

    # regex to match solve_* functions and extract task IDs
    p = r"solve_([a-f0-9]{8})" 
    tasks_solvers = []
    # globals() gives a dict containing all global names (variables
    # and functions), as name: value pairs.
    for name in globals(): 
        m = re.match(p, name)
        if m:
            # if the name fits the pattern eg solve_abcd1234
            ID = m.group(1) # just the task ID
            solve_fn = globals()[name] # the fn itself
            tasks_solvers.append((ID, solve_fn))

    for ID, solve_fn in tasks_solvers:
        # for each task, read the data and call test()
        directory = os.path.join("..", "data", "training")
        json_filename = os.path.join(directory, ID + ".json")
        data = read_ARC_JSON(json_filename)
        test(ID, solve_fn, data)
    
def read_ARC_JSON(filepath):
    """Given a filepath, read in the ARC task data which is in JSON
    format. Extract the train/test input/output pairs of
    grids. Convert each grid to np.array and return train_input,
    train_output, test_input, test_output."""
    
    # Open the JSON file and load it 
    data = json.load(open(filepath))

    # Extract the train/test input/output grids. Each grid will be a
    # list of lists of ints. We convert to Numpy.
    train_input = [np.array(data['train'][i]['input']) for i in range(len(data['train']))]
    train_output = [np.array(data['train'][i]['output']) for i in range(len(data['train']))]
    test_input = [np.array(data['test'][i]['input']) for i in range(len(data['test']))]
    test_output = [np.array(data['test'][i]['output']) for i in range(len(data['test']))]

    return (train_input, train_output, test_input, test_output)


def test(taskID, solve, data):
    """Given a task ID, call the given solve() function on every
    example in the task data."""
    print(taskID)
    train_input, train_output, test_input, test_output = data
    print("Training grids")
    for x, y in zip(train_input, train_output):
        yhat = solve(x)
        show_result(x, y, yhat)
    print("Test grids")
    for x, y in zip(test_input, test_output):
        yhat = solve(x)
        show_result(x, y, yhat)

        
def show_result(x, y, yhat):
    print("Input")
    print(x)
    print("Correct output")
    print(y)
    print("Our output")
    print(yhat)
    print("Correct?")
    # if yhat has the right shape, then (y == yhat) is a bool array
    # and we test whether it is True everywhere. if yhat has the wrong
    # shape, then y == yhat is just a single bool.
    print(np.all(y == yhat))

if __name__ == "__main__": main()

