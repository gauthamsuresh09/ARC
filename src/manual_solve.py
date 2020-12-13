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
    """
    Solves task 0a938d79
    
    Description:
        This task involves filling in elements row wise or column wise depending on the dimensions of the array
        and the non-zero elements. First, I check which dimension(row or column) is larger.Then,the non-zero elements 
        are identified and the distance between them is calculated (in rows or columns) and hereafter the 
        appropriate rows/columns are filled with non-zero elements in the intervals measured by the distance.
        
    Correctness:
        All the given cases are solved.
   
    Arguments:
        x : Input Numpy array of dimension 2 and unequal shape
            values for both axes
    Returns:
        A copy of x with required transformations applied
    """
    dim1,dim2 = x.shape                # get the dimensions(rows,columns respectively) from the array
    x_ = x.copy()                      # make a copy of x
    result = np.argwhere(x_ > 0)       # find the locations of the non-zero points
    values = x_[x_>0]                  # get the values of the non-zero points
    if dim1 > dim2:                    # check if rows are greater than columns
        index0 = result[0][0]          # get the row index of the first non-zero point
        index1 = result[1][0]          # get the row index of the second non-zero point
        intervals = index1-index0      # find the distance between them
        i = index0                     # start from index0
        j = index1                     # start from index1
        while i < dim1:
            x_[i] = values[0]          # assign the first non-zero values to row i
            i+=2*intervals             # increment by twice the interval size
        while j < dim1:
            x_[j] = values[1]          # assign the second non-zero values to row j
            j +=2*intervals            # increment by twice the interval size
    else:
        index0 = result[0][1]          # get the column index of the first non-zero point
        index1 = result[1][1]          # get the column index of the second non-zero point
        intervals = index1-index0      # find the distance between them
        i = index0                     # start from index0
        j = index1                     # start from index1
        while i < dim2:
            x_[:,i] = values[0]       # assign the first non-zero values to column i
            i +=2*intervals           # increment by twice the interval size
        while j < dim2:
            x_[:,j] = values[1]       # assign the second non-zero values to column j
            j +=2*intervals           # increment by twice the interval size
    return x_                         # return solution


def solve_68b16354(x):
    """
    Solves the task 68b16354
   
    Description:
        The task involves finding the mirror image of rows in the given array.The rows in the array can be
        flipped which leads to the solution.
       
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
        This task entails filling in the values for the missing pieces (green colours). I found that for finding the
        solution,the array should be created as a symmetric one and then the missing pieces should be gleaned from that.
        To create such a symmetric array, I divide the array into four halves and chose the lower most left part 
        of the array. After this,I flipped this part vertically to obtain the lower right half and joined the two halves
        together (left and right). When I flip this horizontally,I get the upper part of the array and from there 
        I can create the symmetric array.On comparison of this created array with the original array,I can easily find 
        the solution.
       
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

def solve_af902bf9(x):
    """
    Solves task af902bf9
    
    Description:
        This task involves identifying squares and filling in the center with red colour (2).First, I get the locations
        of non-zero values in the array. Then, I slice the squares from the array by the locations.The next step
        is to fill in the red colour for appropriate rows (all rows except rows containing corner points of the square). 
        Finally, I revert some elements of the square to black/0 (because columns containing corner points of the square
        should not have red elements)
       
    Correctness:
        All the given cases are solved.
   
    Arguments:
        x : Input Numpy array of dimension 2 and equal shape
            values for both axes
    Returns:
        A copy of x with required transformations applied
    """
    x_ = x.copy()                    # make a copy of x
    result = np.argwhere(x_>0)       # find the locations of the corner point of the squares
    i = 0                           
    while i < len(result):           
        corner1 = result[i]          # get the left first corner on top of the square
        corner4 = result[i+3]        # get the right most corner on bottom of the square
        i+=4                         # increment for the next square
        x_square = x_[corner1[0]:corner4[0]+1,corner1[1]:corner4[1]+1] # slice the square from the array
        dim1,dim2 = x_square.shape        # get the dimensions(rows,columns respectively) from the array
        x_square[1:dim1-1] = 2            # set the rows to 2 for all rows except the ones with corner point
        x_square[1:dim1-1,0] = 0          # set the first columns to 0 for all rows except the ones with corner point
        x_square[1:dim1-1,dim2-1] = 0     # set the last columns to 0 for all rows except the ones with corner point
        x_[corner1[0]:corner4[0]+1,corner1[1]:corner4[1]+1] = x_square # set the created square to the original array
    return x_


def solve_de1cd16c(x):
    """
    Solves task de1cd16c
    
    Description:
        This task is about counting isolated points in a cluster of different colour and returning the cluster 
        colour/value with the most number of these points. I start with taking the unique elements in the array.
        Next, for every unique element, I choose an initial point and search for the immediate neighbour. If the
        element does not have one in the array, then I mark it as the element to be counted. After doing this, 
        I count this element in the clusters of the other elements and find the maximum count. The cluster with the
        maximum count is the solution.
       
    Correctness:
        All the given cases are solved.
   
    Arguments:
        x : Input Numpy array of dimension 2 and equal shape
            values for both axes
    Returns:
        A copy of x with required transformations applied
    """
    x_ = x.copy()                               # make a copy of x
    elements = np.unique(x_)                    # find all unique elements in the array
    find_element = -1                           # assume the element that is to counted is -1
    for element in elements:
        locations = np.argwhere(x_ == element)  # find locations of the given element
        x0,y0 = locations[0]                    # get the first location where the element is found
        x1,y1 = locations[1]                    # get the second location where the element is found
        if abs(x0-x1) == 1 or abs(y0-y1) == 1:  # check if the two elements are close to each other 
            continue                            # if yes,continue
        else:
            find_element = element              # if no, we have found the element that is to be counted
            break                               # search is completed,break the loop
    maxcount = 0                                # start with initial count as 0
    x__ = []                                    # start with initial array as empty
    for element in elements:
        if element == find_element:             # if current element is the element to be counted,move to the next element
            continue
        else:
            locations = np.argwhere(x_ == element) # find locations of the current element in the array
            element_area = x_[locations[0][0]:locations[-1][0]+1,locations[0][1]:locations[-1][1]+1] # slice the original array so as to get a array with current element in majority
            count = len(element_area[element_area == find_element]) # count the points where the find_element occurs
            if count > maxcount:
                maxcount = count                                    # update the maxcount to the largest found count
                x__ = np.array(element).reshape(1,1)                # reshape the current element as per requirement
    return x__                                                      # return the solution

def solve_83302e8f(x):
    """
    Solves the task 833302e8f

    Description:
        For this task, we are given a square grid with multiple squares (colour black)
        of equal dimension in it. The number of such square and their position
        are deterministic, for a given grid. The squares share boundary between
        each other (main colour of it being blue) and one or more blocks in the
        boundary can have same colour as square (black). Our task is to modify
        the grid in such a way that the squares that are connected with other
        square(s) are coloured with yellow and isolated ones are coloured with green.
        The colour of black connecting blocks in the boundary is also changed to yellow

    Correctness:
        All the given cases are solved.

    Arguments:
        x : Input Numpy array of dimension 2 and equal shape
            values for both axes
    Returns:
        A copy of x with required transformations applied
    """
    x = x.copy()  # Create copy of input array
    arr_len, _ = x.shape  # Get array length

    # Find length of squares in the grid
    # Start from top-left corner and go diagonally till
    # we find a blue box in any of the blocks on top or left
    # side of current block.
    square_len = -1
    i = 0
    while (square_len == -1 and i < arr_len):
        if np.any(x[:i + 1, i]) or np.any(x[i, :i + 1]):  # Check for any non-zero block on top and left
            square_len = i
        i += 1
    if square_len == -1:
        raise Exception("Square pattern not found. Please check the input")

    num_squares = int((1 + arr_len) / (1 + square_len))  # Get number of squares

    # Loop through all squares, starting from top-left
    # Modify colour of blocks according to pattern matched
    for i in range(num_squares):
        for j in range(num_squares):
            # Set edge indices for boundary around the square
            row_s = i * (square_len + 1) - 1  # Start of row index
            row_e = square_len * (i + 1) + i  # End of row index
            col_s = j * (square_len + 1) - 1  # Start of row column
            col_e = square_len * (j + 1) + j  # End of row column

            # Set corner indices for the square
            sq_row_s = row_s + 1  # Starting of row index
            sq_row_e = row_e  # Ending row index
            sq_col_s = col_s + 1  # Starting column index
            sq_col_e = col_e  # Ending column index

            # Handle cases of row start value being negative
            row_s_t = 0 if row_s == -1 else row_s

            # Check if square touches another one using its boundary
            # and set it to yellow if it does or green otherwise.
            # Check left boundary
            if col_s > -1:
                if np.any(x[row_s_t:row_e + 1, col_s] == 0):  # Check if any value is black
                    x[sq_row_s:sq_row_e, sq_col_s:sq_col_e] = 4  # Set the square to yellow
                    continue
            # Check right boundary
            if col_e < arr_len:
                if np.any(x[row_s_t:row_e + 1, col_e] == 0):  # Check if any value is black
                    x[sq_row_s:sq_row_e, sq_col_s:sq_col_e] = 4  # Set the square to yellow
                    continue

            col_s_t = 0 if col_s == -1 else col_s
            # Check top boundary
            if row_s > -1:
                if np.any(x[row_s, col_s_t:col_e + 1] == 0):  # Check if any value is black
                    x[sq_row_s:sq_row_e, sq_col_s:sq_col_e] = 4  # Set the square to yellow
                    continue
            # Check bottom boundary
            if row_e < arr_len:
                if np.any(x[row_e, col_s_t:col_e + 1] == 0):  # Check if any value is black
                    x[sq_row_s:sq_row_e, sq_col_s:sq_col_e] = 4  # Set the square to yellow
                    continue
            # If we reach here, then the square doesnt have
            # any black block in its boundary. This means
            # it should have green colour in output
            x[sq_row_s:sq_row_e, sq_col_s:sq_col_e] = 3  # Set the square to green
    # All the remaining black blocks are the ones in boundary
    # We can set all of them to yellow
    x[x == 0] = 4
    return x


def solve_6855a6e4(x):
    """
        Solves the task 6855a6e4

        Description:
            For this task, we are given a square grid with two red shaped and
            two grey shaped items in it. The items/figures with red colour have
            square bracket shape ( [ or ] ) and they appear either horizontally
            or vertically in an enclosing manner (eg. [ ] for vertical). The
            grey shapes occur outside the red shapes and their positions can be
            determined. Our task is to find the grey figures, flip them horizontally
            or vertically based on alignment of red shape, and then move them from
            outside to inside blocks of red enclosing brackets.

        Correctness:
            All the given cases are solved.

        Arguments:
            x : Input Numpy array of dimension 2 and equal shape
                values for both axes
        Returns:
            A copy of x with required transformations applied
    """
    x = x.copy()  # Create copy of input array
    # Get row and column positions for the red coloured
    # enclosing shapes
    row_positions, col_positions = np.nonzero(x == 2)
    # Get row and column positions for the grey coloured
    # figures that we need to reposition/flip
    fig_row_positions, fig_col_positions = np.nonzero(x == 5)

    # Check if the blocks are horizontally or vertically placed on the grid
    # We use position data of red blocks for this purpose

    # Check if figures are positioned horizontally by checking first 3 red blocks
    if (row_positions[0] == row_positions[1] == row_positions[2]) and (np.all(np.diff(col_positions[:3]) == 1)):
        # Handle figure in top part
        top_line_row = row_positions[0]  # Row value for the top line
        top_figure_start = np.min(fig_row_positions)  # Start row position for figure
        top_figure = x[top_figure_start:top_line_row - 1, :]  # Get the figure data
        top_figure_flipped = np.flip(top_figure, 0)  # Flip the figure
        top_figure_new_start = top_line_row + 2  # New starting row position for figure
        top_figure_new_end = top_figure_new_start + top_figure.shape[0]  # New ending row position for figure
        x[top_figure_new_start:top_figure_new_end, :] = top_figure_flipped  # Set values in new blocks as flipped figure
        x[:top_line_row - 1, :] = 0  # Set initial blocks of figure to black

        # Handle figure in bottom part
        bottom_line_row = row_positions[-1]  # Row value for the bottom line
        bottom_figure_end = np.max(fig_row_positions)  # End row position for figure
        bottom_figure = x[bottom_line_row + 2:bottom_figure_end + 1, :]  # Get the figure data
        bottom_figure_flipped = np.flip(bottom_figure, 0)  # Flip the figure
        bottom_figure_new_end = bottom_line_row - 1  # New ending row position for figure
        bottom_figure_new_start = bottom_figure_new_end - bottom_figure.shape[0]  # New starting row position for figure
        x[bottom_figure_new_start:bottom_figure_new_end, :] = bottom_figure_flipped  # Set values in new blocks
        x[bottom_line_row + 2:, :] = 0  # Set initial blocks of figure to black
    else:
        # Handle figure on left side
        left_line_col = col_positions[0]  # Column value for the left line
        left_figure_start = np.min(fig_col_positions)  # Start column position for figure
        left_figure = x[:, left_figure_start:left_line_col - 1]  # Get the figure data
        left_figure_flipped = np.flip(left_figure, 1)  # Flip the figure
        left_figure_new_start = left_line_col + 2  # New starting column position for figure
        left_figure_new_end = left_figure_new_start + left_figure.shape[1]  # New ending column position for figure
        x[:, left_figure_new_start:left_figure_new_end] = left_figure_flipped  # Set values in blocks as flipped figure
        x[:, :left_line_col - 1] = 0  # Set initial blocks of figure to black

        # Handle figure on right side
        right_line_col = col_positions[-1]  # Column value for the right line
        right_figure_end = np.max(fig_col_positions)  # End column position for figure
        right_figure = x[:, right_line_col + 2:right_figure_end + 1]  # Get the figure data
        right_figure_flipped = np.flip(right_figure, 1)  # Flip the figure
        right_figure_new_end = right_line_col - 1  # New ending column position for figure
        right_figure_new_start = right_figure_new_end - right_figure.shape[1]  # New starting column position for figure
        x[:, right_figure_new_start:right_figure_new_end] = right_figure_flipped  # Set values in new blocks
        x[:, right_line_col + 2:] = 0  # Set initial blocks of figure to black
    return x

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

