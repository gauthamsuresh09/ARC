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


def solve_794b24be(x):
    """
    Solves the task 794b24be

    Description:
        For this task, we are given a square grid (as Numpy array)
        with few blocks filled with blue colour. Our task is to take
        upper-right half triangle of the grid, fill k blocks with red
        colour by going row-wise from left, where k is the count of
        blocks with blue colour in the original grid. All the blue blocks
        in original grid are set to black as well.

    Correctness:
        All the given cases are solved.

    Arguments:
        x : Input Numpy array of dimension 2 and equal shape
            values for both axes
    Returns:
        A copy of x with required transformations applied
    """
    x = x.copy()  # Create copy of input array
    rows, cols = x.shape  # Get row and column count
    blue_blocks = x==1  # Get blue blocks
    blue_indices = np.nonzero(blue_blocks)  # Get indices of blue blocks
    # Get count of blue blocks
    # Ref : https://stackoverflow.com/questions/8364674/how-to-count-the-number-of-true-elements-in-a-numpy-bool-array
    blue_count = np.sum(blue_blocks)
    x[blue_indices] = 0  # Set blue blocks to black
    # Get first "blue_count" number of blocks in the
    # upper half triangle of the grid
    triu_row_indices, triu_col_indices = map(lambda x: x[:blue_count], np.triu_indices(rows))
    x[triu_row_indices, triu_col_indices] = 2  # Set the blocks to red
    return x


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


def set_to_bounds(i_s, i_e, j_s, j_e, low=0, high=1):
    """
    Makes sure the given index values stay with the
    bounds (low and high). This works only for 2 dimensional
    square matrices where low and high are same in both
    dimensions.

    Arguments:
        i_s : Starting value for row index
        i_e : Ending value for row index
        j_s : Starting value for column index
        j_e : Ending value for column index
        low : Minimum value for index possible
        high : Maximum value for index possible

    Returns:
        Values within bounds for given index positions
    """
    if i_s < low:
        i_s = low
    if i_e > high:
        i_e = high
    if j_s < low:
        j_s = low
    if j_e > high:
        j_e = high
    return i_s, i_e, j_s, j_e


def solve_22233c11(x):
    """
    Solves the task 22233c11

    Description:
        For this task, we are given a square grid with patterns
        having two green squares connected via either principal
        or secondary diagonals of the matrices. There can be multiple
        such patterns in the grid. For each such pattern identified,
        we have to add two blue squares of same size to their neighbourhood.
        Positions of these blue squares will depend on the following :
        1. Positions of the green squares
        2. Whether green squares are connected via principal or secondary
           diagonal
        The positions for blue squares can be determined once we identify
        the above two. Finally, these two squares are marked as blue.

    Correctness:
        All the given cases are solved.

    Arguments:
        x : Input Numpy array of dimension 2 and equal shape
            values for both axes
    Returns:
        A copy of x with required transformations applied
    """
    x = x.copy()  # Create copy of input array
    rows, cols = x.shape  # Get row and column count
    green = 3  # Value for green
    blue = 8  # Value for blue

    found = []  # For figures already visited
    # Loop through each block on the grid
    # We will always find the first block of top
    # part for the figure first. The algorithm makes
    # use of this to find the pattern and solve the task.
    for i in range(rows):
        for j in range(cols):
            if x[i][j] == green:
                # Check if this figure/square was already visited
                done = any([all([i >= f[0], i <= f[1], j >= f[2], j <= f[3]]) for f in found])
                if done:
                    continue  # Go to next block as this square was already visited
                # Get size of smaller squares in the figure
                # For this we start from current cell and we move right till
                # we find a cell that's not green
                # This will give count as n-1 for smaller square of size n
                # So if size of smaller square is 2, we get 1 as c
                c = 0  # Initialise count as 0
                while (j + c + 1) < cols and x[i][j + c + 1] == green:
                    c += 1
                # Set starting column and ending column values for smaller square
                start_pos = j
                end_pos = j + c

                # Now there are two ways the pattern can come
                # Either the squares are connected via principal diagonal or
                # secondary diagonal

                # Check if its connected via secondary diagonal
                # For this, lets set the positions of larger
                # square (parent matrix) which contains the two small squares
                s_i = i  # Starting row for the square
                e_i = i + 2 * (c + 1)  # Ending row for the square
                s_j = j - c - 1  # Starting column + 1 for the square
                e_j = end_pos + 1  # Ending column + 1 for the square
                parent_matrix = x[s_i:e_i, s_j:e_j]  # Get parent matrix
                parent_matrix_diag = np.fliplr(parent_matrix).diagonal()  # Get secondary diagonal
                if np.all(parent_matrix_diag == green):  # Check if all blocks in diagonal are green
                    found.append([s_i, e_i - 1, s_j, e_j - 1])  # Mark the visit to this parent square
                    # Lets set the square on top left first to blue
                    # Get index positions for the square
                    top_left_s_i = s_i - (1 + c)
                    top_left_e_i = s_i - 1
                    top_left_s_j = s_j - (c + 1)
                    top_left_e_j = s_j - 1
                    # Modify the index values so that they are within bounds
                    top_left_s_i, top_left_e_i, top_left_s_j, top_left_e_j = set_to_bounds(
                        top_left_s_i, top_left_e_i, top_left_s_j, top_left_e_j, low=0, high=rows-1)
                    # Mark the required blocks as blue
                    x[top_left_s_i:top_left_e_i + 1, top_left_s_j:top_left_e_j + 1] = blue

                    # Now, set the square on bottom right to blue
                    # Get index positions for the square
                    bottom_right_s_i = e_i
                    bottom_right_e_i = e_i + c
                    bottom_right_s_j = e_j
                    bottom_right_e_j = e_j + c
                    # Modify the index values so that they are within bounds
                    bottom_right_s_i, bottom_right_e_i, bottom_right_s_j, bottom_right_e_j = set_to_bounds(
                        bottom_right_s_i, bottom_right_e_i, bottom_right_s_j, bottom_right_e_j, low=0, high=rows-1)
                    # Mark the required blocks as blue
                    x[bottom_right_s_i:bottom_right_e_i + 1, bottom_right_s_j:bottom_right_e_j + 1] = blue
                    continue  # Go to next block

                # Check if they are connected via principal diagonal
                # Get index positions for parent matrix
                # s_i and e_i remains the same as before
                s_j = start_pos  # Starting column + 1 for the square
                e_j = j + 2 * (c + 1)  # Ending column + 1 for the square
                parent_matrix = x[s_i:e_i, s_j:e_j]  # Get parent matrix
                parent_matrix_diag = parent_matrix.diagonal()  # Get principal diagonal
                if np.all(parent_matrix_diag == 3):  # Check if all blocks in diagonal are green
                    found.append([s_i, e_i - 1, s_j, e_j - 1])  # Mark the visit to this parent square
                    # Lets set the square on top right first to blue
                    # Get index positions for the square
                    top_right_s_i = s_i - (c + 1)
                    top_right_e_i = s_i - 1
                    top_right_s_j = e_j
                    top_right_e_j = e_j + c
                    # Modify the index values so that they are within bounds
                    top_right_s_i, top_right_e_i, top_right_s_j, top_right_e_j = set_to_bounds(
                        top_right_s_i, top_right_e_i, top_right_s_j, top_right_e_j, low=0, high=rows-1)
                    # Mark the required blocks as blue
                    x[top_right_s_i:top_right_e_i + 1, top_right_s_j:top_right_e_j + 1] = blue

                    # Now, set the square on bottom left to blue
                    # Get index positions for the square
                    bottom_left_s_i = e_i
                    bottom_left_e_i = e_i + c
                    bottom_left_s_j = s_j - (c + 1)
                    bottom_left_e_j = s_j - 1
                    # Modify the index values so that they are within bounds
                    bottom_left_s_i, bottom_left_e_i, bottom_left_s_j, bottom_left_e_j = set_to_bounds(
                        bottom_left_s_i, bottom_left_e_i, bottom_left_s_j, bottom_left_e_j, low=0, high=rows-1)
                    # Mark the required blocks as blue
                    x[bottom_left_s_i:bottom_left_e_i + 1, bottom_left_s_j:bottom_left_e_j + 1] = blue
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

