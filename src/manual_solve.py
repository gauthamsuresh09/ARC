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

"""
Student Names: Gautham Vadakkekara Suresh, Sharath Umesh Prabhu
Student IDs: 20231652, 20231826

GitHub repository : https://github.com/gauthamsuresh09/ARC/

References:
- Numpy documentation : https://numpy.org/doc/stable/
- Stackoverflow (for operations using various Numpy methods) : https://stackoverflow.com/

----------------------------------------------------------------

Summary:
---------

Libraries:
Since all the tasks involve Numpy arrays (or matrices), the main library have used for solving the
tasks is Numpy. Various Numpy methods are used appropriately for sub-tasks such as matching patterns,
fetching indices, modifying values, finding unique values, flipping the array, among others. Array 
slicing is used in almost all the tasks for selecting a portion of the array. Apart from this, all
other methods we have used are standard Python library methods.

Commonalities:
1) Almost all the tasks need to distinguish the patterns from the background. Most of the times, 
   background is black though not for all tasks.
2) The tasks have common intents such as flipping, counting, copying/repeating patterns, fill missing blocks,
   matching and expanding/contracting patterns. This relates to the geometrical and topological concepts of
   ARC. So, we can group the tasks based on their intent. In practice, if we can build a system that can learn 
   and generalise these steps, it can perform better on the ARC tasks. The system will then be more intelligent
   compared to the ones that cannot generalise.
3) For most of the tasks, we have to find locations of specific patterns such as squares, diagonals, lines, etc.
4) We dealt with some tasks for which there was symmetrical patterns in the array.

Differences:
1) Not all the tasks have input as square grid/matrix.
2) The required solution is not always of the same dimensions as the input.
3) Different groups of tasks need different methods for solving. Some need flipping, and some others may need
   expanding specific patterns, etc. But even for tasks that have flipping, the initial method for finding
   pattern, the way of flipping, and then final relative positions where output is placed are different. This
   can be extended to other groups as well.
"""


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
        The colour of black connecting blocks in the boundary should also change to yellow

    How this function works:
        First, it finds length of squares in the grid by starting from top-left corner
        and going diagonally till we find a blue box in any of the blocks on top or left
        side of current block. We use this to calculate total number of squares. Then we
        go through each square row-wise. We find the boundaries around the square and
        make sure the boundaries are within lower and upper bounds. Then, each boundary
        is checked to see if there is a block (black colour) connecting this square to
        another one. If its there, the square is coloured with yellow, otherwise as
        green. Finally, after going through all squares, we set the remaining black blocks
        in boundary to yellow as well.

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

    How this function works:
        We Start by going through each element in the array, row-wise.
        Once an element with green colour is found, its checked to see
        if the larger square containing the pattern is already visited.
        If not, size of smaller squares in the figure/pattern is found
        by starting from current element and moving right till another
        non-green element is found. The number of moves taken is then
        used to calculate the following :
        - Size and position of small two squares in pattern
        - Size and position of large square containing the pattern
        - Size and position of the blocks to be changed to blue colour
        These also depend on whether we find a green diagonal in large
        square as principal one or secondary one. Finally, we modify the
        required blocks to blue colour.

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

                    # Now, set the squaoperationsre on bottom right to blue
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


def solve_dc0a314f(x):
    """
    Solves task dc0a314f

    Description:
        Given an array, we must find what values should be put into a square with green colours(3). The whole array
        seems to be symmetric and the values taken up by the green points should be replaced by the other elements
        present in the array.

    How this function works:
        This task entails filling in the values for the missing pieces (green colours). We found that for finding the
        solution,the array should be created as a symmetric one and then the missing pieces should be gleaned from that.
        To create such a symmetric array, we divide the array into four halves and chose the lower most left part
        of the array. After this,we flipped this part vertically to obtain the lower right half and joined the two halves
        together (left and right). When we flip this horizontally,we get the upper part of the array and from there
        we can create the symmetric array.On comparison of this created array with the original array,we can easily find
        the solution.

    Correctness:
        All the given cases are solved.

    Arguments:
        x : Input Numpy array of dimension 2 and equal shape
            values for both axes

    Returns:
        A copy of x with required transformations applied
    """

    dim1, dim2 = x.shape  # get the dimensions(rows,columns respectively) from the array
    half = int(dim1 / 2)  # calculate the middle for the rows
    x_ = x.copy()  # make a copy of x
    result = np.argwhere(x_ == 3)  # find the locations of the green squares
    x3 = x_[half:dim2, 0:half]  # get the lower most left part of the array
    x4 = np.flip(x3, 1)  # flip the sliced array from above vertically to get the right half
    x2 = np.concatenate([x3, x4],
                        axis=1)  # concatenate the arrays together to form the lower half of the original array
    x1 = np.flip(x2, 0)  # flip the lower half horizontally to get the upper half
    x__ = np.concatenate([x1, x2], axis=0)  # put the lower half and upper half together to create the array
    # slice the created array by the location of the green squares
    x5 = x__[result[0][0]:result[-1][0] + 1, result[0][1]:result[-1][1] + 1]
    return x5  # return the solution


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

    How this function works:
        First, we start by finding positions (indices) of the red coloured (value 2)
        enclosing bracket shaped patterns. Along with this, the positions
        of grey coloured (value 5) figures are also found. After this, the positions
        of red blocks are checked to see if the bracket shape is placed horizontally or
        vertically on the grid. This is done by checking for consecutive red blocks
        forming a line. Based on the orientation, we then find the current blocks
        of figure that are present either top/bottom or left/right of the bracket
        shapes. These are determined using the positions of bracket shapes and their
        orientation. Now the figure is flipped using Numpy operation. We then move
        the figure to inside of the bracket shapes.

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


def solve_af902bf9(x):
    """
    Solves task af902bf9

    Description:
        There is an array given with multiple non-zero corner points of a square. The objective is to find
        these squares and fill all the elements with red colour (2) except the rows and columns containing
        the corner of the squares.

    How this function works:
        This task involves identifying squares and filling in the center with red colour (2).First, we get the locations
        of non-zero values in the array. Then, we slice the squares from the array by the locations.The next step
        is to fill in the red colour for appropriate rows (all rows except rows containing corner points of the square).
        Finally, we revert some elements of the square to black/0 (because columns containing corner points of the square
        should not have red elements).

    Correctness:
        All the given cases are solved.

    Arguments:
        x : Input Numpy array of dimension 2 and equal shape
            values for both axes

    Returns:
        A copy of x with required transformations applied
    """
    x_ = x.copy()  # make a copy of x
    result = np.argwhere(x_ > 0)  # find the locations of the corner point of the squares
    i = 0
    while i < len(result):
        corner1 = result[i]  # get the left first corner on top of the square
        corner4 = result[i + 3]  # get the right most corner on bottom of the square
        i += 4  # increment for the next square
        x_square = x_[corner1[0]:corner4[0] + 1, corner1[1]:corner4[1] + 1]  # slice the square from the array
        dim1, dim2 = x_square.shape  # get the dimensions(rows,columns respectively) from the array
        x_square[1:dim1 - 1] = 2  # set the rows to 2 for all rows except the ones with corner point
        x_square[1:dim1 - 1, 0] = 0  # set the first columns to 0 for all rows except the ones with corner point
        x_square[1:dim1 - 1, dim2 - 1] = 0  # set the last columns to 0 for all rows except the ones with corner point
        x_[corner1[0]:corner4[0] + 1,
        corner1[1]:corner4[1] + 1] = x_square  # set the created square to the original array
    return x_


def solve_0a938d79(x):
    """
    Solves task 0a938d79
    
    Description:
        The initial stage of the task contains two non-zero elements in the given array.The problem to solve
        here is then identifying the rows or columns they are situated in and expanding it to the whole row/columns
        depending on the dimensions of the array.The same must be done to equidistant rows/columns starting
        from the non-zero elements to the end of the array. The aforementioned distance is the space
        between the non-zero elements.

    How this function works:
        This task involves filling in elements row wise or column wise depending on the dimensions of the array
        and the non-zero elements. First, we check which dimension(row or column) is larger.Then,the non-zero elements
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
    dim1, dim2 = x.shape  # get the dimensions(rows,columns respectively) from the array
    x_ = x.copy()  # make a copy of x
    result = np.argwhere(x_ > 0)  # find the locations of the non-zero points
    values = x_[x_>0]  # get the values of the non-zero points
    if dim1 > dim2:  # check if rows are greater than columns
        index0 = result[0][0]  # get the row index of the first non-zero point
        index1 = result[1][0]  # get the row index of the second non-zero point
        intervals = index1-index0  # find the distance between them
        i = index0  # start from index0
        j = index1  # start from index1
        while i < dim1:
            x_[i] = values[0]  # assign the first non-zero values to row i
            i+=2*intervals  # increment by twice the interval size
        while j < dim1:
            x_[j] = values[1]  # assign the second non-zero values to row j
            j +=2*intervals  # increment by twice the interval size
    else:
        index0 = result[0][1]  # get the column index of the first non-zero point
        index1 = result[1][1]  # get the column index of the second non-zero point
        intervals = index1-index0  # find the distance between them
        i = index0  # start from index0
        j = index1  # start from index1
        while i < dim2:
            x_[:,i] = values[0]  # assign the first non-zero values to column i
            i +=2*intervals  # increment by twice the interval size
        while j < dim2:
            x_[:,j] = values[1]  # assign the second non-zero values to column j
            j +=2*intervals  # increment by twice the interval size
    return x_  # return solution


def solve_de1cd16c(x):
    """
    Solves task de1cd16c

    Description:
        There is an array given with multiple values as elements. Some of these have quadrilateral shaped
        areas in the array where they are present in the majority. The one in minority is one of these other
        values. This pattern(a colour dominating other in a quadrilateral) is repeated with other values too.
        The task here is to find the colour/value of the area where there are most number of these 'minority'
        values in comparison to others.

    How this function works:
        This task is about counting isolated points in a cluster of different colour and returning the cluster
        colour/value with the most number of these points. we start with taking the unique elements in the array.
        Next, for every unique element, we choose an initial point and search for the immediate neighbour. If the
        element does not have one in the array, then we mark it as the element to be counted. After doing this,
        we count this element in the clusters of the other elements and find the maximum count. The cluster with the
        maximum count is the solution.

    Correctness:
        All the given cases are solved.

    Arguments:
        x : Input Numpy array of dimension 2 and equal shape
            values for both axes

    Returns:
        A copy of x with required transformations applied
    """
    x_ = x.copy()  # make a copy of x
    elements = np.unique(x_)  # find all unique elements in the array
    find_element = -1  # assume the element that is to counted is -1
    for element in elements:
        locations = np.argwhere(x_ == element)  # find locations of the given element
        x0, y0 = locations[0]  # get the first location where the element is found
        x1, y1 = locations[1]  # get the second location where the element is found
        if abs(x0 - x1) == 1 or abs(y0 - y1) == 1:  # check if the two elements are close to each other
            continue  # if yes,continue
        else:
            find_element = element  # if no, we have found the element that is to be counted
            break  # search is completed,break the loop
    maxcount = 0  # start with initial count as 0
    x__ = []  # start with initial array as empty
    for element in elements:
        if element == find_element:  # if current element is the element to be counted,move to the next element
            continue
        else:
            locations = np.argwhere(x_ == element)  # find locations of the current element in the array
            # slice the original array so as to get a array with current element in majority
            element_area = x_[locations[0][0]:locations[-1][0] + 1, locations[0][1]:locations[-1][1] + 1]
            count = len(element_area[element_area == find_element])  # count the points where the find_element occurs
            if count > maxcount:
                maxcount = count  # update the maxcount to the largest found count
                x__ = np.array(element).reshape(1, 1)  # reshape the current element as per requirement
    return x__  # return the solution


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

    How this function works:
        First we find all the blue blocks, their count and indices in array.
        Then, all the blue blocks are changed to black. Then, the upper half
        triangle of array along principal diagonal is found. This is filtered
        row-wise to select only "blue block count" number of blocks. Finally,
        the selected blocks are coloured as red.

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


def solve_68b16354(x):
    """
    Solves the task 68b16354
   
    Description:
        There is an array given with multiple non-zero values. The task here is to create the array which
        closely resembles the input array.

    How this function works:
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
    x_ = x.copy()  # make a copy of x
    x_ = np.flip(x_,0)  # flip the array horizontally
    return x_  # return the solution


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

