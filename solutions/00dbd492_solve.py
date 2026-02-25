
import json
import numpy as np

def solve(input_grid):
    grid = np.array(input_grid)
    output_grid = grid.copy()
    rows, cols = grid.shape
    
    # We need to find rectangular boxes of 2s.
    # A box is defined by its top-left (r1, c1) and bottom-right (r2, c2)
    # where all boundary pixels are 2.
    
    # Let's iterate through all possible top-left and bottom-right corners
    # to find boxes where the boundary is all 2s.
    # To avoid finding sub-boxes, we should look for "maximal" boxes or 
    # specific structure.
    
    visited_cells = set()
    
    # Better approach: find all pixels with 2, and for each, see if it could be 
    # the top-left corner of a box.
    for r1 in range(rows):
        for c1 in range(cols):
            if grid[r1, c1] == 2:
                # Try all possible bottom-right corners
                for r2 in range(r1 + 2, rows):
                    for c2 in range(c1 + 2, cols):
                        if grid[r2, c2] == 2:
                            # Check if the boundary (r1, c1) to (r2, c2) is all 2s
                            is_box = True
                            # Top and bottom edges
                            for c in range(c1, c2 + 1):
                                if grid[r1, c] != 2 or grid[r2, c] != 2:
                                    is_box = False
                                    break
                            if not is_box: continue
                            # Left and right edges
                            for r in range(r1, r2 + 1):
                                if grid[r, c1] != 2 or grid[r, c2] != 2:
                                    is_box = False
                                    break
                            
                            if is_box:
                                # We found a box. Now check if it's a "base" box
                                # (it doesn't contain another box boundary, or it's the 
                                # specific pattern we are looking for: 
                                # rectangular outline of 2s).
                                
                                # Interior dimensions
                                h_int = r2 - r1 - 1
                                w_int = c2 - c1 - 1
                                min_dim = min(h_int, w_int)
                                
                                fill_color = None
                                if min_dim == 3:
                                    fill_color = 8
                                elif min_dim == 5:
                                    fill_color = 4
                                elif min_dim == 7:
                                    fill_color = 3
                                
                                if fill_color is not None:
                                    # Fill interior 0s
                                    for r_int in range(r1 + 1, r2):
                                        for c_int in range(c1 + 1, c2):
                                            if grid[r_int, c_int] == 0:
                                                output_grid[r_int, c_int] = fill_color

    return output_grid.tolist()

def main():
    with open('/Users/sybil/sybil-arc-challenge/data/data/training/00dbd492.json', 'r') as f:
        data = json.load(f)
    
    all_pass = True
    for i, example in enumerate(data['train']):
        input_grid = example['input']
        expected_output = example['output']
        actual_output = solve(input_grid)
        
        if actual_output == expected_output:
            print(f"Example {i}: PASS")
        else:
            print(f"Example {i}: FAIL")
            all_pass = False
            # Optional: print details
            # print("Expected:", expected_output)
            # print("Actual:", actual_output)
    
    if all_pass:
        print("PASS (all training correct)")
    else:
        print("FAIL")

if __name__ == "__main__":
    main()
