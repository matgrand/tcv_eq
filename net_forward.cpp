// net_forward.cpp
// Ultra-simple MEX function: reads 3 inputs, prints parts, creates random output, prints parts.
// ASSUMPTIONS:
// 1. Exactly 3 input arguments are provided from MATLAB.
// 2. All inputs are MATLAB double arrays.
// 3. Input 1 has at least 10 elements (assumed total 95).
// 4. Input 2 has at least 10 elements (assumed total 16).
// 5. Input 3 has at least 10 elements (assumed total 16).
// 6. One output argument is expected.
// NO LibTorch, NO significant error checking.

#include "mex.h"    // MATLAB MEX API
#include <cstdio>   // For formatting strings for mexPrintf if needed (not directly used here)
#include <cstdlib>  // For rand(), srand()
#include <ctime>    // For time() to seed srand()

// Main MEX function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    // Seed random number generator once when the MEX file is loaded/first run.
    // This makes sequences somewhat different across MATLAB sessions or MEX reloads.
    static bool random_seeded = false;
    if (!random_seeded) {
        srand((unsigned int)time(NULL));
        random_seeded = true;
    }

    mexPrintf("--- net_forward.cpp (Ultra-Simple MEX Version) ---\n");

    // --- Access and Print Inputs (TRUSTING they exist and are valid doubles) ---

    // Input 1 (assumed length 95, printing first 10)
    mexPrintf("\nInput 1 (first 10 elements):\n");
    double* input1_data = mxGetPr(prhs[0]); // Directly access, assuming prhs[0] is valid
    for (int i = 0; i < 10; ++i) {
        mexPrintf("  in1[%d] = %f\n", i, input1_data[i]);
    }

    // Input 2 (assumed length 16, printing first 10)
    mexPrintf("\nInput 2 (first 10 elements):\n");
    double* input2_data = mxGetPr(prhs[1]); // Directly access
    for (int i = 0; i < 10; ++i) {
        mexPrintf("  in2[%d] = %f\n", i, input2_data[i]);
    }

    // Input 3 (assumed length 16, printing first 10)
    mexPrintf("\nInput 3 (first 10 elements):\n");
    double* input3_data = mxGetPr(prhs[2]); // Directly access
    for (int i = 0; i < 10; ++i) {
        mexPrintf("  in3[%d] = %f\n", i, input3_data[i]);
    }

    // --- Create and Initialize Output ---
    // Output target: MATLAB array representing a [1,1,16,16] tensor
    // MATLAB dimensions: H, W, C, B  =>  16, 16, 1, 1
    const mwSize output_dims[] = {16, 16, 1, 1};
    const int num_output_dims = 4;

    // Create a MATLAB numeric array (double, real) for the output
    plhs[0] = mxCreateNumericArray(num_output_dims, output_dims, mxDOUBLE_CLASS, mxREAL);
    double* output_data = mxGetPr(plhs[0]); // Get pointer to the output data buffer

    // Fill output with random double values between 0.0 and 1.0
    int total_output_elements = 16 * 16 * 1 * 1; // 256 elements
    for (int i = 0; i < total_output_elements; ++i) {
        output_data[i] = (double)rand() / RAND_MAX;
    }

    // --- Print Part of the Output ---
    mexPrintf("\nOutput (randomly initialized, first 4x4 elements of the first slice):\n");
    // MATLAB stores arrays in column-major order.
    // For a 4D array A(h,w,c,b), element A(r,c,0,0) (0-indexed) is at
    // linear index = r + c*H (where H is the number of rows, 16 here).
    const mwSize H_out = output_dims[0]; // Number of rows = 16

    for (int r = 0; r < 4; ++r) { // Print first 4 rows
        mexPrintf("  [ ");
        for (int c = 0; c < 4; ++c) { // Print first 4 columns
            // Calculate linear index for element (r,c) in the first 2D slice (channel 0, batch 0)
            int linear_index = r + c * H_out;
            mexPrintf("%6.4f ", output_data[linear_index]);
        }
        mexPrintf("]\n");
    }

    mexPrintf("\n--- End of Ultra-Simple MEX Version ---\n");
}