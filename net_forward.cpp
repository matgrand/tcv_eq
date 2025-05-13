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

#include <torch/script.h> // For torch::jit::load, torch::jit::script::Module
#include <torch/torch.h>  // For torch::Tensor, torch::randn, etc.

#include <iostream>
#include <vector>
#include <string>
#include <stdexcept> // For std::runtime_error, std::exception
#include <memory>    // For std::unique_ptr
#include <chrono>    // For timing
#include <numeric>   // For std::accumulate
#include <cmath>     // For std::sqrt
#include <iomanip>   // For std::fixed, std::setprecision
#include <algorithm> // For std::min_element, std::max_element

const int NGR = 16, NGZ = 16, NIN = 95; // grid sizes and input size, check utils.py


/**
 * @brief Performs a forward pass through a PyTorch TorchScript model.
 *
 * Loads the model on the first call and keeps it in static memory for subsequent calls.
 * The model is set to evaluation mode.
 *
 * @param input1 First input tensor (e.g., [1, 95]).
 * @param input2 Second input tensor (e.g., [1, 16]).
 * @param input3 Third input tensor (e.g., [1, 16]).
 * @param model_path Path to the TorchScript model file (e.g., "net.pt").
 *                   This is only used during the first call to load the model.
 * @return torch::Tensor The output tensor from the model.
 * @throws std::exception if model loading or inference fails.
 */
torch::Tensor forward_pass(
    const torch::Tensor& input1,
    const torch::Tensor& input2,
    const torch::Tensor& input3,
    const std::string& model_path = "net.pt")
{
    // Static storage for the module to ensure it's loaded only once.
    // std::unique_ptr ensures proper cleanup if an exception occurs during construction.
    static std::unique_ptr<torch::jit::script::Module> module_ptr;
    static bool module_initialized = false; // Flag to track initialization

    // Load the module on the first call
    if (!module_initialized) {
        try {
            std::cout << "Loading model from: " << model_path << "..." << std::endl;
            module_ptr = std::make_unique<torch::jit::script::Module>(torch::jit::load(model_path));
            module_ptr->eval(); // Set to evaluation mode (important for dropout, batchnorm, etc.)
            
            // If using a CPU-only LibTorch build, the model will be on CPU.
            // If LibTorch has CUDA support and model might be on GPU, uncomment:
            // module_ptr->to(torch::kCPU); 

            module_initialized = true; // Set flag after successful loading and setup
            std::cout << "Model '" << model_path << "' loaded and initialized for inference." << std::endl;
        } catch (const c10::Error& e) {
            // c10::Error is LibTorch's base exception
            std::cerr << "LibTorch Error loading the model '" << model_path << "':\n" << e.what() << std::endl;
            throw; // Re-throw to be caught by the caller
        } catch (const std::exception& e) {
            std::cerr << "Standard C++ Exception loading the model '" << model_path << "':\n" << e.what() << std::endl;
            throw; // Re-throw
        }
    }

    
    // This check ensures that if initialization was attempted but failed in a way
    // that didn't set module_initialized but also didn't throw (unlikely with current setup),
    // or if module_initialized was true but module_ptr is somehow null.
    if (!module_ptr) {
        throw std::runtime_error("Model is not loaded or not available. Initialization might have failed.");
    }
    
    // Package inputs into a vector of IValue (LibTorch's generic value type)
    std::vector<torch::jit::IValue> inputs_ivalue;
    inputs_ivalue.push_back(input1);
    inputs_ivalue.push_back(input2);
    inputs_ivalue.push_back(input3);
    
    torch::Tensor output_tensor;
    {
        // Disable gradient calculations during inference for speed and to save memory.
        torch::NoGradGuard no_grad;
        try {
            torch::jit::IValue output_ivalue = module_ptr->forward(inputs_ivalue);
            
            if (!output_ivalue.isTensor()) {
                throw std::runtime_error("Model output is not a tensor. Actual type: " + output_ivalue.tagKind());
            }
            output_tensor = output_ivalue.toTensor();
            
            // If the model could be on GPU and you need CPU output:
            // output_tensor = output_tensor.cpu();
            
        } catch (const c10::Error& e) {
            std::cerr << "LibTorch Error during model inference:\n" << e.what() << std::endl;
            throw;
        } catch (const std::exception& e) {
            std::cerr << "Standard C++ Exception during model inference:\n" << e.what() << std::endl;
            throw;
        }
    }
    std::cout << "Input sizes: " << input1.sizes() << ", " << input2.sizes() << ", " << input3.sizes() << std::endl;
    std::cout << "Output size: " << output_tensor.sizes() << std::endl;
    return output_tensor;
}


// Main MEX function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    // Seed random number generator once when the MEX file is loaded/first run.
    // This makes sequences somewhat different across MATLAB sessions or MEX reloads.
    static bool random_seeded = false;
    if (!random_seeded) {
        srand((unsigned int)time(NULL));
        random_seeded = true;
    }

    mexPrintf("--- net_forward.cpp (Simple MEX Version) ---\n");
    
    int n = 5; // number of values to print

    //// Prepare inputs
    // // x (assumed length 95)
    // double* x_ptr = mxGetPr(prhs[0]); 
    // torch::Tensor x = torch::from_blob(x_ptr, {1, NIN}, torch::kDouble);
    // x = x.to(torch::kFloat); // Convert to float tensor
    
    // // r (assumed length 16)
    // double* r_ptr = mxGetPr(prhs[1]);
    // torch::Tensor r = torch::from_blob(r_ptr, {1, NGR}, torch::kDouble);
    // r = r.to(torch::kFloat); // Convert to float tensor
    
    // // z (assumed length 16)
    // double* z_ptr = mxGetPr(prhs[2]);
    // torch::Tensor z = torch::from_blob(z_ptr, {1, NGZ}, torch::kDouble);
    // z = z.to(torch::kFloat); // Convert to float tensor
    
    // try with random values:
    // Define tensor options (float type, default device which is CPU for CPU-only LibTorch)
    auto tensor_opts = torch::TensorOptions().dtype(torch::kFloat);
    torch::Tensor x = torch::randn({1, 95}, tensor_opts);
    torch::Tensor r = torch::randn({1, 16}, tensor_opts);
    torch::Tensor z = torch::randn({1, 16}, tensor_opts);
    
    // // print x values
    // std::cout << "x -> [ ";
    // for (int i = 0; i < n; ++i) {
    //     std::cout << std::showpos << std::fixed << std::setprecision(4) << x[0][i].item<float>() << " ";
    // }
    // std::cout << "]\n";
    // // print r values
    // std::cout << "r -> [ ";
    // for (int i = 0; i < n; ++i) {
    //     std::cout << std::showpos << std::fixed << std::setprecision(4) << r[0][i].item<float>() << " ";
    // }
    // std::cout << "]\n";
    // // print z values
    // std::cout << "z -> [ ";
    // for (int i = 0; i < n; ++i) {
    //     std::cout << std::showpos << std::fixed << std::setprecision(4) << z[0][i].item<float>() << " ";
    // }
    // std::cout << "]\n";
    
    
    
    // print x, r, z shapes
    std::cout << "Shapes -> x: " << x.sizes() << ", r: " << r.sizes() << ", z: " << z.sizes() << "\n";

    // --- Perform Forward Pass ---
    // Call the forward_pass function with the inputs
    try {
        torch::Tensor y = forward_pass(x, r, z);
    } catch (const std::exception& e) {
        mexErrMsgIdAndTxt("MATLAB:net_forward:forward_pass", "Error during forward pass: %s", e.what());
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
    // MATLAB stores arrays in column-major order.
    // For a 4D array A(h,w,c,b), element A(r,c,0,0) (0-indexed) is at
    // linear index = r + c*H (where H is the number of rows, 16 here).
    const mwSize H_out = output_dims[0]; // Number of rows = 16

    // for (int r = 0; r < 4; ++r) { // Print first 4 rows
    //     mexPrintf("  [ ");
    //     for (int c = 0; c < 4; ++c) { // Print first 4 columns
    //         // Calculate linear index for element (r,c) in the first 2D slice (channel 0, batch 0)
    //         int linear_index = r + c * H_out;
    //         mexPrintf("%6.4f ", output_data[linear_index]);
    //     }
    //     mexPrintf("]\n");
    // }

    mexPrintf("\n--- end of net_forward.cpp ---\n");
}