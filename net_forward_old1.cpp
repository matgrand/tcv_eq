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
#include "matrix.h" // For mxArray, mxGetPr, etc.
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
    if (!module_initialized || true) { // TODO: remove true
        std::cout << "Loading model from: " << model_path << "..." << std::endl;
        module_ptr = std::make_unique<torch::jit::script::Module>(torch::jit::load(model_path));
        module_ptr->eval(); // Set to evaluation mode (important for dropout, batchnorm, etc.)
        
        // If using a CPU-only LibTorch build, the model will be on CPU.
        // If LibTorch has CUDA support and model might be on GPU, uncomment:
        module_ptr->to(torch::kCPU); 

        module_initialized = true; // Set flag after successful loading and setup
        std::cout << "Model '" << model_path << "' loaded and initialized for inference." << std::endl;
    }
    
    // Package inputs into a vector of IValue (LibTorch's generic value type)
    // std::vector<torch::jit::IValue> inputs_ivalue = {input1, input2, input3};
    torch::Tensor output_tensor;
    {
        // Disable gradient calculations during inference for speed and to save memory.
        torch::NoGradGuard no_grad;
        torch::jit::IValue output_ivalue = module_ptr->forward({input1, input2, input3});
        
        if (!output_ivalue.isTensor()) {
            throw std::runtime_error("Model output is not a tensor. Actual type: " + output_ivalue.tagKind());
        }
        output_tensor = output_ivalue.toTensor();
    }
    std::cout << "Input sizes: " << input1.sizes() << ", " << input2.sizes() << ", " << input3.sizes() << std::endl;
    std::cout << "Output size: " << output_tensor.sizes() << std::endl;
    return output_tensor;
}


// Main MEX function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    mexPrintf("--- net_forward.cpp (Simple MEX Version) ---\n");
    
    int n = 5; // number of values to print

    // Prepare inputs
    // x (assumed length 95)
    double* x_ptr = mxGetPr(prhs[0]); 
    torch::Tensor x = torch::from_blob(x_ptr, {1, NIN}, torch::kDouble);
    // x = x.to(torch::kFloat); // Convert to float tensor
    x = x.to(torch::kDouble).contiguous(); // Ensure contiguous memory layout
    
    // r (assumed length 16)
    double* r_ptr = mxGetPr(prhs[1]);
    torch::Tensor r = torch::from_blob(r_ptr, {1, NGR}, torch::kDouble);
    // r = r.to(torch::kFloat); // Convert to float tensor
    r = r.to(torch::kDouble).contiguous(); // Ensure contiguous memory layout
    
    // z (assumed length 16)
    double* z_ptr = mxGetPr(prhs[2]);
    torch::Tensor z = torch::from_blob(z_ptr, {1, NGZ}, torch::kDouble);
    // z = z.to(torch::kFloat); // Convert to float tensor
    z = z.to(torch::kDouble).contiguous(); // Ensure contiguous memory layout

    
    // // with random values:
    // // Define tensor options (float type, default device which is CPU for CPU-only LibTorch)
    // torch::Tensor x = torch::randn({1, 95}, torch::kDouble);
    // torch::Tensor r = torch::randn({1, 16}, torch::kDouble);
    // torch::Tensor z = torch::randn({1, 16}, torch::kDouble);
    
    // print x values
    std::cout << "x -> [ ";
    for (int i = 0; i < n; ++i) {
        std::cout << std::showpos << std::fixed << std::setprecision(4) << x[0][i].item<float>() << " ";
    }
    std::cout << "]\n";
    // print r values
    std::cout << "r -> [ ";
    for (int i = 0; i < n; ++i) {
        std::cout << std::showpos << std::fixed << std::setprecision(4) << r[0][i].item<float>() << " ";
    }
    std::cout << "]\n";
    // print z values
    std::cout << "z -> [ ";
    for (int i = 0; i < n; ++i) {
        std::cout << std::showpos << std::fixed << std::setprecision(4) << z[0][i].item<float>() << " ";
    }
    std::cout << "]\n";
    
    // print x, r, z shapes
    std::cout << "Shapes -> x: " << x.sizes() << ", r: " << r.sizes() << ", z: " << z.sizes() << "\n";

    // --- Perform Forward Pass ---
    std::cout << "Performing forward pass..." << std::endl;
    torch::Tensor y = forward_pass(x, r, z);

    //reshape to [NGR, NGZ]
    y = y.view({NGR, NGZ});
    // make it contiguous
    y = y.contiguous();

    //print y shape
    std::cout << "y shape: " << y.sizes() << "\n";

    // print y values
    torch::Tensor yp = y.view({NGR * NGZ});
    std::cout << "y -> [ ";
    for (int i = 0; i < n; ++i) {
        std::cout << std::showpos << std::fixed << std::setprecision(4) << yp[i].item<float>() << " ";
    }
    std::cout << "]\n";

    // --- Create and Initialize Output ---

    // Create MATLAB output matrix (NGZ x NGR)
    plhs[0] = mxCreateDoubleMatrix(NGR, NGZ, mxREAL);
    double* y_ptr = mxGetPr(plhs[0]);

    // Copy data from the tensor to the MATLAB matrix
    for (int i = 0; i < NGR; ++i) {
        for (int j = 0; j < NGZ; ++j) {
            double val = y[i][j].item<double>();
            std::cout << val << " ";
            y_ptr[i*NGR+j] = val; // Copy to MATLAB output
        }
    }
    

    mexPrintf("\n--- end of net_forward.cpp ---\n");
}