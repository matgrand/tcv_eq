// net_forward.cpp
// MEX function to load a PyTorch model and perform inference.
// Inputs:
//   prhs[0]: double array (1x95 or 95x1)
//   prhs[1]: double array (1x16 or 16x1)
//   prhs[2]: double array (1x16 or 16x1)
// Output:
//   plhs[0]: double array (16x16x1x1) - representing [1,1,16,16] B,C,H,W tensor

#include "mex.h"

#include <torch/script.h> // LibTorch main header for TorchScript
#include <torch/torch.h>  // LibTorch tensor operations and utilities

#include <vector>
#include <string>
#include <stdexcept> // For standard exceptions
#include <memory>    // For std::unique_ptr

// Helper function to check dimensions and get data pointer
// Returns a torch::Tensor converted to float, handling potential reshape
torch::Tensor processInput(const mxArray* mxArr, const std::vector<int64_t>& expected_torch_shape, const char* input_name) {
    if (!mxIsDouble(mxArr) || mxIsComplex(mxArr)) {
        mexErrMsgIdAndTxt("NetForward:InputTypeError", "Input '%s' must be a real double array.", input_name);
    }

    mwSize ndims = mxGetNumberOfDimensions(mxArr);
    const mwSize* dims_mw = mxGetDimensions(mxArr);
    size_t numel = mxGetNumberOfElements(mxArr);

    size_t expected_numel = 1;
    for(int64_t dim : expected_torch_shape) {
        expected_numel *= dim;
    }

    if (numel != expected_numel) {
         char err_msg[200];
         snprintf(err_msg, sizeof(err_msg), "Input '%s' has %zu elements, expected %zu.", input_name, numel, expected_numel);
         mexErrMsgIdAndTxt("NetForward:InputSizeError", err_msg);
    }

    // Determine if MATLAB provided a row vector (1xN) or column vector (Nx1)
    // LibTorch expects [Batch, Features], so usually [1, N]
    std::vector<int64_t> current_shape;
    bool needs_reshape = false;
    if (ndims == 2 && dims_mw[0] == 1 && dims_mw[1] == expected_torch_shape[1]) {
        // Input is 1xN (row vector), matches expected [1, N]
        current_shape = {1, (int64_t)dims_mw[1]};
    } else if (ndims == 2 && dims_mw[0] == expected_torch_shape[1] && dims_mw[1] == 1) {
        // Input is Nx1 (column vector), needs reshape to [1, N]
        current_shape = {(int64_t)dims_mw[0], 1};
        needs_reshape = true;
    } else {
        // Construct string representation of dimensions for error message
        std::string dims_str;
        for(mwSize i=0; i<ndims; ++i) {
            dims_str += std::to_string(dims_mw[i]) + (i == ndims - 1 ? "" : "x");
        }
        char err_msg[200];
        snprintf(err_msg, sizeof(err_msg), "Input '%s' has dimensions [%s], expected 1x%lld or %lldx1.",
                 input_name, dims_str.c_str(), expected_torch_shape[1], expected_torch_shape[1]);
        mexErrMsgIdAndTxt("NetForward:InputShapeError", err_msg);
    }

    // Get pointer to MATLAB data (double)
    double* data_ptr = mxGetPr(mxArr);

    // Create a tensor wrapper around the MATLAB data (no copy yet)
    // Specify the dimensions as they are in MATLAB
    auto options = torch::TensorOptions().dtype(torch::kDouble);
    torch::Tensor tensor_double = torch::from_blob(data_ptr, current_shape, options);

    // Reshape if necessary (e.g., Nx1 -> 1xN)
    if (needs_reshape) {
        tensor_double = tensor_double.reshape(expected_torch_shape);
    }

    // Clone and convert to float for the network
    // Cloning ensures we have our own copy if the original MATLAB array goes out of scope
    // or if from_blob doesn't own the memory (which it doesn't).
    return tensor_double.clone().to(torch::kFloat);
}


// Main MEX function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // --- Argument Validation ---
    if (nrhs != 3) {
        mexErrMsgIdAndTxt("NetForward:Usage", "Usage: output = net_forward(input1[1x95], input2[1x16], input3[1x16])");
    }
    if (nlhs > 1) {
        mexErrMsgIdAndTxt("NetForward:Usage", "Too many output arguments requested.");
    }

    // --- Load Model (TorchScript) ---
    static std::unique_ptr<torch::jit::script::Module> module_ptr = nullptr;
    if (!module_ptr) {
        try {
            // Load the traced model. Ensure 'net.pt' is accessible from where MATLAB runs.
            // You might need an absolute path or ensure it's on MATLAB's path.
            module_ptr = std::make_unique<torch::jit::script::Module>(torch::jit::load("net.pt"));
            // Set the model to evaluation mode (disables dropout, batch norm updates, etc.)
            module_ptr->eval();
             // Optional: Force to CPU if your LibTorch is CPU-only and model might be GPU
             // module_ptr->to(torch::kCPU);
            mexPrintf("Loaded PyTorch model 'net.pt' successfully.\n");
            // Lock the module in memory so it persists between calls
            mexMakeMemoryPersistent(module_ptr.get()); // Requires C++11 smart pointer support or raw pointer mgmt
        } catch (const c10::Error& e) {
            module_ptr = nullptr; // Ensure reset on failure
            std::string msg = "Error loading the model 'net.pt': ";
            msg += e.what();
            mexErrMsgIdAndTxt("NetForward:LoadError", msg.c_str());
        } catch (const std::exception& e) {
            module_ptr = nullptr;
             std::string msg = "Standard exception loading the model 'net.pt': ";
             msg += e.what();
             mexErrMsgIdAndTxt("NetForward:LoadError", msg.c_str());
        } catch (...) {
            module_ptr = nullptr;
            mexErrMsgIdAndTxt("NetForward:LoadError", "Unknown error loading the model 'net.pt'.");
        }
    }


    // --- Prepare Input Tensors ---
    std::vector<torch::jit::IValue> inputs;
    try {
        inputs.push_back(processInput(prhs[0], {1, 95}, "input1"));
        inputs.push_back(processInput(prhs[1], {1, 16}, "input2"));
        inputs.push_back(processInput(prhs[2], {1, 16}, "input3"));
    } catch (const std::exception& e) { // Catch errors from processInput
        // Error message already sent by processInput using mexErrMsgIdAndTxt
        return; // Exit mexFunction
    } catch (...) {
         mexErrMsgIdAndTxt("NetForward:InputError", "Unknown error processing inputs.");
    }


    // --- Perform Inference ---
    torch::Tensor output_tensor;
    try {
        // Disable gradient calculations during inference
        torch::NoGradGuard no_grad;

        // Run the forward pass
        torch::jit::IValue output_ivalue = module_ptr->forward(inputs);

        if (!output_ivalue.isTensor()) {
            mexErrMsgIdAndTxt("NetForward:OutputError", "Model output is not a tensor.");
        }
        output_tensor = output_ivalue.toTensor();

        // Optional: If model might be on GPU, move tensor to CPU
        // output_tensor = output_tensor.cpu();

    } catch (const c10::Error& e) {
        std::string msg = "Error during model inference: ";
        msg += e.what();
        mexErrMsgIdAndTxt("NetForward:InferenceError", msg.c_str());
    } catch (const std::exception& e) {
         std::string msg = "Standard exception during model inference: ";
         msg += e.what();
         mexErrMsgIdAndTxt("NetForward:InferenceError", msg.c_str());
    } catch (...) {
        mexErrMsgIdAndTxt("NetForward:InferenceError", "Unknown error during model inference.");
    }

    // --- Process Output Tensor ---
    try {
        // Expected LibTorch output shape: [B, C, H, W] = [1, 1, 16, 16]
        const std::vector<int64_t> expected_out_shape = {1, 1, 16, 16};
        auto actual_out_sizes = output_tensor.sizes();

        if (actual_out_sizes.vec() != expected_out_shape) {
            std::string actual_shape_str;
            for(size_t i=0; i<actual_out_sizes.size(); ++i) {
                 actual_shape_str += std::to_string(actual_out_sizes[i]) + (i == actual_out_sizes.size() - 1 ? "" : "x");
            }
             char err_msg[200];
             snprintf(err_msg, sizeof(err_msg), "Unexpected output tensor shape. Got [%s], expected [1x1x16x16].", actual_shape_str.c_str());
            mexErrMsgIdAndTxt("NetForward:OutputShapeError", err_msg);
        }

        // Convert output tensor to double for MATLAB
        output_tensor = output_tensor.to(torch::kDouble);

        // Ensure the tensor is contiguous in memory (row-major) for reliable data access
        // This might involve a copy if it's not already contiguous.
        output_tensor = output_tensor.contiguous();

        // Create the MATLAB output array
        // MATLAB dimension order: H, W, C, B -> {16, 16, 1, 1}
        const mwSize matlab_out_dims[] = {16, 16, 1, 1};
        plhs[0] = mxCreateNumericArray(4, matlab_out_dims, mxDOUBLE_CLASS, mxREAL);
        double* out_ptr = mxGetPr(plhs[0]); // Pointer to the output MATLAB array data

        // Get a pointer to the contiguous LibTorch tensor data
        const double* tensor_data_ptr = output_tensor.data_ptr<double>();

        // --- Copy data element-by-element with index transposition ---
        // LibTorch tensor access (row-major): B, C, H, W
        // MATLAB array access (column-major): H, W, C, B
        const int64_t B = expected_out_shape[0]; // 1
        const int64_t C = expected_out_shape[1]; // 1
        const int64_t H = expected_out_shape[2]; // 16
        const int64_t W = expected_out_shape[3]; // 16

        for (int64_t b = 0; b < B; ++b) {
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t w = 0; w < W; ++w) { // Iterate width before height for better cache locality with LibTorch's row-major
                    for (int64_t h = 0; h < H; ++h) {
                        // Calculate linear index for source (LibTorch, row-major)
                        int64_t torch_linear_idx = b * (C * H * W) + c * (H * W) + h * W + w;

                        // Calculate linear index for destination (MATLAB, column-major)
                        // H is fastest changing dim, B is slowest
                        int64_t matlab_linear_idx = h + w * H + c * (H * W) + b * (H * W * C);

                        // Copy the value
                        out_ptr[matlab_linear_idx] = tensor_data_ptr[torch_linear_idx];
                    }
                }
            }
        }
         // --- End data copy ---

    } catch (const c10::Error& e) {
        std::string msg = "Error processing output tensor: ";
        msg += e.what();
        mexErrMsgIdAndTxt("NetForward:OutputError", msg.c_str());
    } catch (const std::exception& e) {
         std::string msg = "Standard exception processing output tensor: ";
         msg += e.what();
         mexErrMsgIdAndTxt("NetForward:OutputError", msg.c_str());
    } catch (...) {
        mexErrMsgIdAndTxt("NetForward:OutputError", "Unknown error processing output tensor.");
    }
}