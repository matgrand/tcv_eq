// net_forward.cpp
// MEX function to load a PyTorch model and perform inference with persistent memory.

#include "mex.h"    // MATLAB MEX API

// LibTorch headers
#include <torch/script.h> // For torch::jit::load, torch::jit::script::Module
#include <torch/torch.h>  // For torch::Tensor, torch::randn, etc.

// Standard C++ headers
#include <vector>
#include <string>
#include <memory>    // For std::unique_ptr
#include <stdexcept> // For std::runtime_error
#include <cstdio>    // For snprintf for error messages

// Define constants for input/output sizes (as in your provided code)
const int NGR = 16, NGZ = 16, NIN = 95;

// --- Global static variables for the PyTorch model and MEX state ---
// These persist as long as the MEX file is loaded in MATLAB's memory.
static std::unique_ptr<torch::jit::script::Module> G_module_ptr;
static bool G_module_initialized = false;
static bool G_mex_is_locked = false; // To ensure mexLock/mexAtExit are called only once

// --- Forward declaration for the cleanup function registered with mexAtExit ---
void cleanup_resources_at_exit();

// --- Helper function to convert MATLAB mxArray (double) to torch::Tensor (float) ---
torch::Tensor mxArray_to_torch_tensor(const mxArray* mx_arr, const std::vector<int64_t>& expected_torch_shape, const char* input_name) {
    // Basic type check
    if (!mxIsDouble(mx_arr) || mxIsComplex(mx_arr)) {
        char err_msg[256];
        snprintf(err_msg, sizeof(err_msg), "Input '%s' must be a real double array.", input_name);
        mexErrMsgIdAndTxt("NetForward:InputTypeError", err_msg);
    }

    // Element count check
    size_t numel_matlab = mxGetNumberOfElements(mx_arr);
    size_t expected_numel = 1;
    for(int64_t dim : expected_torch_shape) { expected_numel *= dim; }

    if (numel_matlab != expected_numel) {
        char err_msg[256];
        snprintf(err_msg, sizeof(err_msg), "Input '%s' has %zu elements, expected %zu (for PyTorch shape [",
                 input_name, numel_matlab, expected_numel);
        for(size_t i=0; i < expected_torch_shape.size(); ++i) {
            snprintf(err_msg + strlen(err_msg), sizeof(err_msg) - strlen(err_msg), "%lld%s",
                     expected_torch_shape[i], (i == expected_torch_shape.size() - 1 ? "" : "x"));
        }
        snprintf(err_msg + strlen(err_msg), sizeof(err_msg) - strlen(err_msg), "]).");
        mexErrMsgIdAndTxt("NetForward:InputSizeError", err_msg);
    }

    // Get MATLAB dimensions and data pointer
    mwSize num_dims_matlab = mxGetNumberOfDimensions(mx_arr);
    const mwSize* dims_matlab = mxGetDimensions(mx_arr);
    double* data_ptr_double = mxGetPr(mx_arr);

    torch::Tensor tensor_double;

    // Handle MATLAB 1xN (row vector) vs. Nx1 (column vector) for 2D tensors
    // This logic assumes expected_torch_shape is [1, Features]
    if (expected_torch_shape.size() == 2 && expected_torch_shape[0] == 1) {
        if (num_dims_matlab == 2 && dims_matlab[0] == 1 && (mwSize)expected_torch_shape[1] == dims_matlab[1]) {
            // MATLAB is 1xN (row vector), shape matches {1, Features}
            tensor_double = torch::from_blob(data_ptr_double, {1, (int64_t)dims_matlab[1]}, torch::kDouble);
        } else if (num_dims_matlab == 2 && dims_matlab[1] == 1 && (mwSize)expected_torch_shape[1] == dims_matlab[0]) {
            // MATLAB is Nx1 (column vector), needs reshape to {1, Features}
            tensor_double = torch::from_blob(data_ptr_double, {(int64_t)dims_matlab[0], 1}, torch::kDouble);
            tensor_double = tensor_double.reshape(expected_torch_shape); // Reshape to {1, Features}
        } else {
            // Shape mismatch
            char err_msg[512];
            std::string matlab_dims_str;
            for(mwSize i=0; i<num_dims_matlab; ++i) matlab_dims_str += std::to_string(dims_matlab[i]) + (i==num_dims_matlab-1 ? "" : "x");
            snprintf(err_msg, sizeof(err_msg), "Input '%s': MATLAB dims [%s] are incompatible with expected PyTorch shape [1, %lld] after handling row/column vectors.",
                     input_name, matlab_dims_str.c_str(), expected_torch_shape[1]);
            mexErrMsgIdAndTxt("NetForward:InputShapeError", err_msg);
        }
    } else {
        // For other shapes, a more general solution or specific error is needed.
        // This simple example focuses on the [1, Features] case.
        mexErrMsgIdAndTxt("NetForward:InputShapeError", "This example primarily handles 2D inputs of shape [1, N]. More complex shape handling needed.");
    }
    // Clone to take ownership of the data, then convert to float
    return tensor_double.clone().to(torch::kFloat);
}

// Helper function to convert IntArrayRef to a string representation like "[1, 2, 3]"
std::string intArrayRefToString(const torch::IntArrayRef& arr) {
    if (arr.empty()) {
        return "[]";
    }
    std::string s = "[";
    for (size_t i = 0; i < arr.size(); ++i) {
        s += std::to_string(arr[i]);
        if (i < arr.size() - 1) {
            s += ", ";
        }
    }
    s += "]";
    return s;
}

// --- Main MEX Function ---
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

    // --- Argument Count Validation ---
    if (nrhs != 3) {
        mexErrMsgIdAndTxt("NetForward:NumInputs", "Requires 3 input arguments.");
    }
    if (nlhs > 1) {
        mexErrMsgIdAndTxt("NetForward:NumOutputs", "Produces 1 output argument.");
    }

    // --- Initialize Model (on first call) & Manage MEX Persistence ---
    if (!G_module_initialized) {
        mexPrintf("net_forward: First call, initializing PyTorch model...\n");
        try {
            // Define the path to your TorchScript model
            // Ensure "net.pt" is on MATLAB's path or provide an absolute path.
            std::string model_path = "net.pt";
            G_module_ptr = std::make_unique<torch::jit::script::Module>(torch::jit::load(model_path));
            G_module_ptr->eval(); // Set the model to evaluation mode
            // Optional: Force model to CPU if it might be on GPU and LibTorch is CPU-only or you prefer CPU
            // G_module_ptr->to(torch::kCPU);

            G_module_initialized = true; // Mark as initialized
            mexPrintf("net_forward: Model '%s' loaded successfully.\n", model_path.c_str());

            // Lock the MEX file in memory and register cleanup function (only once)
            if (!G_mex_is_locked) {
                mexLock();
                mexAtExit(cleanup_resources_at_exit);
                G_mex_is_locked = true;
                mexPrintf("net_forward: MEX file locked and cleanup function registered.\n");
            }

        } catch (const c10::Error& e) { // LibTorch specific exceptions
            G_module_ptr.reset(); G_module_initialized = false; // Reset state on failure
            char err_msg[512];
            snprintf(err_msg, sizeof(err_msg), "LibTorch error loading model: %s", e.what());
            mexErrMsgIdAndTxt("NetForward:LoadErrorTorch", err_msg);
        } catch (const std::exception& e) { // Standard C++ exceptions
            G_module_ptr.reset(); G_module_initialized = false;
            char err_msg[512];
            snprintf(err_msg, sizeof(err_msg), "Standard C++ error loading model: %s", e.what());
            mexErrMsgIdAndTxt("NetForward:LoadErrorStd", err_msg);
        } catch (...) { // Any other exceptions
            G_module_ptr.reset(); G_module_initialized = false;
            mexErrMsgIdAndTxt("NetForward:LoadErrorUnknown", "Unknown error loading PyTorch model.");
        }
    }

    // Ensure model is ready after initialization attempt
    if (!G_module_ptr) {
        mexErrMsgIdAndTxt("NetForward:ModelNotReady", "PyTorch model is not loaded. Initialization might have failed on a previous call.");
    }

    // --- Prepare Input Tensors from MATLAB mxArrays ---
    torch::Tensor x_torch, r_torch, z_torch;
    try {
        x_torch = mxArray_to_torch_tensor(prhs[0], {1, NIN}, "input_x (features)");
        r_torch = mxArray_to_torch_tensor(prhs[1], {1, NGR}, "input_r (grid_r)");
        z_torch = mxArray_to_torch_tensor(prhs[2], {1, NGZ}, "input_z (grid_z)");
    } catch (...) {
        // mxArray_to_torch_tensor already called mexErrMsgIdAndTxt, so just return
        return;
    }

    // --- Perform Inference ---
    torch::Tensor output_tensor_torch;
    try {
        std::vector<torch::jit::IValue> inputs_ivalue;
        inputs_ivalue.push_back(x_torch);
        inputs_ivalue.push_back(r_torch);
        inputs_ivalue.push_back(z_torch);

        // Disable gradient calculations for inference
        torch::NoGradGuard no_grad;
        torch::jit::IValue output_ivalue = G_module_ptr->forward(inputs_ivalue);

        if (!output_ivalue.isTensor()) {
            mexErrMsgIdAndTxt("NetForward:OutputError", "Model output is not a tensor.");
        }
        output_tensor_torch = output_ivalue.toTensor();
        // output_tensor_torch = output_tensor_torch.cpu(); // Ensure on CPU if needed

    } catch (const c10::Error& e) {
        char err_msg[512];
        snprintf(err_msg, sizeof(err_msg), "LibTorch error during inference: %s", e.what());
        mexErrMsgIdAndTxt("NetForward:InferenceErrorTorch", err_msg);
    } catch (const std::exception& e) {
        char err_msg[512];
        snprintf(err_msg, sizeof(err_msg), "Standard C++ error during inference: %s", e.what());
        mexErrMsgIdAndTxt("NetForward:InferenceErrorStd", err_msg);
    }

    // --- Convert Output Tensor to MATLAB mxArray ---
    // PyTorch output shape: [B, C, H, W] = [1, 1, NGR, NGZ] (e.g., [1,1,16,16])
    // MATLAB output dimensions: H, W, C, B => {NGR, NGZ, 1, 1} (e.g., {16,16,1,1})
    try {
        const std::vector<int64_t> expected_torch_out_shape_vec = {1, 1, NGR, NGZ};
        torch::IntArrayRef expected_torch_out_shape(expected_torch_out_shape_vec); // Use IntArrayRef for direct comparison
        auto actual_torch_out_sizes = output_tensor_torch.sizes(); // This is already an IntArrayRef

        if (actual_torch_out_sizes.vec() != expected_torch_out_shape_vec) { // Compare as vectors for simplicity
            char err_msg[512];
            // Use the helper function for string conversion
            std::string actual_shape_str = intArrayRefToString(actual_torch_out_sizes);
            std::string expected_shape_str = intArrayRefToString(expected_torch_out_shape);

            snprintf(err_msg, sizeof(err_msg), "Unexpected output tensor shape. Got %s, expected %s.",
                     actual_shape_str.c_str(), expected_shape_str.c_str());
            mexErrMsgIdAndTxt("NetForward:OutputShapeError", err_msg);
        }

        // Convert output tensor to double (for MATLAB) and ensure it's contiguous
        output_tensor_torch = output_tensor_torch.to(torch::kDouble).contiguous();

        // MATLAB dimensions: H, W, C, B
        const mwSize matlab_out_dims[] = {(mwSize)NGR, (mwSize)NGZ, 1, 1};
        plhs[0] = mxCreateNumericArray(4, matlab_out_dims, mxDOUBLE_CLASS, mxREAL);
        double* out_matlab_ptr = mxGetPr(plhs[0]); // Get pointer to MATLAB output array data
        const double* out_torch_ptr = output_tensor_torch.data_ptr<double>(); // Pointer to LibTorch tensor data

        // Copy data element-by-element, transposing dimensions:
        // LibTorch (BCHW) to MATLAB (HWCB)
        const int64_t B = expected_torch_out_shape_vec[0]; // Should be 1
        const int64_t C = expected_torch_out_shape_vec[1]; // Should be 1
        const int64_t H = expected_torch_out_shape_vec[2]; // NGR
        const int64_t W = expected_torch_out_shape_vec[3]; // NGZ

        for (int64_t b_idx = 0; b_idx < B; ++b_idx) {
            for (int64_t c_idx = 0; c_idx < C; ++c_idx) {
                for (int64_t h_idx = 0; h_idx < H; ++h_idx) {     // MATLAB's H is fastest changing
                    for (int64_t w_idx = 0; w_idx < W; ++w_idx) { // MATLAB's W is next
                        // Source (Torch BCHW) linear index:
                        int64_t torch_linear_idx = b_idx * (C * H * W) +
                                                   c_idx * (H * W) +
                                                   h_idx * W +
                                                   w_idx;
                        // Destination (MATLAB HWCB) linear index:
                        int64_t matlab_linear_idx = h_idx +                     // H
                                                    w_idx * H +                 // W*H_dim
                                                    c_idx * (H * W) +           // C*H_dim*W_dim
                                                    b_idx * (H * W * C); // B*H_dim*W_dim*C_dim

                        out_matlab_ptr[matlab_linear_idx] = out_torch_ptr[torch_linear_idx];
                    }
                }
            }
        }
    } catch (const c10::Error& e) {
        char err_msg[512];
        snprintf(err_msg, sizeof(err_msg), "LibTorch error converting output to MATLAB: %s", e.what());
        mexErrMsgIdAndTxt("NetForward:OutputConversionErrorTorch", err_msg);
    } catch (const std::exception& e) {
        char err_msg[512];
        snprintf(err_msg, sizeof(err_msg), "Standard C++ error converting output to MATLAB: %s", e.what());
        mexErrMsgIdAndTxt("NetForward:OutputConversionErrorStd", err_msg);
    }
    // mexPrintf("net_forward: Inference complete.\n");
}

// --- Cleanup Function (called by MATLAB when MEX file is unloaded) ---
void cleanup_resources_at_exit() {
    mexPrintf("net_forward: Unloading MEX file and releasing PyTorch model...\n");
    if (G_module_ptr) {
        G_module_ptr.reset(); // Explicitly calls destructor of unique_ptr and releases model
        mexPrintf("net_forward: PyTorch model released.\n");
    }
    G_module_initialized = false;
    // G_mex_is_locked = false; // No need to change this here
    // mexUnlock(); // Not strictly necessary here, as MATLAB is already unloading.
    //              // Useful if you want to allow `clear mex` after some operations.
}