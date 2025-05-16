#include <onnxruntime_cxx_api.h> // ONNX Runtime Header
#include <iostream>   // For std::cout in debug, though mexPrintf is preferred
#include <vector>
#include <cstring>    // For std::memcpy
#include <string>     // For std::string, std::to_string
#include <stdexcept>  // For std::runtime_error (though Ort::Exception is primary)
#include <mex.h>      // For MATLAB MEX functions
#include <matrix.h>   // For mxArray, mxGetPr, etc.
#include <filesystem> // For std::filesystem::path
// #include <fstream>   // For std::ifstream (if needed for file operations)
#include <unistd.h> // For readlink (to get executable path)
#include <limits.h> // For PATH_MAX (to define buffer size for readlink)


// ONNX Runtime global objects, similar to how 'module' was global for LibTorch
Ort::Env ort_env(ORT_LOGGING_LEVEL_WARNING, "net_forward_mex_env"); // Initialize once
Ort::SessionOptions session_options;
Ort::Session* ort_session = nullptr; // Global session pointer, initialized in load_session_once
bool session_loaded = false;
Ort::AllocatorWithDefaultOptions allocator; // Global allocator

// Store owned strings for node names, and char* pointers for the ONNX Runtime API
std::string global_input_name_str;
std::string global_output_name_str;
const char* global_input_node_names[1];  // Array for Run API, assuming 1 input
const char* global_output_node_names[1]; // Array for Run API, assuming 1 output

const char* onnx_env_dir = std::getenv("ONNX_NET_FORWARD_DIR");
const std::string net_default_path = onnx_env_dir ? (std::string(onnx_env_dir) + "/net.onnx") : "net.onnx"; // Default model path


// Cleanup function to be called by mexAtExit to release ONNX Runtime session
static void cleanup_session() {
    if (ort_session) {
        delete ort_session;
        ort_session = nullptr; // Prevent double deletion
        mexPrintf("ONNX Runtime session cleaned up.\n");
    }
    session_loaded = false; // Reset state
    // ort_env, session_options, and allocator are global stack objects;
    // they will be destructed when the MEX DLL is unloaded.
    // global_input_name_str and global_output_name_str too.
}


// Loads the ONNX session once. Corresponds to 'load_module_once'
void load_session_once(std::filesystem::path model_path) {
    if (!session_loaded) {
        try {
            mexPrintf("Loading ONNX model from: %s ...", model_path.string().c_str());

            // Optional: configure session_options here (e.g., for execution providers)
            // session_options.SetIntraOpNumThreads(1);
            
            ort_session = new Ort::Session(ort_env, model_path.string().c_str(), session_options);
            
            // Get input node name (assuming single input for this model)
            Ort::AllocatedStringPtr input_name_alloc = ort_session->GetInputNameAllocated(0, allocator);
            global_input_name_str = input_name_alloc.get(); // Store the string data in our global std::string
            global_input_node_names[0] = global_input_name_str.c_str(); // Use pointer to our stored string data

            // Get output node name (assuming single output for this model)
            Ort::AllocatedStringPtr output_name_alloc = ort_session->GetOutputNameAllocated(0, allocator);
            global_output_name_str = output_name_alloc.get(); // Store the string data
            global_output_node_names[0] = global_output_name_str.c_str(); // Use pointer

            session_loaded = true;
            mexPrintf(" Success.\n");

            // Register cleanup function with MATLAB to be called when MEX is cleared or MATLAB exits
            // This should only be done once.
            static bool cleanup_registered = false;
            if (!cleanup_registered) {
                mexAtExit(cleanup_session);
                cleanup_registered = true;
            }

        } catch (const Ort::Exception& e) {
            delete ort_session; // Clean up if 'new Ort::Session' succeeded but a subsequent Ort call failed
            ort_session = nullptr;
            std::string err_msg = "Failed to load ONNX model: " + std::string(e.what()) + " (ErrorCode: " + std::to_string(e.GetOrtErrorCode()) + ")";
            mexErrMsgIdAndTxt("MATLAB:net_forward:sessionLoadFailed", err_msg.c_str());
        } catch (const std::exception& e) { // Catch other potential errors during setup
            delete ort_session; // If new succeeded but std::string op failed etc.
            ort_session = nullptr;
            std::string err_msg = "A standard error occurred during ONNX session loading: " + std::string(e.what());
            mexErrMsgIdAndTxt("MATLAB:net_forward:sessionLoadFailedStdEx", err_msg.c_str());
        }
    }
}

// Corresponds to 'run_inference'. Takes an ONNX input tensor and returns ONNX output tensors.
// The Ort::Value objects in the returned vector own their data.
std::vector<Ort::Value> run_inference(Ort::Value& input_tensor) {
    try {
        // Run inference
        auto output_values = ort_session->Run(Ort::RunOptions{nullptr},
                                              global_input_node_names, &input_tensor, 1, // Pass pointer to input_tensor
                                              global_output_node_names, 1);
        return output_values; // std::vector<Ort::Value>
    } catch (const Ort::Exception& e) {
        std::string err_msg = "Error during ONNX Runtime inference: " + std::string(e.what()) + " (ErrorCode: " + std::to_string(e.GetOrtErrorCode()) + ")";
        mexErrMsgIdAndTxt("MATLAB:net_forward:inferenceError", err_msg.c_str());
        return {}; // Should not be reached due to mexErrMsgIdAndTxt throwing an exception
    } catch (const std::exception& e) {
        std::string err_msg = "A standard error occurred during ONNX inference: " + std::string(e.what());
        mexErrMsgIdAndTxt("MATLAB:net_forward:inferenceErrorStdEx", err_msg.c_str());
        return {}; // Should not be reached
    }
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    std::filesystem::path model_path = net_default_path;
    // Check number of input arguments
    if (nrhs < 1 && session_loaded) {
        mexErrMsgIdAndTxt("MATLAB:net_forward:invalidNumInputs", "Model already loaded: at least one input required.");
    }
    if (nrhs > 2) {
        mexErrMsgIdAndTxt("MATLAB:net_forward:invalidNumInputs", "Too many input arguments");
    }
    if (nrhs == 1 && !session_loaded && mxIsChar(prhs[0])) {
        // If only one input and it's a string, treat it as the model path
        char model_path_buf[PATH_MAX];
        if (mxGetString(prhs[0], model_path_buf, sizeof(model_path_buf)) != 0) {
            mexErrMsgIdAndTxt("MATLAB:net_forward:modelPathTooLong", "Model path string is too long.");
        }
        model_path = model_path_buf;
        mexPrintf("Using model path: %s\n", model_path_buf);
    } else if (nrhs == 1) {
        // If no model path is provided, use the default
        model_path = net_default_path;
    }
    
    // Ensure ONNX session is loaded (this also registers mexAtExit if it's the first successful load)
    load_session_once(model_path); 

    // Check input type
    if (!mxIsDouble(prhs[0])) {
        mexErrMsgIdAndTxt("MATLAB:net_forward:inputNotDouble", "Input must be a double array.");
    }

    // Get input dimensions and data pointer
    size_t n_elements = mxGetNumberOfElements(prhs[0]);
    
    // For the model nn.Linear(2,3), input must be (BatchSize, 2)
    // MATLAB's [1.0, 2.0] is 1x2 (row vector) or 2x1 (column vector), n_elements = 2.
    // The Python script used x.reshape(1,2). We maintain this expectation.
    if (n_elements != 2) {
        mexErrMsgIdAndTxt("MATLAB:net_forward:invalidInputSize", "Input must have 2 elements for a Linear(2,3) model.");
    }

    double* input_data_ptr_matlab = mxGetPr(prhs[0]);

    // Create ONNX Runtime input tensor from MATLAB data
    std::vector<int64_t> input_tensor_shape = {1, (long int)n_elements}; // Expected shape {1, 2}
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    Ort::Value input_ort_tensor = Ort::Value::CreateTensor<double>(
        memory_info, input_data_ptr_matlab, n_elements,
        input_tensor_shape.data(), input_tensor_shape.size()
    );

    // Run inference
    // Original: torch::Tensor output_tensor = run_inference(input_torch_tensor);
    std::vector<Ort::Value> output_ort_tensors = run_inference(input_ort_tensor);
    
    // Process output
    if (output_ort_tensors.empty() || !output_ort_tensors[0].IsTensor()) {
        mexErrMsgIdAndTxt("MATLAB:net_forward:inferenceFailed", "Inference did not return a valid tensor.");
    }
    Ort::Value& output_onnx_tensor_ref = output_ort_tensors[0]; // Get ref to the first output tensor

    // Get output tensor properties
    Ort::TensorTypeAndShapeInfo output_shape_info = output_onnx_tensor_ref.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType output_type = output_shape_info.GetElementType();

    if (output_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE) {
        mexErrMsgIdAndTxt("MATLAB:net_forward:unexpectedOutputType", "Output tensor is not double type as expected by this MEX function.");
    }

    std::vector<int64_t> output_dims = output_shape_info.GetShape();
    size_t output_total_elements = output_shape_info.GetElementCount();

    // Create MATLAB output matrix
    // Original code: plhs[0] = mxCreateDoubleMatrix(1, output_tensor.numel(), mxREAL);
    // This implies the output is expected to be a row vector in MATLAB.
    // For a Linear(2,3) model with input [1,2], output is [1,3].
    if (output_dims.size() == 2 && output_dims[0] == 1) { // Standard case: [1, N]
        plhs[0] = mxCreateDoubleMatrix(output_dims[0], output_dims[1], mxREAL);
    } else if (output_dims.size() == 1) { // If ONNX output is 1D vector [N], create as [1, N] MATLAB matrix
        plhs[0] = mxCreateDoubleMatrix(1, output_dims[0], mxREAL);
    } else {
        // For more complex N-D shapes, mxCreateNumericArray would be needed.
        // This simplified handling matches the original's apparent expectation.
        std::string shape_str;
        for(size_t i=0; i<output_dims.size(); ++i) shape_str += std::to_string(output_dims[i]) + (i == output_dims.size()-1 ? "" : "x");
        std::string err_msg = "Output tensor shape {" + shape_str + "} is not the expected [1,N] or [N] for simple MATLAB matrix creation.";
        mexErrMsgIdAndTxt("MATLAB:net_forward:unsupportedOutputDim", err_msg.c_str());
    }
    
    double* output_matlab_ptr = mxGetPr(plhs[0]);
    const double* inferred_output_data_ptr = output_onnx_tensor_ref.GetTensorData<double>();

    // Copy data from ONNX tensor to MATLAB matrix
    // Original: std::memcpy(output_data_ptr, output_contiguous.data_ptr<double>(), output_contiguous.numel() * sizeof(double));
    std::memcpy(output_matlab_ptr, inferred_output_data_ptr, output_total_elements * sizeof(double));
    
    // Ort::Value objects in output_ort_tensors (and their underlying data buffers if owned by ONNX Runtime)
    // are managed. They will be destructed when output_ort_tensors goes out of scope.
}