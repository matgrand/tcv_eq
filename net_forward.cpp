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

const int PHYS_SIZE = 136; // Size of the 'phys' input

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

// Loads the ONNX session once
void load_session_once(std::filesystem::path model_path) {
    if (!session_loaded) {
        try {
            mexPrintf("Loading ONNX model from: %s ...", model_path.string().c_str());

            // Optional: configure session_options here (e.g., for execution providers)
            // session_options.SetIntraOpNumThreads(1); // Set number of threads for intra-op parallelism
            // session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            
            // Create the ONNX Runtime session
            ort_session = new Ort::Session(ort_env, model_path.string().c_str(), session_options);
            
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


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // TODO: impleent this as -> if read "-model" interpret next argument as model path
    std::filesystem::path model_path = net_default_path;
    // Check number of input arguments
    if (nrhs < 1 && session_loaded) {
        mexErrMsgIdAndTxt("MATLAB:net_forward:invalidNumInputs", "Model already loaded: 3 inputs required: (phys, r, z) or a single string argument for model path.");
    }
    if (nrhs > 3) {
        mexErrMsgIdAndTxt("MATLAB:net_forward:invalidNumInputs", "Too many input arguments");
    }
    if (nrhs == 1 && mxIsChar(prhs[0])) { // (nrhs == 1 && !session_loaded && mxIsChar(prhs[0]))
        // If only one input and it's a string, treat it as the model path
        char model_path_buf[PATH_MAX];
        if (mxGetString(prhs[0], model_path_buf, sizeof(model_path_buf)) != 0) {
            mexErrMsgIdAndTxt("MATLAB:net_forward:modelPathTooLong", "Model path string is too long.");
        }
        std::filesystem::path model_path = model_path_buf;
        mexPrintf("Using model path: %s\n", model_path_buf);
        load_session_once(model_path); // Load the session with the provided model path
        return; // Exit after loading the session
    } else {
        if (nrhs != 3) { // If not a single string input, expect three inputs
            mexErrMsgIdAndTxt("MATLAB:net_forward:invalidNumInputs", "Expected 3 inputs: (phys, r, z) or a single string argument for model path.");
        }
        if (!mxIsSingle(prhs[0])) { // Check input type
            mexErrMsgIdAndTxt("MATLAB:net_forward:inputNotSingle", "Input must be a single array.");
        }
    }
    
    // Ensure ONNX session is loaded (this also registers mexAtExit if it's the first successful load)
    load_session_once(net_default_path); 

    // Get input dimensions and data pointer
    size_t phys_size = mxGetNumberOfElements(prhs[0]);
    size_t n_pts = mxGetNumberOfElements(prhs[1]);
    size_t n_pts2 = mxGetNumberOfElements(prhs[2]);

    // check n_pts == n_pts2
    if (n_pts != n_pts2) {
        mexErrMsgIdAndTxt("MATLAB:net_forward:inputSizeMismatch", "Inputs 'r' and 'z' must have the same number of elements.");
    }

    //check prhs[1] and prhs[2] have the same size
    if (mxGetNumberOfElements(prhs[1]) != mxGetNumberOfElements(prhs[2])) {
        mexErrMsgIdAndTxt("MATLAB:net_forward:inputSizeMismatch", "Inputs 'r' and 'z' must have the same size.");
    }
    
    if (phys_size != PHYS_SIZE) {
        mexErrMsgIdAndTxt("MATLAB:net_forward:invalidInputSize", "Input size must be %d, but got %zu.", PHYS_SIZE, phys_size);
    }

    // names
    std::vector<const char*> input_names = {"phys", "r", "z"};
    std::vector<const char*> output_names = {"Fx", "Br", "Bz"};
    // data
    float* phys = (float*)mxGetData(prhs[0]);
    float* r    = (float*)mxGetData(prhs[1]);
    float* z    = (float*)mxGetData(prhs[2]);

    // Create ONNX Runtime input tensor from MATLAB data
    std::vector<int64_t> phys_shape = {PHYS_SIZE}; // Shape for 'phys'
    std::vector<int64_t> rz_shape = {static_cast<int64_t>(n_pts)}; // Shape for 'r' and 'z'
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<Ort::Value> input_tensors; // Vector to hold input tensors
    
    input_tensors.emplace_back( Ort::Value::CreateTensor<float>(
        memory_info, phys, phys_size, phys_shape.data(), phys_shape.size()));
    input_tensors.emplace_back( Ort::Value::CreateTensor<float>(
        memory_info, r, n_pts, rz_shape.data(), rz_shape.size()));
    input_tensors.emplace_back( Ort::Value::CreateTensor<float>(
        memory_info, z, n_pts, rz_shape.data(), rz_shape.size()));

    // Run inference
    auto output_tensors = ort_session->Run(
                            Ort::RunOptions{nullptr},
                            input_names.data(), input_tensors.data(), input_names.size(),
                            output_names.data(), output_names.size());

    // Process output
    if (output_tensors.size() != 3) {
        mexErrMsgIdAndTxt("MATLAB:net_forward:invalidNumOutputs", "Expected 3 output tensors, but got %zu.", output_tensors.size());
    }

    Ort::Value& fx_ref = output_tensors[0];
    Ort::Value& br_ref = output_tensors[1];
    Ort::Value& bz_ref = output_tensors[2];

    // Get output tensor properties
    Ort::TensorTypeAndShapeInfo fx_shape_info = fx_ref.GetTensorTypeAndShapeInfo();
    Ort::TensorTypeAndShapeInfo br_shape_info = br_ref.GetTensorTypeAndShapeInfo();
    Ort::TensorTypeAndShapeInfo bz_shape_info = bz_ref.GetTensorTypeAndShapeInfo();
    
    // check types
    ONNXTensorElementDataType fx_type = fx_shape_info.GetElementType();
    ONNXTensorElementDataType br_type = br_shape_info.GetElementType();
    ONNXTensorElementDataType bz_type = bz_shape_info.GetElementType();
    if (fx_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        br_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        bz_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        mexErrMsgIdAndTxt("MATLAB:net_forward:invalidOutputType", "Expected output tensors to be of type float.");
    }

    std::vector<int64_t> fx_shape = fx_shape_info.GetShape();
    std::vector<int64_t> br_shape = br_shape_info.GetShape();
    std::vector<int64_t> bz_shape = bz_shape_info.GetShape();
    size_t fx_n = fx_shape_info.GetElementCount();
    size_t br_n = br_shape_info.GetElementCount();
    size_t bz_n = bz_shape_info.GetElementCount();
    if (fx_n != n_pts || br_n != n_pts || bz_n != n_pts) {
        mexErrMsgIdAndTxt("MATLAB:net_forward:invalidOutputSize", "Output tensors must have the same number of elements as input 'r' and 'z': expected %zu, but got fx: %zu, br: %zu, bz: %zu.", n_pts, fx_n, br_n, bz_n);
    }

    // Check output shapes
    if (fx_shape.size() != 1 || fx_shape[0] != n_pts) {
        mexErrMsgIdAndTxt("MATLAB:net_forward:invalidOutputShape", "Output tensor 'Fx' must have shape (n_pts, ) but got [%zu].", fx_shape);
    }
    if (br_shape.size() != 1 || br_shape[0] != n_pts) {
        mexErrMsgIdAndTxt("MATLAB:net_forward:invalidOutputShape", "Output tensor 'Br' must have shape (n_pts, ) but got [%zu].", br_shape);
    }
    if (bz_shape.size() != 1 || bz_shape[0] != n_pts) {
        mexErrMsgIdAndTxt("MATLAB:net_forward:invalidOutputShape", "Output tensor 'Bz' must have shape (n_pts, ) but got [%zu].", bz_shape);
    }

    // Get pointer to ONNX output tensor data
    const float* fx_ort_ptr = fx_ref.GetTensorData<float>(); 
    const float* br_ort_ptr = br_ref.GetTensorData<float>(); 
    const float* bz_ort_ptr = bz_ref.GetTensorData<float>(); 

    // // no checks for output tensors, assume they are valid
    // const float* fx_ort_ptr = output_tensors[0].GetTensorData<float>(); 
    // const float* br_ort_ptr = output_tensors[1].GetTensorData<float>(); 
    // const float* bz_ort_ptr = output_tensors[2].GetTensorData<float>(); 

    // initialize output matrices
    plhs[0] = mxCreateNumericMatrix(1, n_pts, mxSINGLE_CLASS, mxREAL); // Fx
    plhs[1] = mxCreateNumericMatrix(1, n_pts, mxSINGLE_CLASS, mxREAL); // Br
    plhs[2] = mxCreateNumericMatrix(1, n_pts, mxSINGLE_CLASS, mxREAL); // Bz
    
    // Get pointer to MATLAB output matrix
    float* fx_matlab_ptr = (float*)mxGetData(plhs[0]); 
    float* br_matlab_ptr = (float*)mxGetData(plhs[1]); 
    float* bz_matlab_ptr = (float*)mxGetData(plhs[2]); 

    // Copy data from ONNX tensor to MATLAB matrix
    std::memcpy(fx_matlab_ptr, fx_ort_ptr, n_pts * sizeof(float));
    std::memcpy(br_matlab_ptr, br_ort_ptr, n_pts * sizeof(float));
    std::memcpy(bz_matlab_ptr, bz_ort_ptr, n_pts * sizeof(float));
    
    // Ort::Value objects in output_tensors (and their underlying data buffers if owned by ONNX Runtime)
    // are managed. They will be destructed when output_tensors goes out of scope.
}