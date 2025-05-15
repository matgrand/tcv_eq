#include <torch/script.h>  // PyTorch header

#ifdef STANDALONE_MODE
// Standalone mode implementation

#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
    std::cout << "Running net_forward in standalone mode" << std::endl;
    
    try {
        // Load the module
        torch::jit::script::Module module = torch::jit::load("net.pt");
        module.eval();
        std::cout << "Module loaded successfully" << std::endl;
        
        // Create a sample input tensor
        torch::Tensor input = torch::ones({1, 2}, torch::kDouble);
        // input values should be 3.0, 5.0
        input[0][0] = 3.0;
        input[0][1] = 5.0;
        std::cout << "x -> [ ";
        for (int i = 0; i < input.size(1); ++i) {
            std::cout << std::showpos << std::fixed << std::setprecision(4) << input[0][i].item<double>();
            if (i < input.size(1) - 1) {
            std::cout << ", ";
            }
        }
        std::cout << " ]" << std::endl;
        
        // Forward pass
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input);
        torch::Tensor output = module.forward(inputs).toTensor();
        
        // Print output
        std::cout << "y -> [ ";
        for (int i = 0; i < output.size(1); ++i) {
            std::cout << std::showpos << std::fixed << std::setprecision(4) << output[0][i].item<double>();
            if (i < output.size(1) - 1) {
                std::cout << ", ";
            }
        }
        std::cout << " ]" << std::endl;
        
    } catch (const c10::Error& e) {
        std::cerr << "Error loading/running the model: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}

#else // MEX file implementation

#include <iostream>
#include <vector>
#include <cstring> // For std::memcpy
#include <stdexcept> // For std::runtime_error
#include <mex.h> // For MATLAB MEX functions
#include <matrix.h> // For mxArray, mxGetPr, etc.

// It's better to load the module once if the MEX function is called multiple times.
// For simplicity in this example, we'll keep loading it in run_inference,
// but for performance, consider making 'module' a persistent or static variable.
torch::jit::script::Module module; // Global or static module
bool module_loaded = false;

void load_module_once() {
    if (!module_loaded) {
        try {
            module = torch::jit::load("net.pt");
            module.eval(); // Ensure it's in evaluation mode
            module_loaded = true;
            mexPrintf("PyTorch module loaded successfully.\n");
        } catch (const c10::Error& e) {
            mexErrMsgIdAndTxt("MATLAB:net_forward:moduleLoadFailed", "Failed to load TorchScript module: %s", e.what());
        }
    }
}

torch::Tensor run_inference(const torch::Tensor& input) {
    load_module_once(); // Ensure module is loaded

    // Wrap input in a vector
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input);

    // Forward pass
    // Note: module.forward() might return an IValue that is not directly a Tensor
    // if the model has multiple outputs or a different kind of output.
    // For this simple model, .toTensor() is fine.
    at::Tensor output = module.forward(inputs).toTensor();
    return output;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    // Ensure module is loaded on first call or if MEX is cleared
    // Using mexMakeMemoryPersistent or similar for the module handle might be needed
    // if you want it to survive `clear mex`. For now, load_module_once works per session.
    // A mexAtExit function could be used to clean up if module were more complex.

    // Check number of input arguments
    if (nrhs != 1) {
        mexErrMsgIdAndTxt("MATLAB:net_forward:invalidNumInputs", "One input required.");
    }

    // Check input type
    if (!mxIsDouble(prhs[0])) {
        mexErrMsgIdAndTxt("MATLAB:net_forward:inputNotDouble", "Input must be a double array.");
    }

    // Get input dimensions
    mwSize num_dims = mxGetNumberOfDimensions(prhs[0]);
    const mwSize *dims = mxGetDimensions(prhs[0]);
    
    // For this example, we expect a 1xN or Nx1 vector, or a 2D matrix where one dim is 1.
    // We'll assume it's effectively a flat array of N elements for the model's 1xN input.
    size_t n_elements = mxGetNumberOfElements(prhs[0]);
    
    // For the model nn.Linear(2,3), input must be (BatchSize, 2)
    // MATLAB's [1.0, 2.0] is 1x2, so n_elements = 2.
    // The Python script uses x.reshape(1,2)
    if (n_elements != 2) {
         mexErrMsgIdAndTxt("MATLAB:net_forward:invalidInputSize", "Input must have 2 elements for a Linear(2,3) model.");
    }

    double* input_data_ptr = mxGetPr(prhs[0]);

    // Create a tensor from the input data
    torch::Tensor input = torch::from_blob(input_data_ptr, {1, (long int)n_elements}, torch::kDouble);

    // Run inference
    torch::Tensor output_tensor;
    output_tensor = run_inference(input);
    
    // Create MATLAB output matrix
    // Output of Linear(2,3) for a single batch item is (1,3)
    plhs[0] = mxCreateDoubleMatrix(1, output_tensor.numel(), mxREAL);
    double* output_data_ptr = mxGetPr(plhs[0]);

    // Copy data from the tensor to the MATLAB matrix
    // Ensure the output tensor is contiguous before accessing data_ptr if it might not be.
    // For a simple linear layer output, it's usually contiguous.
    torch::Tensor output_contiguous = output_tensor.contiguous();
    std::memcpy(output_data_ptr, output_contiguous.data_ptr<double>(), output_contiguous.numel() * sizeof(double));
}

// Optional: Add a cleanup function if you make `module` persistent across `clear mex`
// static void cleanup() {
//    // Code to release module if necessary
//    mexPrintf("MEX function unloaded, cleaning up module.\n");
//    module_loaded = false; 
//    // If 'module' was allocated on heap and managed by smart ptr, it's fine.
//    // If it was raw ptr, delete it. For torch::jit::script::Module, it's RAII.
// }

#endif // STANDALONE_MODE