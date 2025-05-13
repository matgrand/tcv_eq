// standalone_net_benchmark.cpp
// Benchmarks a PyTorch model inference.

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


int main(int argc, char* argv[]) {
    std::string model_path = "net.pt"; // Default model path
    if (argc > 1) { // If an argument is provided, use it as the model path
        model_path = argv[1];
    }
    // Basic usage instruction if too many args
    if (argc > 2) {
        std::cerr << "Usage: " << argv[0] << " [optional_path_to_net.pt]" << std::endl;
        return 1;
    }

    const int num_warmup_runs = 100;    // Number of untimed runs to load model and let JIT (if any) settle
    const int num_timed_runs = 100000;   // Number of timed inference runs for benchmarking

    std::cout << "Starting network inference benchmark..." << std::endl;
    std::cout << "Model path: " << model_path << std::endl;
    std::cout << "Warm-up runs: " << num_warmup_runs << std::endl;
    std::cout << "Timed runs: " << num_timed_runs << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // Define tensor options (float type, default device which is CPU for CPU-only LibTorch)
    auto tensor_opts = torch::TensorOptions().dtype(torch::kFloat);

    // --- Warm-up Phase ---
    // The first call to forward_pass will load the model.
    std::cout << "Performing warm-up runs..." << std::endl;
    for (int i = 0; i < num_warmup_runs; ++i) {
        try {
            // Generate random inputs for each warm-up run
            torch::Tensor input1 = torch::randn({1, 95}, tensor_opts);
            torch::Tensor input2 = torch::randn({1, 16}, tensor_opts);
            torch::Tensor input3 = torch::randn({1, 16}, tensor_opts);

            torch::Tensor output = forward_pass(input1, input2, input3, model_path);
            
            // Print info for the first warm-up run (which includes model loading)
            if (i == 0) { 
                 std::cout << "First warm-up run (includes model load) output shape: " << output.sizes() << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "Fatal Error during warm-up run " << i + 1 << ": " << e.what() << std::endl;
            return 1; // Terminate if warm-up fails
        }
    }
    std::cout << "Warm-up complete." << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // --- Timed Inference Phase ---
    std::vector<double> durations_ms; // To store durations of timed runs
    durations_ms.reserve(num_timed_runs);

    std::cout << "Performing " << num_timed_runs << " timed inference runs..." << std::endl;
    int print_interval = std::max(1, num_timed_runs / 10); // Print progress roughly 10 times

    for (int i = 0; i < num_timed_runs; ++i) {
        try {
            // Generate new random inputs for each timed run
            torch::Tensor input1 = torch::randn({1, 95}, tensor_opts);
            torch::Tensor input2 = torch::randn({1, 16}, tensor_opts);
            torch::Tensor input3 = torch::randn({1, 16}, tensor_opts);

            auto start_time = std::chrono::high_resolution_clock::now();
            torch::Tensor output = forward_pass(input1, input2, input3, model_path); // model_path is passed but only used if model not yet loaded
            auto end_time = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> duration_chrono = end_time - start_time;
            durations_ms.push_back(duration_chrono.count());

            if ((i + 1) % print_interval == 0 || i == num_timed_runs - 1) {
                std::cout << "Run " << std::setw(3) << i + 1 << "/" << num_timed_runs
                          << " | Duration: " << std::fixed << std::setprecision(3) << std::setw(7) << duration_chrono.count() << " ms"
                          //<< " | Output sum: " << std::setprecision(5) << output.sum().item<float>() // Optional: check output
                          << std::endl;
            }

        } catch (const std::exception& e) {
            std::cerr << "Fatal Error during timed run " << i + 1 << ": " << e.what() << std::endl;
            return 1; // Terminate if a timed run fails
        }
    }
    std::cout << "Timed inference runs complete." << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // --- Calculate and Print Statistics ---
    if (durations_ms.empty()) {
        std::cout << "No timed runs were completed to calculate statistics." << std::endl;
    } else {
        double sum_durations = std::accumulate(durations_ms.begin(), durations_ms.end(), 0.0);
        double mean_duration = sum_durations / durations_ms.size();

        double sum_sq_diff = 0.0;
        for (double d : durations_ms) {
            sum_sq_diff += (d - mean_duration) * (d - mean_duration);
        }
        // Use (N-1) for sample standard deviation if N is small, N for population if N is large.
        // For N=100, N is fine. If durations_ms.size() could be 1, then N-1 would be division by zero.
        double std_dev_duration = (durations_ms.size() > 1) ? 
                                  std::sqrt(sum_sq_diff / (durations_ms.size() - 1)) : // Sample StdDev
                                  0.0; // Or std::sqrt(sum_sq_diff / durations_ms.size()) for Population StdDev

        std::cout << std::fixed << std::setprecision(3); // Set precision for printing stats
        std::cout << "Inference Time Statistics (" << durations_ms.size() << " runs):" << std::endl;
        std::cout << "  Mean:   " << std::setw(7) << mean_duration << " ms" << std::endl;
        std::cout << "  StdDev: " << std::setw(7) << std_dev_duration << " ms" << std::endl;
        if (!durations_ms.empty()) {
            std::cout << "  Min:    " << std::setw(7) << *std::min_element(durations_ms.begin(), durations_ms.end()) << " ms" << std::endl;
            std::cout << "  Max:    " << std::setw(7) << *std::max_element(durations_ms.begin(), durations_ms.end()) << " ms" << std::endl;
        }
    }
    std::cout << "----------------------------------------" << std::endl;

    return 0;
}