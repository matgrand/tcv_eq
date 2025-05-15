#!/bin/bash
if [ ! -d "$(pwd)/libtorch" ]; then
    echo "libtorch directory not found. Downloading libtorch..."
    wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip -O libtorch.zip
    echo "Unzipping libtorch..."
    unzip libtorch.zip
    rm libtorch.zip
    echo "libtorch downloaded and extracted."
fi

echo "Compiling net_forward_mex.cpp..."
libtorch_path="$(pwd)/libtorch"
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH="$libtorch_path"
make

cd ..

# test standalone version
cp build/net_forward net_forward_standalone
echo "Testing standalone version..."
echo "---------------------------------------------------------"
./net_forward_standalone
echo "---------------------------------------------------------"
echo "Standalone version test completed."

echo "Copying mex file to main directory for MATLAB..."
# Copy the compiled mex files to the current directory
for file in build/net_forward_mex.mex*; do
    cp "$file" "net_forward.${file##*.}"
done

# copy libs to build directory
cp -r libtorch/lib build/

# # for now this is forced:
# export LD_PRELOAD="$(pwd)/libtorch/lib/libtorch_cpu.so:$(pwd)/libtorch/lib/libtorch.so"
export LD_PRELOAD="$(pwd)/libtorch/lib/libtorch.so"

# export LD_LIBRARY_PATH="$(pwd)/libtorch/lib:${LD_LIBRARY_PATH}" # doesnt work


# start MATLAB -> run the script forward_test.m -> exit
matlab -nodisplay -nosplash -nodesktop -r "run('forward_test.m'); exit;"




