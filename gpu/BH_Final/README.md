# Barnes-Hut CUDA

## Dependencies

* NVIDIA GPU with CUDA support
* OpenCV
  
## Building

```shell
mkdir build; cd build
cmake ..
cmake --build .
./BarnesHut <N> <Sim> <Iter>
```
**N**: the number of bodies  
**Sim**: the simulation (0: Spiral Galaxy, 1: Random, 2: Colliding Galaxies, 3: Our Solar System)  
**Iter**: number of iterations/frames


If you see any issues compiling the code using CMake due to dependency errors or library linking errors, force link the libraries using below command 

```shell
nvcc -o BarnesHutKernelTest -rdc=true main.cu -I/usr/include/opencv4 -L/usr/lib/x86_64-linux-gnu -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_videoio -ccbin /usr/bin/g++-10

```