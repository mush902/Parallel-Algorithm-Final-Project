# DirectSum CUDA 

## Dependencies

* NVIDIA GPU with CUDA support
* OpenCV
  
## Building

```shell
mkdir build; cd build
cmake ..
cmake --build .
./DirectSum <N> <Sim> <Iter>
```
**N**: the number of bodies  
**Sim**: the simulation (0: Spiral Galaxy, 1: Random, 2: Our Solar System)  
**Iter**: number of iterations/frames