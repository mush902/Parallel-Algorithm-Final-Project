#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>



#define NUM_BODIES 300
#define WINDOW_WIDTH 2048
#define WINDOW_HEIGHT 2048
#define NBODY_WIDTH 10.0e11
#define NBODY_HEIGHT 10.0e11
#define GRAVITY 6.67E-11
#define E 0.5
#define DT 25000
#define THETA 0.5
#define CENTERX 0
#define CENTERY 0
#define BLOCK_SIZE 1024
#define GRID_SIZE 512
#define MAX_N 4194304
#define MAX_NODES 349525
#define MAX_DEPTH 9
#define N_LEAF 262144
#define COLLISION_TH 1.0e10
#define MIN_DIST 2.0e10
#define MAX_DIST 5.0e11
#define SUN_MASS 1.9890e30
#define SUN_DIA 1.3927e6
#define EARTH_MASS 5.974e24
#define EARTH_DIA 12756
#define HBL 1.6e29



#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char *const func, const char *const file,
           const int line)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK_LAST_CUDA_ERROR() checkLast(__FILE__, __LINE__)
void checkLast(const char *const file, const int line)
{
    cudaError_t err{cudaGetLastError()};
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}



typedef struct
{
    double x;
    double y;
} Vector;

typedef struct
{
    bool isDynamic;
    double mass;
    double radius;
    Vector position;
    Vector velocity;
    Vector acceleration;

} Body;

typedef struct
{
    Vector topLeft;
    Vector botRight;
    Vector centerMass;
    double totalMass;
    bool isLeaf;
    int start;
    int end;

} Node;

class BarnesHutCuda
{
    int nBodies;
    int nNodes;
    int leafLimit;

    Body *h_b;
    Node *h_node;

    Body *d_b;
    Body *d_b_buffer;
    Node *d_node;
    int *d_mutex;

    void initRandomBodies();
    void initSpiralBodies();
    void initCollideGalaxy();
    void initSolarSystem();
    void setBody(int i, bool isDynamic, double mass, double radius, Vector position, Vector velocity, Vector acceleration);
    void resetCUDA();
    void computeBoundingBoxCUDA();
    void constructQuadTreeCUDA();
    void computeForceCUDA();

public:
    BarnesHutCuda(int n);
    ~BarnesHutCuda();
    void update();
    void setup(int sim);
    void readDeviceBodies();
    Body *getBodies();
};


__global__ void ResetKernel(Node *node, int *mutex, int nNodes, int nBodies)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;

    if (b < nNodes)
    {
        node[b].topLeft = {INFINITY, -INFINITY};
        node[b].botRight = {-INFINITY, INFINITY};
        node[b].centerMass = {-1, -1};
        node[b].totalMass = 0.0;
        node[b].isLeaf = true;
        node[b].start = -1;
        node[b].end = -1;
        mutex[b] = 0;
    }

    if (b == 0)
    {
        node[b].start = 0;
        node[b].end = nBodies - 1;
    }
}

/*
----------------------------------------------------------------------------------------
COMPUTE BOUNDING BOX
----------------------------------------------------------------------------------------
*/
__global__ void ComputeBoundingBoxKernel(Node *node, Body *bodies, int *mutex, int nBodies)
{

    __shared__ double topLeftX[BLOCK_SIZE];
    __shared__ double topLeftY[BLOCK_SIZE];
    __shared__ double botRightX[BLOCK_SIZE];
    __shared__ double botRightY[BLOCK_SIZE];

    int tx = threadIdx.x;
    int b = blockIdx.x * blockDim.x + tx;

    topLeftX[tx] = INFINITY;
    topLeftY[tx] = -INFINITY;
    botRightX[tx] = -INFINITY;
    botRightY[tx] = INFINITY;

    __syncthreads();

    if (b < nBodies)
    {
        Body body = bodies[b];
        topLeftX[tx] = body.position.x;
        topLeftY[tx] = body.position.y;
        botRightX[tx] = body.position.x;
        botRightY[tx] = body.position.y;
    }

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        __syncthreads();
        if (tx < s)
        {
            topLeftX[tx] = fminf(topLeftX[tx], topLeftX[tx + s]);
            topLeftY[tx] = fmaxf(topLeftY[tx], topLeftY[tx + s]);
            botRightX[tx] = fmaxf(botRightX[tx], botRightX[tx + s]);
            botRightY[tx] = fminf(botRightY[tx], botRightY[tx + s]);
        }
    }

    if (tx == 0)
    {
        while (atomicCAS(mutex, 0, 1) != 0)
            ;
        node[0].topLeft.x = fminf(node[0].topLeft.x, topLeftX[0] - 1.0e10);
        node[0].topLeft.y = fmaxf(node[0].topLeft.y, topLeftY[0] + 1.0e10);
        node[0].botRight.x = fmaxf(node[0].botRight.x, botRightX[0] + 1.0e10);
        node[0].botRight.y = fminf(node[0].botRight.y, botRightY[0] - 1.0e10);
        atomicExch(mutex, 0);
    }
}

/*
----------------------------------------------------------------------------------------
CONSTRUCT QUAD TREE
----------------------------------------------------------------------------------------
*/
__device__ int getQuadrant(Vector topLeft, Vector botRight, double x, double y)
{

    if ((topLeft.x + botRight.x) / 2 >= x)
    {
        // Indicates topLeftTree
        if ((topLeft.y + botRight.y) / 2 <= y)
        {
            return 2;
        }
        // Indicates botLeftTree
        else
        {
            return 3;
        }
    }
    else
    {
        // Indicates topRightTree
        if ((topLeft.y + botRight.y) / 2 <= y)
        {
            return 1;
        }
        // Indicates botRightTree
        else
        {
            return 4;
        }
    }
}

__device__ void UpdateChildBound(Vector &tl, Vector &br, Node &childNode, int quadrant)
{

    if (quadrant == 1)
    {
        childNode.topLeft = {(tl.x + br.x) / 2, tl.y};
        childNode.botRight = {br.x, (tl.y + br.y) / 2};
    }
    else if (quadrant == 2)
    {
        childNode.topLeft = {tl.x, tl.y};
        childNode.botRight = {(tl.x + br.x) / 2, (tl.y + br.y) / 2};
    }
    else if (quadrant == 3)
    {
        childNode.topLeft = {tl.x, (tl.y + br.y) / 2};
        childNode.botRight = {(tl.x + br.x) / 2, br.y};
    }
    else
    {
        childNode.topLeft = {(tl.x + br.x) / 2, (tl.y + br.y) / 2};
        childNode.botRight = {br.x, br.y};
    }
}

__device__ void warpReduce(volatile double *totalMass, volatile double2 *centerMass, int tx)
{
    totalMass[tx] += totalMass[tx + 32];
    centerMass[tx].x += centerMass[tx + 32].x;
    centerMass[tx].y += centerMass[tx + 32].y;
    totalMass[tx] += totalMass[tx + 16];
    centerMass[tx].x += centerMass[tx + 16].x;
    centerMass[tx].y += centerMass[tx + 16].y;
    totalMass[tx] += totalMass[tx + 8];
    centerMass[tx].x += centerMass[tx + 8].x;
    centerMass[tx].y += centerMass[tx + 8].y;
    totalMass[tx] += totalMass[tx + 4];
    centerMass[tx].x += centerMass[tx + 4].x;
    centerMass[tx].y += centerMass[tx + 4].y;
    totalMass[tx] += totalMass[tx + 2];
    centerMass[tx].x += centerMass[tx + 2].x;
    centerMass[tx].y += centerMass[tx + 2].y;
    totalMass[tx] += totalMass[tx + 1];
    centerMass[tx].x += centerMass[tx + 1].x;
    centerMass[tx].y += centerMass[tx + 1].y;
}

__device__ void ComputeCenterMass(Node &curNode, Body *bodies, double *totalMass, double2 *centerMass, int start, int end)
{
    int tx = threadIdx.x;
    int total = end - start + 1;
    int sz = ceil((double)total / blockDim.x);
    int s = tx * sz + start;
    double M = 0.0;
    double2 R = make_double2(0.0, 0.0);

    for (int i = s; i < s + sz; ++i)
    {
        if (i <= end)
        {
            Body &body = bodies[i];
            M += body.mass;
            R.x += body.mass * body.position.x;
            R.y += body.mass * body.position.y;
        }
    }

    totalMass[tx] = M;
    centerMass[tx] = R;

    for (unsigned int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        __syncthreads();
        if (tx < stride)
        {
            totalMass[tx] += totalMass[tx + stride];
            centerMass[tx].x += centerMass[tx + stride].x;
            centerMass[tx].y += centerMass[tx + stride].y;
        }
    }

    if (tx < 32)
    {
        warpReduce(totalMass, centerMass, tx);
    }
    __syncthreads();

    if (tx == 0)
    {
        centerMass[0].x /= totalMass[0];
        centerMass[0].y /= totalMass[0];
        curNode.totalMass = totalMass[0];
        curNode.centerMass = {centerMass[0].x, centerMass[0].y};
    }
}

__device__ void CountBodies(Body *bodies, Vector topLeft, Vector botRight, int *count, int start, int end, int nBodies)
{
    int tx = threadIdx.x;
    if (tx < 4)
        count[tx] = 0;
    __syncthreads();

    for (int i = start + tx; i <= end; i += blockDim.x)
    {
        Body body = bodies[i];
        int q = getQuadrant(topLeft, botRight, body.position.x, body.position.y);
        atomicAdd(&count[q - 1], 1);
    }

    __syncthreads();
}

__device__ void ComputeOffset(int *count, int start)
{
    int tx = threadIdx.x;
    if (tx < 4)
    {
        int offset = start;
        for (int i = 0; i < tx; ++i)
        {
            offset += count[i];
        }
        count[tx + 4] = offset;
    }
    __syncthreads();
}

__device__ void GroupBodies(Body *bodies, Body *buffer, Vector topLeft, Vector botRight, int *count, int start, int end, int nBodies)
{
    int *count2 = &count[4];
    for (int i = start + threadIdx.x; i <= end; i += blockDim.x)
    {
        Body body = bodies[i];
        int q = getQuadrant(topLeft, botRight, body.position.x, body.position.y);
        int dest = atomicAdd(&count2[q - 1], 1);
        buffer[dest] = body;
    }
    __syncthreads();
}

__global__ void ConstructQuadTreeKernel(Node *node, Body *bodies, Body *buffer, int nodeIndex, int nNodes, int nBodies, int leafLimit)
{
    __shared__ int count[8];
    __shared__ double totalMass[BLOCK_SIZE];
    __shared__ double2 centerMass[BLOCK_SIZE];
    int tx = threadIdx.x;
    nodeIndex += blockIdx.x;

    if (nodeIndex >= nNodes)
        return;

    Node &curNode = node[nodeIndex];
    int start = curNode.start, end = curNode.end;
    Vector topLeft = curNode.topLeft, botRight = curNode.botRight;

    if (start == -1 && end == -1)
        return;

    ComputeCenterMass(curNode, bodies, totalMass, centerMass, start, end);
    if (nodeIndex >= leafLimit || start == end)
    {
        for (int i = start; i <= end; ++i)
        {
            buffer[i] = bodies[i];
        }

        return;
    }

    CountBodies(bodies, topLeft, botRight, count, start, end, nBodies);
    ComputeOffset(count, start);
    GroupBodies(bodies, buffer, topLeft, botRight, count, start, end, nBodies);

    if (tx == 0)
    {
        Node &topLNode = node[(nodeIndex * 4) + 2],
             &topRNode = node[(nodeIndex * 4) + 1], &botLNode = node[(nodeIndex * 4) + 3], &botRNode = node[(nodeIndex * 4) + 4];

        UpdateChildBound(topLeft, botRight, topLNode, 2);
        UpdateChildBound(topLeft, botRight, topRNode, 1);
        UpdateChildBound(topLeft, botRight, botLNode, 3);
        UpdateChildBound(topLeft, botRight, botRNode, 4);

        curNode.isLeaf = false;

        if (count[0] > 0)
        {
            topRNode.start = start;
            topRNode.end = start + count[0] - 1;
        }

        if (count[1] > 0)
        {
            topLNode.start = start + count[0];
            topLNode.end = start + count[0] + count[1] - 1;
        }

        if (count[2] > 0)
        {
            botLNode.start = start + count[0] + count[1];
            botLNode.end = start + count[0] + count[1] + count[2] - 1;
        }

        if (count[3] > 0)
        {
            botRNode.start = start + count[0] + count[1] + count[2];
            botRNode.end = end;
        }
        ConstructQuadTreeKernel<<<4, BLOCK_SIZE>>>(node, buffer, bodies, nodeIndex * 4 + 1, nNodes, nBodies, leafLimit);
    }
}

/*
----------------------------------------------------------------------------------------
COMPUTE FORCE
----------------------------------------------------------------------------------------
*/
__device__ double getDistance(Vector pos1, Vector pos2)
{

    return sqrt(pow(pos1.x - pos2.x, 2) + pow(pos1.y - pos2.y, 2));
}

__device__ bool isCollide(Body &b1, Vector cm)
{
    return b1.radius * 2 + COLLISION_TH > getDistance(b1.position, cm);
}

__device__ void ComputeForce(Node *node, Body *bodies, int nodeIndex, int bodyIndex, int nNodes, int nBodies, int leafLimit, double width)
{

    if (nodeIndex >= nNodes)
    {
        return;
    }
    Node curNode = node[nodeIndex];
    Body bi = bodies[bodyIndex];
    if (curNode.isLeaf)
    {
        if (curNode.centerMass.x != -1 && !isCollide(bi, curNode.centerMass))
        {
            Vector rij = {curNode.centerMass.x - bi.position.x, curNode.centerMass.y - bi.position.y};
            double r = sqrt((rij.x * rij.x) + (rij.y * rij.y) + (E * E));
            double f = (GRAVITY * bi.mass * curNode.totalMass) / (r * r * r + (E * E));
            Vector force = {rij.x * f, rij.y * f};

            bodies[bodyIndex].acceleration.x += (force.x / bi.mass);
            bodies[bodyIndex].acceleration.y += (force.y / bi.mass);
        }
        return;
    }

    double sd = width / getDistance(bi.position, curNode.centerMass);
    if (sd < THETA)
    {
        if (!isCollide(bi, curNode.centerMass))
        {
            Vector rij = {curNode.centerMass.x - bi.position.x, curNode.centerMass.y - bi.position.y};
            double r = sqrt((rij.x * rij.x) + (rij.y * rij.y) + (E * E));
            double f = (GRAVITY * bi.mass * curNode.totalMass) / (r * r * r + (E * E));
            Vector force = {rij.x * f, rij.y * f};

            bodies[bodyIndex].acceleration.x += (force.x / bi.mass);
            bodies[bodyIndex].acceleration.y += (force.y / bi.mass);
        }

        return;
    }

    ComputeForce(node, bodies, (nodeIndex * 4) + 1, bodyIndex, nNodes, nBodies, leafLimit, width / 2);
    ComputeForce(node, bodies, (nodeIndex * 4) + 2, bodyIndex, nNodes, nBodies, leafLimit, width / 2);
    ComputeForce(node, bodies, (nodeIndex * 4) + 3, bodyIndex, nNodes, nBodies, leafLimit, width / 2);
    ComputeForce(node, bodies, (nodeIndex * 4) + 4, bodyIndex, nNodes, nBodies, leafLimit, width / 2);
}

__global__ void ComputeForceKernel(Node *node, Body *bodies, int nNodes, int nBodies, int leafLimit)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double width = node[0].botRight.x - node[0].topLeft.x;

    if (i < nBodies)
    {
        Body &bi = bodies[i];
        if (bi.isDynamic)
        {
            bi.acceleration = {0.0, 0.0};
            ComputeForce(node, bodies, 0, i, nNodes, nBodies, leafLimit, width);
            bi.velocity.x += bi.acceleration.x * DT;
            bi.velocity.y += bi.acceleration.y * DT;
            bi.position.x += bi.velocity.x * DT;
            bi.position.y += bi.velocity.y * DT;
        }
    }
}








BarnesHutCuda::BarnesHutCuda(int n) : nBodies(n)
{
    nNodes = MAX_NODES;
    leafLimit = MAX_NODES - N_LEAF;
    h_b = new Body[n];
    h_node = new Node[nNodes];

    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_b, sizeof(Body) * n));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_node, sizeof(Node) * nNodes));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_mutex, sizeof(int) * nNodes));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_b_buffer, sizeof(Body) * n));
}

BarnesHutCuda::~BarnesHutCuda()
{

    delete[] h_b;
    delete[] h_node;
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_node));
    CHECK_CUDA_ERROR(cudaFree(d_mutex));
    CHECK_CUDA_ERROR(cudaFree(d_b_buffer));
}

void BarnesHutCuda::resetCUDA()
{
    int blockSize = BLOCK_SIZE;
    dim3 gridSize = ceil((float)nNodes / blockSize);
    ResetKernel<<<gridSize, blockSize>>>(d_node, d_mutex, nNodes, nBodies);
}
void BarnesHutCuda::computeBoundingBoxCUDA()
{
    int blockSize = BLOCK_SIZE;
    dim3 gridSize = ceil((float)nBodies / blockSize);
    ComputeBoundingBoxKernel<<<gridSize, blockSize>>>(d_node, d_b, d_mutex, nBodies);
}
void BarnesHutCuda::constructQuadTreeCUDA()
{
    int blockSize = BLOCK_SIZE;
    dim3 gridSize = ceil((float)nBodies / blockSize);
    ConstructQuadTreeKernel<<<1, blockSize>>>(d_node, d_b, d_b_buffer, 0, nNodes, nBodies, leafLimit);
}

void BarnesHutCuda::computeForceCUDA()
{
    int blockSize = 32;
    dim3 gridSize = ceil((float)nBodies / blockSize);
    ComputeForceKernel<<<gridSize, blockSize>>>(d_node, d_b, nNodes, nBodies, leafLimit);
}

void BarnesHutCuda::initRandomBodies()
{
    srand(time(NULL));
    double maxDistance = MAX_DIST;
    double minDistance = MIN_DIST;
    Vector centerPos = {CENTERX, CENTERY};
    for (int i = 0; i < nBodies - 1; ++i)
    {

        double angle = 2 * M_PI * (rand() / (double)RAND_MAX);
        // Generate random distance from center within the given max distance
        double radius = (maxDistance - minDistance) * (rand() / (double)RAND_MAX) + minDistance;

        // Calculate coordinates of the point
        double x = centerPos.x + radius * std::cos(angle);
        double y = centerPos.y + radius * std::sin(angle);
        Vector position = {x, y};
        h_b[i].isDynamic = true;
        h_b[i].mass = EARTH_MASS;
        h_b[i].radius = EARTH_DIA;
        h_b[i].position = position;
        h_b[i].velocity = {0.0, 0.0};
        h_b[i].acceleration = {0.0, 0.0};
    }
    h_b[nBodies - 1].isDynamic = false;
    h_b[nBodies - 1].mass = SUN_MASS;
    h_b[nBodies - 1].radius = SUN_DIA;
    h_b[nBodies - 1].position = centerPos;
    h_b[nBodies - 1].velocity = {0.0, 0.0};
    h_b[nBodies - 1].acceleration = {0.0, 0.0};
}

void BarnesHutCuda::initSpiralBodies()
{

    srand(time(NULL));
    double maxDistance = MAX_DIST;
    double minDistance = MIN_DIST;
    Vector centerPos = {CENTERX, CENTERY};
    for (int i = 0; i < nBodies - 1; ++i)
    {

        double angle = 2 * M_PI * (rand() / (double)RAND_MAX);
        // Generate random distance from center within the given max distance
        double radius = (maxDistance - minDistance) * (rand() / (double)RAND_MAX) + minDistance;

        // Calculate coordinates of the point
        double x = centerPos.x + radius * std::cos(angle);
        double y = centerPos.y + radius * std::sin(angle);
        Vector position = {x, y};

        double distance = sqrt(pow(x - centerPos.x, 2) + pow(y - centerPos.y, 2));
        Vector r = {position.x - centerPos.x, position.y - centerPos.y};
        Vector a = {r.x / distance, r.y / distance};

        // Calculate velocity vector components
        double esc = sqrt((GRAVITY * SUN_MASS) / (distance));
        Vector velocity = {-a.y * esc, a.x * esc};

        h_b[i].isDynamic = true;
        h_b[i].mass = EARTH_MASS;
        h_b[i].radius = EARTH_DIA;
        h_b[i].position = position;
        h_b[i].velocity = velocity;
        h_b[i].acceleration = {0.0, 0.0};
    }
    h_b[nBodies - 1].isDynamic = false;
    h_b[nBodies - 1].mass = SUN_MASS;
    h_b[nBodies - 1].radius = SUN_DIA;
    h_b[nBodies - 1].position = centerPos;
    h_b[nBodies - 1].velocity = {0.0, 0.0};
    h_b[nBodies - 1].acceleration = {0.0, 0.0};
}

void BarnesHutCuda::initCollideGalaxy()
{

    srand(time(NULL));
    double maxDistance = MAX_DIST / 4.0;
    double minDistance = MIN_DIST;
    Vector centerPos = {-NBODY_WIDTH / 6.0, CENTERY};

    int galaxy1 = nBodies / 2;

    for (int i = 0; i < galaxy1 - 1; ++i)
    {

        double angle = 2 * M_PI * (rand() / (double)RAND_MAX);
        // Generate random distance from center within the given max distance
        double radius = (maxDistance - minDistance) * (rand() / (double)RAND_MAX) + minDistance;

        // Calculate coordinates of the point
        double x = centerPos.x + radius * std::cos(angle);
        double y = centerPos.y + radius * std::sin(angle);
        Vector position = {x, y};

        double distance = sqrt(pow(x - centerPos.x, 2) + pow(y - centerPos.y, 2));
        Vector r = {position.x - centerPos.x, position.y - centerPos.y};
        Vector a = {r.x / distance, r.y / distance};

        // Calculate velocity vector components
        double esc = sqrt((GRAVITY * SUN_MASS) / (distance));
        Vector velocity = {-a.y * esc, a.x * esc};

        h_b[i].isDynamic = true;
        h_b[i].mass = EARTH_MASS;
        h_b[i].radius = EARTH_DIA;
        h_b[i].position = position;
        h_b[i].velocity = velocity;
        h_b[i].acceleration = {0.0, 0.0};
    }
    h_b[galaxy1 - 1].isDynamic = true;
    h_b[galaxy1 - 1].mass = SUN_MASS;
    h_b[galaxy1 - 1].radius = SUN_DIA;
    h_b[galaxy1 - 1].position = centerPos;
    h_b[galaxy1 - 1].velocity = {0.0, 0.0};
    h_b[galaxy1 - 1].acceleration = {0.0, 0.0};

    centerPos = {NBODY_WIDTH / 6.0, CENTERY};

    for (int i = galaxy1; i < nBodies - 1; ++i)
    {

        double angle = 2 * M_PI * (rand() / (double)RAND_MAX);
        // Generate random distance from center within the given max distance
        double radius = (maxDistance - minDistance) * (rand() / (double)RAND_MAX) + minDistance;

        // Calculate coordinates of the point
        double x = centerPos.x + radius * std::cos(angle);
        double y = centerPos.y + radius * std::sin(angle);
        Vector position = {x, y};

        double distance = sqrt(pow(x - centerPos.x, 2) + pow(y - centerPos.y, 2));
        Vector r = {position.x - centerPos.x, position.y - centerPos.y};
        Vector a = {r.x / distance, r.y / distance};

        // Calculate velocity vector components
        double esc = sqrt((GRAVITY * SUN_MASS) / (distance));
        Vector velocity = {-a.y * esc, a.x * esc};

        h_b[i].isDynamic = true;
        h_b[i].mass = EARTH_MASS;
        h_b[i].radius = EARTH_DIA;
        h_b[i].position = position;
        h_b[i].velocity = velocity;
        h_b[i].acceleration = {0.0, 0.0};
    }
    h_b[nBodies - 1].isDynamic = true;
    h_b[nBodies - 1].mass = SUN_MASS;
    h_b[nBodies - 1].radius = SUN_DIA;
    h_b[nBodies - 1].position = centerPos;
    h_b[nBodies - 1].velocity = {0.0, 0.0};
    h_b[nBodies - 1].acceleration = {0.0, 0.0};
}

void BarnesHutCuda::setBody(int i, bool isDynamic, double mass, double radius, Vector position, Vector velocity, Vector acceleration)
{
    h_b[i].isDynamic = isDynamic;
    h_b[i].mass = mass;
    h_b[i].radius = radius;
    h_b[i].position = position;
    h_b[i].velocity = velocity;
    h_b[i].acceleration = acceleration;
}

void BarnesHutCuda::initSolarSystem()
{
    setBody(0, true, 5.9740e24, 1.3927e6, {1.4960e11, 0}, {0, 2.9800e4}, {0, 0});
    setBody(1, true, 6.4190e23, 1.3927e6, {2.2790e11, 0}, {0, 2.4100e4}, {0, 0});
    setBody(2, true, 3.3020e23, 1.3927e6, {5.7900e10, 0}, {0, 4.7900e4}, {0, 0});
    setBody(3, true, 4.8690e24, 1.3927e6, {1.0820e11, 0}, {0, 3.5000e4}, {0, 0});
    setBody(4, false, 1.9890e30, 1.3927e6, {CENTERX, CENTERY}, {0, 0}, {0, 0});
}

Body *BarnesHutCuda::getBodies()
{

    return h_b;
}

void BarnesHutCuda::readDeviceBodies()
{
    CHECK_CUDA_ERROR(cudaMemcpy(h_b, d_b, sizeof(Body) * nBodies, cudaMemcpyDeviceToHost));
}

void BarnesHutCuda::setup(int sim)
{
    if (sim == 0)
    {
        initSpiralBodies();
    }
    else if (sim == 1)
    {
        initRandomBodies();
    }
    else if (sim == 2)
    {
        initCollideGalaxy();
    }
    else
    {
        initSolarSystem();
    }

    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, sizeof(Body) * nBodies, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_node, h_node, sizeof(Node) * nNodes, cudaMemcpyHostToDevice));
}
void BarnesHutCuda::update()
{
    resetCUDA();
    computeBoundingBoxCUDA();
    constructQuadTreeCUDA();
    computeForceCUDA();
    CHECK_LAST_CUDA_ERROR();
}



cv::VideoWriter video("nbody.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(WINDOW_WIDTH, WINDOW_HEIGHT));

Vector scaleToWindow(Vector pos)
{

    double scaleX = WINDOW_HEIGHT / NBODY_HEIGHT;
    double scaleY = WINDOW_WIDTH / NBODY_WIDTH;
    return {(pos.x - 0) * scaleX + WINDOW_WIDTH / 2, (pos.y - 0) * scaleY + WINDOW_HEIGHT / 2};
}

void storeFrame(Body *bodies, int n, int id)
{
    cv::Mat image = cv::Mat::zeros(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
    cv::Scalar color; // White color
    int radius;
    for (int i = 0; i < n; i++)
    {
        Vector pos = scaleToWindow(bodies[i].position);
        cv::Point center(pos.x, pos.y);

        // stars will be red and planets will be white
        if (bodies[i].mass >= HBL)
        {
            color = cv::Scalar(0, 0, 255);
            radius = 5;
        }
        else
        {
            color = cv::Scalar(255, 255, 255);
            radius = 1;
        }
        cv::circle(image, center, radius, color, -1);
    }
    video.write(image);
}

bool checkArgs(int nBodies, int sim, int iter)
{

    if (nBodies < 1)
    {
        std::cout << "ERROR: need to have at least 1 body" << std::endl;
        return false;
    }

    if (sim < 0 || sim > 3)
    {
        std::cout << "ERROR: simulation doesn't exist" << std::endl;
        return false;
    }

    if (iter < 1)
    {
        std::cout << "ERROR: need to have at least 1 iteration" << std::endl;
        return false;
    }

    return true;
}

int main(int argc, char **argv)
{
    int nBodies = NUM_BODIES;
    int sim = 0;
    int iters = 300;
    if (argc == 4)
    {
        nBodies = atoi(argv[1]);
        sim = atoi(argv[2]);
        iters = atoi(argv[3]);
    }

    if (!checkArgs(nBodies, sim, iters))
        return -1;

    if (sim == 3)
        nBodies = 5;

    BarnesHutCuda *bh = new BarnesHutCuda(nBodies);
    bh->setup(sim);

    for (int i = 0; i < iters; ++i)
    {
        bh->update();
        bh->readDeviceBodies();
        storeFrame(bh->getBodies(), nBodies, i);
    }

    video.release();
    delete bh;
    return 0;
}
