#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include<memory>
#include<algorithm>
#include <GLFW/glfw3.h>


using namespace std;
using namespace std::chrono;



#define NUM_BODIES 100
#define WINDOW_WIDTH 1600
#define WINDOW_HEIGHT 1600
#define NBODY_WIDTH 10.0e11
#define NBODY_HEIGHT 10.0e11
#define CENTERX 0
#define CENTERY 0
#define GRAVITY 6.67E-11
#define COLLISION_TH 1.0e10
#define MIN_DIST 2.0e10
#define MAX_DIST 5.0e11
#define SUN_MASS 1.9890e30
#define SUN_DIA 1.3927e6
#define EARTH_MASS 5.974e24
#define EARTH_DIA 12756


class Body
{
public:
    bool isDynamic;
    double mass;
    double radius;
    Vector position;
    Vector velocity;
    Vector acceleration;

    Body(double m, double r, Vector p, Vector v, Vector a, bool d = true) : mass(m), radius(r), position(p), velocity(v), acceleration(a), isDynamic(d) {}
    friend std::ostream &operator<<(std::ostream &os, const Body &b);
};


std::ostream &operator<<(std::ostream &os, const Body &b)
{

    os << "Body(" << b.mass << "," << b.radius << "," << b.position << "," << b.velocity << "," << b.acceleration << "," << b.isDynamic;
    return os;
}



class Algorithm
{
protected:
    std::vector<std::shared_ptr<Body> > &bodies;
    int nBodies;

public:
    Algorithm(std::vector<std::shared_ptr<Body> > &bs, int n);
    virtual ~Algorithm() = default;
    virtual void update() = 0;
};


Algorithm::Algorithm(std::vector<std::shared_ptr<Body>> &bs, int n) : bodies(bs), nBodies(n)
{
}



class DirectSum : public Algorithm
{
    const double epsilon = 0.5;
    const double dt = 25000.0;

    void calculateAcceleration();
    void calculateVelocity();
    void calculatePosition();
    bool isCollide(Body b1, Body b2);

public:
    DirectSum(std::vector<std::shared_ptr<Body> > &bs);
    void update() override;
};

class NBody
{
    std::unique_ptr<Algorithm> alg;
    int nBodies;
    void initRandomBodies();
    void initSpiralBodies();
    void initSolarSystem();

public:
    std::vector<std::shared_ptr<Body> > bodies;
    NBody(int n, int a, int s);
    void update();
};



class BarnesHut;

class QuadTree
{

    Vector topLeft;
    Vector botRight;
    Vector centerMass;
    double totalMass;
    bool isLeaf;
    std::shared_ptr<Body> b;
    std::unique_ptr<QuadTree> topLeftTree;
    std::unique_ptr<QuadTree> topRightTree;
    std::unique_ptr<QuadTree> botLeftTree;
    std::unique_ptr<QuadTree> botRightTree;

public:
    QuadTree();
    QuadTree(Vector topL, Vector botR);
    ~QuadTree();
    void insert(std::shared_ptr<Body> n);
    std::shared_ptr<Body> search(Vector point);
    bool inBoundary(Vector point);
    double getWidth();
    Vector getCenter();
    int getQuadrant(Vector pos);
    friend void updateCenterMass(std::unique_ptr<QuadTree> &root);
    friend void traverse(std::unique_ptr<QuadTree> &root);
    friend double getTotalMass(std::unique_ptr<QuadTree> &root);

    friend class BarnesHut;
};


class BarnesHut : public Algorithm
{
    const double epsilon = 0.5;
    const double dt = 25000.0;
    const double theta = 0.5;

    std::unique_ptr<QuadTree> quadTree;
    void constructQuadTree();
    void computeCenterMass();
    void calculateForceHelper(std::unique_ptr<QuadTree> &root, std::shared_ptr<Body> body);
    void computeBoundingBox();
    void calculateForce(std::shared_ptr<Body> b);
    void calculateAcceleration();
    void calculateVelocity();
    void calculatePosition();
    bool isCollide(Body b1, Body b2);
    bool isCollide(Body b, Vector cm);

public:
    BarnesHut(std::vector<std::shared_ptr<Body>> &bs);
    void update() override;
};



NBody::NBody(int n, int a, int s) : nBodies(n)
{

    if (s == 0)
    {
        initSpiralBodies();
    }
    else if (s == 1)
    {
        initRandomBodies();
    }
    else
    {
        nBodies = 5;
        initSolarSystem();
    }

    if (a == 0)
    {
        alg = std::make_unique<DirectSum>(bodies);
    }
    else
    {
        alg = std::make_unique<BarnesHut>(bodies);
    }
}

void NBody::initRandomBodies()
{
    srand(time(NULL));

    bodies.clear();
    double maxDistance = MAX_DIST;
    double minDistance = MIN_DIST;
    Vector centerPos(CENTERX, CENTERY);
    for (int i = 0; i < nBodies - 1; ++i)
    {
        double angle = 2 * M_PI * (rand() / (double)RAND_MAX);
        // Generate random distance from center within the given max distance
        double radius = (maxDistance - minDistance) * (rand() / (double)RAND_MAX) + minDistance;

        // Calculate coordinates of the point
        double x = centerPos.x + radius * std::cos(angle);
        double y = centerPos.y + radius * std::sin(angle);
        Vector position(x, y);
        bodies.push_back(std::make_shared<Body>(EARTH_MASS, EARTH_DIA, position, Vector(0, 0), Vector(0, 0)));
    }
    bodies.push_back(std::make_shared<Body>(SUN_MASS, SUN_DIA, centerPos, Vector(0, 0), Vector(0, 0), false));
}

void NBody::initSpiralBodies()
{
    srand(time(NULL));

    bodies.clear();
    double maxDistance = MAX_DIST;
    double minDistance = MIN_DIST;
    Vector centerPos(CENTERX, CENTERY);
    for (int i = 0; i < nBodies - 1; ++i)
    {

        double angle = 2 * M_PI * (rand() / (double)RAND_MAX);
        // Generate random distance from center within the given max distance
        double radius = (maxDistance - minDistance) * (rand() / (double)RAND_MAX) + minDistance;

        // Calculate coordinates of the point
        double x = centerPos.x + radius * std::cos(angle);
        double y = centerPos.y + radius * std::sin(angle);

        Vector position(x, y);
        double distance = position.getDistance(centerPos);
        Vector r = position - centerPos;
        Vector a = r / distance;

        // Calculate velocity vector components
        double esc = sqrt((GRAVITY * SUN_MASS) / (distance));
        Vector velocity(-a.y * esc, a.x * esc);

        bodies.push_back(std::make_shared<Body>(EARTH_MASS, EARTH_DIA, position, velocity, Vector(0, 0)));
    }

    bodies.push_back(std::make_shared<Body>(SUN_MASS, SUN_DIA, centerPos, Vector(0, 0), Vector(0, 0), false));
}

void NBody::initSolarSystem()
{
    bodies.clear();

    bodies.push_back(std::make_shared<Body>(5.9740e24, 1.3927e6, Vector(1.4960e11, 0), Vector(0, 2.9800e4), Vector(0, 0)));
    bodies.push_back(std::make_shared<Body>(6.4190e23, 1.3927e6, Vector(2.2790e11, 0), Vector(0, 2.4100e4), Vector(0, 0)));
    bodies.push_back(std::make_shared<Body>(3.3020e23, 1.3927e6, Vector(5.7900e10, 0), Vector(0, 4.7900e4), Vector(0, 0)));
    bodies.push_back(std::make_shared<Body>(4.8690e24, 1.3927e6, Vector(1.0820e11, 0), Vector(0, 3.5000e4), Vector(0, 0)));
    bodies.push_back(std::make_shared<Body>(1.9890e30, 1.3927e6, Vector(CENTERX, CENTERY), Vector(0, 0), Vector(0, 0), false));
}

void NBody::update()
{
    alg->update();
}






DirectSum::DirectSum(std::vector<std::shared_ptr<Body>> &bs) : Algorithm(bs, bs.size()) {}

void DirectSum::calculateAcceleration()
{

    for (int i = 0; i < nBodies; ++i)
    {
        Body &bi = *bodies[i];
        bi.acceleration = Vector(0, 0);
        Vector force(0, 0);
        for (int j = 0; j < nBodies; ++j)
        {
            Body &bj = *bodies[j];
            if (i != j && bi.isDynamic && !isCollide(bi, bj))
            {

                Vector rij = bj.position - bi.position;
                double r = sqrt((rij.x * rij.x) + (rij.y * rij.y) + (epsilon * epsilon));
                double f = (GRAVITY * bi.mass * bj.mass) / (r * r * r + (epsilon * epsilon));
                force += rij * f;
            }
        }

        bi.acceleration += (force / bi.mass);
    }
}

void DirectSum::calculateVelocity()
{

    for (auto &body : bodies)
    {
        body->velocity += (body->acceleration * dt);
    }
}

void DirectSum::calculatePosition()
{
    double boundaryWidth = NBODY_WIDTH, boundaryHeight = NBODY_HEIGHT;

    // check if body is at boundary
    for (auto &body : bodies)
    {
        body->position += body->velocity * dt;
    }
}

bool DirectSum::isCollide(Body b1, Body b2)
{
    return b1.radius + b2.radius + COLLISION_TH >= b1.position.getDistance(b2.position);
}

void DirectSum::update()
{
    calculateAcceleration();
    calculateVelocity();
    calculatePosition();
}







QuadTree::QuadTree() : QuadTree(Vector(-1, -1), Vector(-1, -1))
{
}
QuadTree::QuadTree(Vector topL, Vector botR)
{
    topLeft = topL;
    botRight = botR;
    centerMass = Vector(-1, -1);
    totalMass = 0.0;
    isLeaf = true;
    b = nullptr;
    topLeftTree = nullptr;
    topRightTree = nullptr;
    botLeftTree = nullptr;
    botRightTree = nullptr;
}
QuadTree::~QuadTree()
{
}

int QuadTree::getQuadrant(Vector pos)
{

    if ((topLeft.x + botRight.x) / 2.0 >= pos.x)
    {
        // Indicates topLeftTree
        if ((topLeft.y + botRight.y) / 2.0 <= pos.y)
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
        if ((topLeft.y + botRight.y) / 2.0 <= pos.y)
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
void QuadTree::insert(std::shared_ptr<Body> body)
{
    if (body == nullptr)
        return;

    if (!inBoundary(body->position))
    {
        std::cout << "ERROR: body out of bound" << std::endl;
        return;
    }

    // If node x does not contain a body, put the new body here.
    if (b == nullptr && isLeaf)
    {
        b = body;
        return;
    }

    if (b != nullptr)
    {
        if (b == body)
        {
            b = nullptr;
        }
        else
        {
            this->insert(b);
        }
    }

    isLeaf = false;

    int q = getQuadrant(body->position);

    if (q == 1)
    {
        if (topRightTree == nullptr)
            topRightTree = std::make_unique<QuadTree>(Vector((topLeft.x + botRight.x) / 2,
                                                             topLeft.y),
                                                      Vector(botRight.x,
                                                             (topLeft.y + botRight.y) / 2));

        topRightTree->insert(body);
    }
    else if (q == 2)
    {
        if (topLeftTree == nullptr)
            topLeftTree = std::make_unique<QuadTree>(Vector(topLeft.x, topLeft.y), Vector((topLeft.x + botRight.x) / 2,
                                                                                          (topLeft.y + botRight.y) / 2));

        topLeftTree->insert(body);
    }
    else if (q == 3)
    {
        if (botLeftTree == nullptr)
            botLeftTree = std::make_unique<QuadTree>(Vector(topLeft.x,
                                                            (topLeft.y + botRight.y) / 2),
                                                     Vector((topLeft.x + botRight.x) / 2,
                                                            botRight.y));

        botLeftTree->insert(body);
    }
    else
    {
        if (botRightTree == nullptr)
            botRightTree = std::make_unique<QuadTree>(Vector((topLeft.x + botRight.x) / 2,
                                                             (topLeft.y + botRight.y) / 2),
                                                      Vector(botRight.x, botRight.y));

        botRightTree->insert(body);
    }
}
std::shared_ptr<Body> QuadTree::search(Vector p)
{

    if (!inBoundary(p))
        return nullptr;

    if (b != nullptr)
        return b;

    int q = getQuadrant(p);

    if (q == 1)
    {
        if (topRightTree == nullptr)
            return nullptr;
        return topRightTree->search(p);
    }
    else if (q == 2)
    {
        if (topLeftTree == nullptr)
            return nullptr;
        return topLeftTree->search(p);
    }
    else if (q == 3)
    {
        if (botLeftTree == nullptr)
            return nullptr;
        return botLeftTree->search(p);
    }
    else
    {
        if (botRightTree == nullptr)
            return nullptr;
        return botRightTree->search(p);
    }
}
double getTotalMass(std::unique_ptr<QuadTree> &root)
{
    if (!root)
        return 0.0;
    return root->totalMass;
}
void updateCenterMass(std::unique_ptr<QuadTree> &root)
{
    if (!root)
        return;
    if (root->b)
    {
        root->totalMass = root->b->mass;
        root->centerMass = root->b->position;
        return;
    }

    updateCenterMass(root->topLeftTree);
    updateCenterMass(root->topRightTree);
    updateCenterMass(root->botLeftTree);
    updateCenterMass(root->botRightTree);

    double totalChildMass = getTotalMass(root->topLeftTree) + getTotalMass(root->topRightTree) + getTotalMass(root->botLeftTree) + getTotalMass(root->botRightTree);

    double totalCenterMassX = 0.0, totalCenterMassY = 0.0;
    if (root->topLeftTree)
    {
        totalCenterMassX += root->topLeftTree->centerMass.x * root->topLeftTree->totalMass;
        totalCenterMassY += root->topLeftTree->centerMass.y * root->topLeftTree->totalMass;
    }
    if (root->topRightTree)
    {
        totalCenterMassX += root->topRightTree->centerMass.x * root->topRightTree->totalMass;
        totalCenterMassY += root->topRightTree->centerMass.y * root->topRightTree->totalMass;
    }
    if (root->botLeftTree)
    {
        totalCenterMassX += root->botLeftTree->centerMass.x * root->botLeftTree->totalMass;
        totalCenterMassY += root->botLeftTree->centerMass.y * root->botLeftTree->totalMass;
    }
    if (root->botRightTree)
    {
        totalCenterMassX += root->botRightTree->centerMass.x * root->botRightTree->totalMass;
        totalCenterMassY += root->botRightTree->centerMass.y * root->botRightTree->totalMass;
    }
    root->totalMass = totalChildMass;
    root->centerMass = Vector(totalCenterMassX / totalChildMass, totalCenterMassY / totalChildMass);
}

bool QuadTree::inBoundary(Vector p)
{
    return (p.x >= topLeft.x && p.x <= botRight.x && p.y <= topLeft.y && p.y >= botRight.y);
}
double QuadTree::getWidth()
{
    return botRight.x - topLeft.x;
}
Vector QuadTree::getCenter()
{
    return centerMass;
}
void traverse(std::unique_ptr<QuadTree> &root)
{
    if (!root)
        return;

    if (root->b)
        std::cout << root->topLeft << " " << root->botRight << " " << root->totalMass << " " << root->centerMass << " " << *root->b << std::endl;
    else
        std::cout << root->topLeft << " " << root->botRight << " " << root->totalMass << " " << root->centerMass << " " << std::endl;

    traverse(root->topLeftTree);
    traverse(root->topRightTree);
    traverse(root->botLeftTree);
    traverse(root->botRightTree);
}







BarnesHut::BarnesHut(std::vector<std::shared_ptr<Body>> &bs) : Algorithm(bs, bs.size()) {}

void BarnesHut::computeBoundingBox()
{
    Vector topLeft = Vector(INFINITY, -INFINITY), botRight = Vector(-INFINITY, INFINITY);
    for (auto &body : bodies)
    {
        topLeft.x = fminf(topLeft.x, body->position.x - 1.0e10);
        topLeft.y = fmaxf(topLeft.y, body->position.y + 1.0e10);
        botRight.x = fmaxf(botRight.x, body->position.x + 1.0e10);
        botRight.y = fminf(botRight.y, body->position.y - 1.0e10);
    }

    quadTree = std::make_unique<QuadTree>(topLeft, botRight);
}

void BarnesHut::constructQuadTree()
{
    for (auto &body : bodies)
    {
        quadTree->insert(body);
    }
}

void BarnesHut::computeCenterMass()
{
    updateCenterMass(quadTree);
}

void BarnesHut::calculateForceHelper(std::unique_ptr<QuadTree> &root, std::shared_ptr<Body> body)
{
    if (!root)
        return;

    if (root->b)
    {

        Body &bi = *body, &bj = *root->b;
        if (isCollide(bi, bj) || root->b == body)
            return;

        Vector rij = bj.position - bi.position;

        double r = sqrt((rij.x * rij.x) + (rij.y * rij.y) + (epsilon * epsilon));
        Vector force = rij * ((GRAVITY * bi.mass * bj.mass) / (r * r * r + (epsilon * epsilon)));
        bi.acceleration += (force / bi.mass);
        return;
    }

    double sd = root->getWidth() / body->position.getDistance(root->centerMass);
    if (sd < theta)
    {
        Body &bi = *body;
        Vector rij = root->centerMass - bi.position;
        if (!isCollide(bi, root->centerMass))
        {
            double r = sqrt((rij.x * rij.x) + (rij.y * rij.y) + (epsilon * epsilon));
            Vector force = rij * ((GRAVITY * bi.mass * root->totalMass) / (r * r * r + (epsilon * epsilon)));
            bi.acceleration += (force / bi.mass);
        }

        return;
    }

    calculateForceHelper(root->topLeftTree, body);
    calculateForceHelper(root->topRightTree, body);
    calculateForceHelper(root->botLeftTree, body);
    calculateForceHelper(root->botRightTree, body);
}

void BarnesHut::calculateForce(std::shared_ptr<Body> b)
{
    calculateForceHelper(quadTree, b);
}

void BarnesHut::calculateAcceleration()
{
    for (auto &body : bodies)
    {
        if (body->isDynamic)
        {
            body->acceleration = Vector(0, 0);
            calculateForce(body);
        }
    }
}

void BarnesHut::calculateVelocity()
{
    for (auto &body : bodies)
    {
        body->velocity += body->acceleration * dt;
    }
}

void BarnesHut::calculatePosition()
{
    // check if body is at boundary
    for (auto &body : bodies)
    {
        body->position += body->velocity * dt;
    }
}

bool BarnesHut::isCollide(Body b1, Body b2)
{

    return b1.radius + b2.radius + COLLISION_TH > b1.position.getDistance(b2.position);
}

bool BarnesHut::isCollide(Body b, Vector cm)
{
    return b.radius * 2 + COLLISION_TH > b.position.getDistance(cm);
}

void BarnesHut::update()
{
    computeBoundingBox();
    constructQuadTree();
    computeCenterMass();
    calculateAcceleration();
    calculateVelocity();
    calculatePosition();
}



Vector scaleToWindow(Vector pos)
{

     double scaleX = WINDOW_HEIGHT / NBODY_HEIGHT;
     double scaleY = WINDOW_WIDTH / NBODY_WIDTH;
     return Vector((pos.x - 0) * scaleX + WINDOW_WIDTH / 2, (pos.y - 0) * scaleY + WINDOW_HEIGHT / 2);
}

void drawDots(NBody &nb)
{

     glColor3f(1.0, 1.0, 1.0); // set drawing color to white

     for (auto &body : nb.bodies)
     {
          glPointSize(5);     // set point size to 5 pixels
          glBegin(GL_POINTS); // start drawing points
          Vector pos = scaleToWindow(body->position);
          glVertex2f(pos.x, pos.y);
          glEnd(); // end drawing points
     }
}

bool checkArgs(int nBodies, int alg, int sim)
{

     if (nBodies < 1)
     {
          std::cout << "ERROR: need to have at least 1 body" << std::endl;
          return false;
     }

     if (alg < 0 && alg > 1)
     {
          std::cout << "ERROR: algorithm doesn't exist" << std::endl;
          return false;
     }

     if (sim < 0 || sim > 2)
     {
          std::cout << "ERROR: simulation doesn't exist" << std::endl;
          return false;
     }

     return true;
}

int main(int argc, char **argv)
{
     int nBodies = NUM_BODIES;
     int alg = 0;
     int sim = 0;
     if (argc == 4)
     {
          nBodies = atoi(argv[1]);
          alg = atoi(argv[2]);
          sim = atoi(argv[3]);
     }

     if (!checkArgs(nBodies, alg, sim))
          return -1;

     
     NBody nb(nBodies, alg, sim);
     
     auto start = std::chrono::high_resolution_clock::now();

     nb.update();

     auto end = std::chrono::high_resolution_clock::now();

     std::chrono::duration<double> duration = end - start;

     std::cout << "Iteration took " << duration.count() << " seconds." << std::endl;




     // // initialize GLFW
     // if (!glfwInit())
     //      return -1;
     // GLFWwindow *window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "N-Body Simulation CPU", NULL, NULL); // create window
     // if (!window)
     // {
     //      glfwTerminate();
     //      return -1;
     // }
     // glfwMakeContextCurrent(window); // set context to current window

     // glClearColor(0.0, 0.0, 0.0, 1.0); // set background color to black
     // glMatrixMode(GL_PROJECTION);      // set up projection matrix
     // glLoadIdentity();
     // glOrtho(0.0f, WINDOW_WIDTH, WINDOW_HEIGHT, 0.0f, -1.0f, 1.0f);
     // while (!glfwWindowShouldClose(window)) // main loop
     // {
     //      glClear(GL_COLOR_BUFFER_BIT); // clear the screen
     //      nb.update();
     //      drawDots(nb);
     //      glfwSwapBuffers(window); // swap front and back buffers
     //      glfwPollEvents();        // poll for events
     // }

     // glfwTerminate(); // terminate GLFW

     return 0;
}