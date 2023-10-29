#include "nbody.hpp"
#include <iostream>
#include <vector>

// void simulate

int main(int argc, char ** argv) {
    // Create an array of Body objects to represent the celestial bodies

    const int step_num = atoi(argv[1]);

    int n_body_num = 4;
    std::vector<Body> celestialBodies;

    // Initialize the celestial bodies with some values
    celestialBodies.push_back(Body(1.0, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, 0));
    celestialBodies.push_back(Body(2.0, {0.0, 0.0}, {0.0, 0.0}, {0.0, 1.0}, 1));
    celestialBodies.push_back(Body(1.0, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}, 2));
    celestialBodies.push_back(Body(2.0, {0.0, 0.0}, {0.0, 0.0}, {0.0, -1.0}, 3));

    // allocate the output image in memory
    float ** image_arr = (float**)malloc(step_num*sizeof(float**));
    for(int t = 0; t < step_num; ++t){
        image_arr[t] = (float*)malloc(2*n_body_num*sizeof(float));
    }
    auto image0 = image_arr[0];
    for(int i = 0; i < n_body_num; ++i){
        auto pos = celestialBodies[i].getPositionVector();
        image0[2*i] = pos[0];
        image0[2*i+1] = pos[1];
    }

    // Simulate the N-body problem (update positions, velocities, etc.)
    for(int t = 0; t < step_num; ++t){

    }    

    // output the array of image matrices

    // Print the final state of the celestial bodies
    for (const Body& body : celestialBodies) {
        std::cout << "Body ID: " << body.getId() << std::endl;
        std::cout << "Mass: " << body.getMass() << std::endl;
        std::cout << "Position: (" << body.getPositionVector()[0] << ", " << body.getPositionVector()[1] << ", " << body.getPositionVector()[2] << ")" << std::endl;
        std::cout << "Velocity: (" << body.getVelocity()[0] << ", " << body.getVelocity()[1] << ", " << body.getVelocity()[2] << ")" << std::endl;
        std::cout << "Acceleration: (" << body.getAcceleration()[0] << ", " << body.getAcceleration()[1] << ", " << body.getAcceleration()[2] << ")" << std::endl;
        std::cout << std::endl;
    }

    return 0;
}
