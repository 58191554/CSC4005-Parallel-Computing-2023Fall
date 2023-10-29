#include "nbody.hpp"

// Constructor with default values
Body::Body(float _mass, 
            const std::vector<float>& _velocity, 
            const std::vector<float>& _acceleration,
            const std::vector<float>& _positionVector, 
            int _id)
            : mass(_mass), 
            velocity(_velocity), 
            acceleration(_acceleration),
            positionVector(_positionVector), 
            id(_id) {}

// Getter and setter methods for each attribute
float Body::getMass() const {
    return mass;
}

void Body::setMass(float _mass) {
    mass = _mass;
}

const std::vector<float>& Body::getVelocity() const {
    return velocity;
}

void Body::setVelocity(const std::vector<float>& _velocity) {
    velocity = _velocity;
}

const std::vector<float>& Body::getAcceleration() const {
    return acceleration;
}

void Body::setAcceleration(const std::vector<float>& _acceleration) {
    acceleration = _acceleration;
}

const std::vector<float>& Body::getPositionVector() const {
    return positionVector;
}

void Body::setPositionVector(const std::vector<float>& _positionVector) {
    positionVector = _positionVector;
}

int Body::getId() const {
    return id;
}

void Body::setId(int _id) {
    id = _id;
}
