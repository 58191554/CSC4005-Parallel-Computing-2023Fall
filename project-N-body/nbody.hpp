#ifndef BODY_H
#define BODY_H

#include <vector>
#include <iostream>

class Body {
private:
    float mass;
    std::vector<float> velocity;
    std::vector<float> acceleration;
    std::vector<float> positionVector;
    int id;

public:
    // Constructor with default values
    Body(float _mass = 0.0, 
         const std::vector<float>& _velocity = {0.0, 0.0},
         const std::vector<float>& _acceleration = {0.0, 0.0},
         const std::vector<float>& _positionVector = {0.0, 0.0}, 
         int _id = 0);

    // Getter and setter methods for each attribute
    float getMass() const;
    void setMass(float _mass);

    const std::vector<float>& getVelocity() const;
    void setVelocity(const std::vector<float>& _velocity);

    const std::vector<float>& getAcceleration() const;
    void setAcceleration(const std::vector<float>& _acceleration);

    const std::vector<float>& getPositionVector() const;
    void setPositionVector(const std::vector<float>& _positionVector);

    int getId() const;
    void setId(int _id);
};

#endif
