/*
Eigen Matrix Library test of Coordinate frame transformations
	
Compile with:
	g++ -O3 -std=c++11 CoordinateFrame-Sim.cpp -I$EIGEN_ROOT_DIR -o CoordinateFrame-Sim

	Origin and Camera frame start at the same location, and the Camera starts
	translating in the positive y-direction and rotating about the x-axis

	gx_y means: Transformation of frame Y in the coordinates of X
*/

#include <iostream>
#include <chrono>
#include "lieAlgebras.h"

using namespace std;
using namespace lieAlgebras;

// UTILS
static inline float degrees2radians(float deg)
{
  return deg*(PI/180.0f);
}
static inline float degrees(float radians)
{
    return radians * (180.0f / PI);
}
//Get Euler Angles in degrees: roll, pitch, yaw
void getEulerAngles(Eigen::Vector3f& Ang, const Eigen::Matrix3f& R)
{
    //Output is X,Y,Z
    //Derived from rotation of form: transpose(Y1*X2*Z3)
    // Y(alpha), X(beta), Z(gamma)
    //https://en.wikipedia.org/wiki/Euler_angles
    Ang(0) = -asinf(R(1,2));                          //roll
    float cosA1 = cosf(Ang(0));
    Ang(2) =  atan2(R(1,0)/cosA1, R(1,1)/cosA1);   //pitch
    Ang(1) =  atan2(R(0,2)/cosA1, R(2,2)/cosA1);   //yaw

    Ang(0) = degrees(Ang(0));
    Ang(1) = degrees(Ang(1));
    Ang(2) = degrees(Ang(2));
}
Eigen::Vector3f EA;

//////////
// MAIN //
//////////
int main()
{
	Eigen::Vector3f w0;
	Eigen::Vector3f t0_1, t1_1;
	Eigen::Matrix3f R0, Rc;
	Eigen::Matrix4f g0_1, g1_1;

	//Change in Rotation and Translation
	//At each timestep, the camera frame rotates about y-axis by theta
	float theta = degrees2radians(25.0f);
	w0 << 0.0f, 1.0f, 0.0f;
	w0.normalize();
	w0 *= theta;
	Rc  = RodriguesRotation(w0);

	//At each timestep, the camera translates 0.10 m about an axis, NOT THE ROTATION AXIS
	float rho = 0.1f;
	t0_1 << 0.0, 0.0f, rho;

	//Initialize
	g0_1 << Eigen::Matrix4f::Identity();
	g1_1 << Eigen::Matrix4f::Identity();

	for (int t=0; t<9; t++)
	{
		/* 
		The Camera Frame assumes canonical xyz, but these are rotating at each timestep relative to the Origin
		Therefore, to maintain a constant translation along the y-axis, this must be corrected by the current rotation relative to the origin
		*/
		R0   = g0_1.block<3,3>(0,0);
		
		//WORKS
		t1_1 = R0.transpose()*t0_1;

		//TESTING
		//t1_1 = t0_1;
		
		g1_1 << Rc, t1_1, Eigen::Vector3f::Zero().transpose(), 1.0f;

		//Update Camera position in coordinates of the Origin
		g0_1 *= g1_1;
		
		//TESTING
		//g0_1 *= (g0_1.inverse()*g1_1*g0_1);

		std::cout << "Change In Camera Position in Camera Coords:\n" << g1_1 << std::endl;
		std::cout << "Camera Position in Origin Coords:\n" << g0_1 << std::endl;
		getEulerAngles(EA, g0_1.block<3,3>(0,0));
		std::cout << "Origin Coords Euler Angles: " << EA.transpose() << std::endl << std::endl;
	}
}