/*
Eigen Matrix Library tests of Adjoint equivalences

MITRM 4.2 Rigid body velocity pg 53	

Compile with:
	g++ -O3 -std=c++11 Adjoint-Tests.cpp -I$EIGEN_ROOT_DIR -o Adjoint-Tests

It is assumed that g0 is the transformation from the standard, world, basis to the basis of 
g1 and g1 is a subsequent transformation. g1 is initially expressed in the basis of g1. If g1
were expressed in the basis of g0 and we wished to change the basis to g1, then the transformation
would be:

g1' = g0.inverse()*g1*g0

But because g1 is expressed in the coordinates of g1, and we wish to change the basis to g0, the
transformation is:

g1' = g0*g1*g0.inverse()

The latter is what the adjoint is doing: it re-expresses a twist in the coordinates of b into 
a twist in the coordinates of a via g_ab i.e. Va = g_ab*Vb*g_ba.inverse()

*/


#include <iostream>
#include <chrono>
#include "lieAlgebras.h"

using namespace std;
using namespace lieAlgebras;

//////////
// MAIN //
//////////
int main()
{
	//Set random seed
	srand((unsigned int) time(0));

	//Our friends for this exercise
	Eigen::Vector3f w0, v0, t0, w1, v1, t1, w2_hat, v2_hat, t2_hat;
	Eigen::Matrix3f R0, R1, R2_hat;
	Eigen::Matrix4f g0, g1, g2, g2_hat;
	Matrix6f Adjg0;

	//Generate a random rotation axis-angle and translation
	v0 = Eigen::Vector3f::Random(3, 1);
	w0 = Eigen::Vector3f::Random(3, 1);
	v1 = Eigen::Vector3f::Random(3, 1);
	w1 = Eigen::Vector3f::Random(3, 1);

	//Compose g0 and g1
	expMapSE3(R0, t0, v0, w0);
	g0 << R0, t0, Eigen::Vector3f::Zero().transpose(), 1.0f;

	expMapSE3(R1, t1, v1, w1);
	g1 << R1, t1, Eigen::Vector3f::Zero().transpose(), 1.0f;

	std::cout << "G0:\n" << g0 << std::endl;
	std::cout << "G1:\n" << g1 << std::endl;

	//Change of Basis
	g2 = g0*g1*g0.inverse();

	//Adjoint Change of Basis
	Adjg0 = Adjoint(R0, t0);
	Vector6f tmp0, tmp1;
	tmp1 << v1, w1;

	//ADJOINT
	tmp0   = Adjg0*tmp1;
	v2_hat = tmp0.head(3);
	w2_hat = tmp0.tail(3);

	expMapSE3(R2_hat, t2_hat, v2_hat, w2_hat);
	g2_hat << R2_hat, t2_hat, Eigen::Vector3f::Zero().transpose(), 1.0f;

	std::cout << "G2:\n" << g2 << std::endl;
	std::cout << "G2_hat:\n" << g2_hat << std::endl;

	std::cout << "\nSum of Absolute differences between g2 and g2_hat: " << (g2-g2_hat).array().abs().sum() << std::endl << std::endl;
}
