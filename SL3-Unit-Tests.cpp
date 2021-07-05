/*
Eigen Matrix Library tools for operating with Homographies in SL(3)
	
	RE: recovered R and t using the 8 solution method of Faugeras
		-The constraint that n is [0 0 1] reduces it down to 4 solutions
		-Comparison with IMU initial rotation reduces it down to 2 solutions
		-Comparing cost of remaining 2 solutions determines unique solution?
			-It would be better to come up with a different constraint

Compile with:
	g++ -O3 -std=c++11 SL3-Unit-Tests.cpp -I$EIGEN_ROOT_DIR -o SL3-Unit-Tests

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
	Eigen::Vector3f w, t, t_chk;
	Eigen::Matrix3f R, R_chk, H0, H1;
	vMatrix3f vR_chk;
	vVector3f vt_chk, vn_chk;
	Vector8f x, x_chk;

	//Generate a random rotation axis-angle and translation
	w = Eigen::Vector3f::Random(3, 1);
	t = Eigen::Vector3f::Random(3, 1);
	t.normalize();
	//Small Translations are much more accurately estimated than large ones
	t *= 0.1f;

	//Derive rotation matrix
	R = RodriguesRotation(w); 
	std::cout << "Reference Rotation Matrix:" << std::endl;
	std::cout << R << std::endl << std::endl;
	std::cout << "Reference Translation Vector:" << std::endl;
	std::cout << t.transpose() << std::endl << std::endl;

	//Compose Reference Homography from R and t according to ESM
	H0 = homographyESM(R,t);
	std::cout << "H0:" << std::endl;
	std::cout << H0 << std::endl;

	//Extract sl(3) vector
	x = logMatExpSL3(H0);
	std::cout << "sl(3) from H0: " << x.transpose() << std::endl;

	//Reconstruct Homography from sl(3)
	expMapSL3(H1, x);
	std::cout << "H1:" << std::endl;
	std::cout << H1 << std::endl;

	//Extract sl(3) vector
	x_chk = logMatExpSL3(H1);
	std::cout << "sl(3) from H1: " << x_chk.transpose() << std::endl;

	float H_diff = (H0-H1).array().abs().sum();
	std::cout << "\nSum of Absolute differences between H0 and H1: " << H_diff << std::endl << std::endl;

	//Create Timers
	auto start = std::chrono::steady_clock::now();
	auto end   = std::chrono::steady_clock::now();
	std::chrono::duration<double, std::milli> elapsed;

	int N = 1000;
	for (int i=0; i<N;i++)
	{
		start = std::chrono::steady_clock::now();
		//Recover Rotation and translation from H1 and compare against reference R
		extractRTfromH(vR_chk, vt_chk, vn_chk, H1, R); 
		end   = std::chrono::steady_clock::now();
		elapsed += end-start;
	}
	std::cout << N << " iterations of RT extraction with an average time per iteration of " << elapsed.count()/N << " ms" << std::endl;

	std::cout << "\nTop 2 Rotation/Translation Pairs:" << std::endl;
	for (int i=0; i<2;i++)
	{
		std::cout << "\nRTN tuple " << i << std::endl;
		std::cout << vR_chk[i] << std::endl<< std::endl;
		std::cout << vt_chk[i].transpose() << std::endl<< std::endl;
		std::cout << vn_chk[i].transpose() << std::endl<< std::endl;
		std::cout << "Reconstructed H:\n" << homographyESM(vR_chk[i], vt_chk[i], vn_chk[i]) <<std::endl;

		std::cout << "Percentage Error between reference and target Trans: " << (100.0f*((t-vt_chk[i]).array().abs().sum())/t.array().abs().sum()) << std::endl;
	}
}
