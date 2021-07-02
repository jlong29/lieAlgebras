/*
Eigen Matrix Library example of the Rodrigues' equation for the matrix exponential

Compile with:
	g++ SO3-SE3-Unit-Tests.cpp -I$EIGEN_ROOT_DIR -o SO3-SE3-Unit-Tests

	A handy website to check rotations against:
	https://www.andre-gaschler.com/rotationconverter/

*/

#include <iostream>
#include "lieAlgebras.h"

using namespace std;
using namespace lieAlgebras;

// UTILS
float degrees2radians(float deg)
{
  return deg*(PI/180.0f);
}

//////////
// MAIN //
//////////
int main()
{
	//Test Inputs
	float angles[3] = { 10.0f, 24.0f, 0.5f };
	float trans[3]  = { 0.1f, 0.05f, 0.03f };

	Eigen::Vector3f EA, EA_chk, w, w_chk;
	Eigen::Matrix3f R, R_chk;

	// SO(3)
	cout << "//////////////////\n";
	cout << "// SO(3) CHECKS //\n";
	cout << "//////////////////\n\n";

	EA << degrees2radians(angles[0]), degrees2radians(angles[1]), degrees2radians(angles[2]);
	cout << "Reference Euler Angles:" << endl;
	cout << EA.transpose() << endl << endl;

	//Create Rotation matrix from Euler Angles
	composeRotationFromEuler(EA, R);
	cout << "Reference Euler Rotation Matrix:" << endl;
	cout << R << endl << endl;

	//Recover Euler Angles from Rotation Matrix
	getEulerAngles(R, EA_chk);
	cout << "Estimated Euler Angles:" << endl;
	cout << EA_chk.transpose() << endl << endl;

	// Check if original Euler Angles were recovered
	cout << "Difference between Estimated and Reference Euler Angles: " << (EA-EA_chk).norm() << endl << endl;

	//Extract Axis+Angle from Euler Rotation Matrix
	w = logMatExpR(R);
	cout << "Estimated Axis+Angle from Euler Matrix:" << endl;
	cout << w.transpose() << endl << endl;

	//Create Rotation Matrix from Axis+Angle
	R_chk = RodriguesRotation(w);
	cout << "Estimated Axis+Angle Rotation Matrix:" << endl;
	cout << R_chk << endl << endl;

	//Check Axis+Angle Rotation Against Euler Angle Rotation
	cout << "Difference between Euler and Axis+Angle Rotation Matrices: " << (R*R_chk.transpose()).trace()-3.0f << endl << endl;

	//Recover Euler Angles from Rotation Matrix
	getEulerAngles(R_chk, EA_chk);
	cout << "Euler Angles estimated from Axis+Angle Rotation Matrix:" << endl;
	cout << EA_chk.transpose() << endl << endl;

	// Check if original Euler Angles were recovered
	cout << "Difference between Estimated and Reference Euler Angles: " << (EA-EA_chk).norm() << endl << endl;

	//Extract Axis+Angle from Axis+Angle Rotation Matrix
	w_chk = logMatExpR(R_chk);
	cout << "Estimated Axis+Angle from Axis+Angle Matrix:" << endl;
	cout << w_chk.transpose() << endl << endl;

	//Check Extracted Axis+Angle against original
	cout << "Difference between Axis+Angle Representations: " << (w-w_chk).norm() << endl << endl;

	// SE(3)
	cout << "//////////////////\n";
	cout << "// SE(3) CHECKS //\n";
	cout << "//////////////////\n\n";
	Eigen::Vector3f T, T_chk, u, u_chk;
	Eigen::Matrix3f V, Vinv;

	T << trans[0], trans[1], trans[2];

	//Map from SE(3) to se(3)
	logMatExpSE3(u, w_chk, R, T);
	cout << "SE(3) to se(3): u: " << u.transpose() << " w: " << w_chk.transpose() << endl << endl;
	//Map back from se(3) to SE(3)
	expMapSE3(R_chk, T_chk, u, w_chk);
	cout << "Difference between Reference and SE(3) Rotation Matrices: " << (R*R_chk.transpose()).trace()-3.0f << endl << endl;
	cout << "Difference between Reference and SE(3) Trans Vector: " << (T-T_chk).norm() << endl << endl;

	//Check results
	expmapSE3transV(V, Vinv, w);

	cout << "V:" << endl;
	cout << V << endl << endl;
	cout << "Vinv:" << endl;
	cout << Vinv << endl << endl;
	cout << "V*Vinv:" << endl;
	cout << V*Vinv << endl;
}
