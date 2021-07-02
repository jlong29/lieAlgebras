#include "lieAlgebras.h"

#include <iostream>
#include <algorithm>
#include <vector>
#include <utility>
#include <Eigen/Core>
#include <Eigen/SVD>
#include <unsupported/Eigen/MatrixFunctions>

namespace lieAlgebras
{

//////////////////
// EULER ANGLES //
//////////////////
//Rotation matrix computation from Euler Angles
void composeRotationFromEuler(const Eigen::Vector3f& EA, Eigen::Matrix3f& Rot)
{
	//Convert rotation values into matrix using rotations about principle axes
	//Most IMUs uses the roll, pitch, yaw Tai-Bryan convention
	//https://en.wikipedia.org/wiki/Euler_angles

	//Active Tait-Bryan Angles: Y1*X2*Z3
	//This parameterization places the singularity at pitch (X) +/- pi/2 when computing Euler Angles

	float c1 = cosf(EA[1]); float s1 = sinf(EA[1]);
	float c2 = cosf(EA[0]); float s2 = sinf(EA[0]);
	float c3 = cosf(EA[2]); float s3 = sinf(EA[2]);

	Rot << c1*c3+s1*s2*s3, c3*s1*s2-c1*s3, c2*s1, c2*s3, c2*c3, -s2, c1*s2*s3-c3*s1, c1*c3*s2+s1*s3, c1*c2;
}

void getEulerAngles(Eigen::Matrix3f& R, Eigen::Vector3f& Ang)
{
	//Output is X,Y,Z
	//Derived from rotation of form: Y1*X2*Z3
	// Y(alpha), X(beta), Z(gamma)
	//https://en.wikipedia.org/wiki/Euler_angles
	Ang(0) = -asin(R(1,2));                        //roll
	float cosA1 = cos(Ang(0));
	Ang(2) =  atan2(R(1,0)/cosA1, R(1,1)/cosA1);   //pitch
	Ang(1) =  atan2(R(0,2)/cosA1, R(2,2)/cosA1);   //yaw
}

///////////
// SO(3) //
///////////
//The Rodrigues' equation for the matrix exponential
Eigen::Matrix3f RodriguesRotation(Eigen::Vector3f& w)
{
	float theta = w.norm();
	if (theta == 0)
		return Eigen::Matrix3f::Identity();

	float theta2 = (theta*theta);

	// Generate cross product operator so(3) and its square
	Eigen::Matrix3f w_hat, w_hat2;
	w_hat << 0.0f, -w(2),  w(1),
			 w(2),  0.0f, -w(0),
			-w(1),  w(0), 0.0f;

	w_hat2 = w*w.transpose() - theta2*Eigen::Matrix3f::Identity();

	// Generate rotation matrix using the Rodrigues' equation
	return Eigen::Matrix3f::Identity() + w_hat*(sin(theta)/theta) + w_hat2*((1-cos(theta))/theta2);
}

//The Rodrigues' equation for the matrix exponential: unit w and theta version
Eigen::Matrix3f RodriguesRotation(Eigen::Vector3f& w, float theta)
{
	// Generate cross product operator so(3) and its square
	Eigen::Matrix3f w_hat, w_hat2;
	w_hat << 0.0f, -w(2),  w(1),
			 w(2),  0.0f, -w(0),
			-w(1),  w(0), 0.0f;

	w_hat2 = w*w.transpose() - Eigen::Matrix3f::Identity();

	// Generate rotation matrix using the Rodrigues' equation
	return Eigen::Matrix3f::Identity() + w_hat*sin(theta) + w_hat2*(1-cos(theta));
}

//log of matrix exponential for converting from SO(3) to so(3)
Eigen::Vector3f logMatExpR(Eigen::Matrix3f& R)
{	
	float tmp   = 0.5f*(R.trace()-1.0f);
	float theta = acos(tmp);
	if ((fabsf(tmp) > 1.0f) || (fabsf(theta) < minAngle)){
		return Eigen::Vector3f::Zero();
	}

	Eigen::Vector3f w;
	float stheta = sin(theta);
	w << R(2,1)-R(1,2), R(0,2)-R(2,0), R(1,0)-R(0,1);
	w /= 2.0f*stheta;
	w *= theta;
	return w;
}

///////////
// SE(3) //
///////////
//log from SE(3) to se(3)
void logMatExpSE3(Eigen::Vector3f& u, Eigen::Vector3f& w, Eigen::Matrix3f& R, Eigen::Vector3f& t)
{
	//w is just the Axis+Angle Representation
	w = logMatExpR(R);

	//u is more complicated
	float theta  = w.norm();
	float theta2 = (theta*theta);

	// Generate cross product operator so(3) and its square
	Eigen::Matrix3f w_hat, w_hat2;
	w_hat << 0.0f, -w(2),  w(1),
			 w(2),  0.0f, -w(0),
			-w(1),  w(0),  0.0f;

	w_hat2 = w*w.transpose() - theta2*Eigen::Matrix3f::Identity();

	float stheta = sin(theta);

	Eigen::Matrix3f Vinv = Eigen::Matrix3f::Identity() - 0.5f*w_hat + ((1.0f/theta2)*(1.0f - (theta*stheta)/(2.0f*(1.0f - cos(theta)))))*w_hat2;
	u = Vinv*t;
}

//Exponential Map for SE(3)
void expMapSE3(Eigen::Matrix3f& R, Eigen::Vector3f& t, Eigen::Vector3f& u, Eigen::Vector3f& w)
{
	float theta  = w.norm();
	float theta2 = (theta*theta);
	float theta3 = theta2*theta;

	// Generate cross product operator so(3) and its square
	Eigen::Matrix3f w_hat, w_hat2;
	w_hat << 0.0f, -w(2),  w(1),
			 w(2),  0.0f, -w(0),
			-w(1),  w(0), 0.0f;

	w_hat2 = w*w.transpose() - theta2*Eigen::Matrix3f::Identity();

	float stheta = sin(theta);
	float tmp1   = ((1-cos(theta))/theta2);

	Eigen::Matrix3f V;
	V = Eigen::Matrix3f::Identity() + w_hat*tmp1 + w_hat2*((theta - stheta)/theta3);

	R = Eigen::Matrix3f::Identity() + w_hat*(sin(theta)/theta) + w_hat2*((1-cos(theta))/(theta*theta));
	t = V*u;
}

//Adjoint for g = [R t;0 1];
Matrix6f Adjoint(Eigen::Matrix3f& R, Eigen::Vector3f& t)
{
	Eigen::Matrix3f t_hat;
	t_hat << 0.0f, -t(2),  t(1),
			 t(2),  0.0f, -t(0),
			-t(1),  t(0),  0.0f;

	Matrix6f Adj;
	Adj << R, t_hat*R, Eigen::Matrix3f::Zero(), R;
	return Adj;
}
Matrix6f Adjoint(Eigen::Vector3f& u, Eigen::Vector3f& w)
{
	Eigen::Matrix3f R;
	Eigen::Vector3f t;
	expMapSE3(R, t, u, w);

	return Adjoint(R, t);
}

Matrix6f invAdjoint(Eigen::Matrix3f& R, Eigen::Vector3f& t)
{
	Eigen::Matrix3f t_hat;
	t_hat << 0.0f, -t(2),  t(1),
			 t(2),  0.0f, -t(0),
			-t(1),  t(0),  0.0f;

	Matrix6f invAdj;
	Eigen::Matrix3f Rt = R.transpose();
	invAdj << Rt, -Rt*t_hat, Eigen::Matrix3f::Zero(), Rt;
	return invAdj;
}
//Checks on translation mapping
void expmapSE3transV(Eigen::Matrix3f& V, Eigen::Matrix3f& Vinv, Eigen::Vector3f& w)
{
	float theta  = w.norm();
	float theta2 = (theta*theta);
	float theta3 = theta2*theta;

	// Generate cross product operator so(3) and its square
	Eigen::Matrix3f w_hat, w_hat2;
	w_hat << 0.0f, -w(2),  w(1),
			 w(2),  0.0f, -w(0),
			-w(1),  w(0), 0.0f;

	w_hat2 = w*w.transpose() - theta2*Eigen::Matrix3f::Identity();

	float stheta = sin(theta);
	float tmp1   = ((1-cos(theta))/theta2);

	V    = Eigen::Matrix3f::Identity() + tmp1*w_hat + ((theta - stheta)/theta3)*w_hat2;
	Vinv = Eigen::Matrix3f::Identity() - 0.5f*w_hat + ((1.0f/theta2)*(1.0f - (theta*stheta)/(2.0f*(1.0f - cos(theta)))))*w_hat2;
}

///////////
// SL(3) //
///////////
//Compose a homography from Rotation Matrix and translation
Eigen::Matrix3f homographyESM(Eigen::Matrix3f& R, Eigen::Vector3f& t)
{
	/*
		Compose Homography:
		This parameterization follows the ESM work of Mei and Malis:
		H ~ R + t*n_d.tranpose()
		n is assumed to be the z-axis and d = 1
		det(H) = 1
		therefore,
		H = [R1 | R2 | R3+t]
	*/
	//H = [R1 | R2 | R3+t]
	R.col(2) += t;
	// Scale to have det(H) = 1
	float scale = R.determinant();
	R *= pow(1.0f/scale, 1.0f/3.0f);
	return R;
}
Eigen::Matrix3f homographyESM(Eigen::Matrix3f& R, Eigen::Vector3f& t, Eigen::Vector3f& n)
{
	R += t*n.transpose();
	// Scale to have det(H) = 1
	float scale = R.determinant();
	R *= pow(1.0f/scale, 1.0f/3.0f);
	return R;
}

//log of matrix exponential for converting from SL(3) to sl(3)
Vector8f logMatExpSL3(Eigen::Matrix3f& H)
{
	//H = exp(A)
	//A = sum_i<8(Ai*xi)
	//Ai elem R3x3

	//Take matrix log to recover A
	Eigen::Matrix3f A = H.log();

	Vector8f x;
	x << A(0,2), A(1,2), (A(0,1)-A(1,0))/2.0f, A(1,1), A(0,0)-A(1,1), (A(0,1)+A(1,0))/2.0f, A(2,0), A(2,1);
	return x;
}

//Exponential Map for SL(3)
void expMapSL3(Eigen::Matrix3f& H, Vector8f& x)
{
	//H = exp(A)
	//A = sum_i<8(Ai*xi)
	//Ai elem R3x3

	Eigen::Matrix3f A;
	A << x(3)+x(4), x(2)+x(5), x(0), -x(2)+x(5), x(3), x(1), x(6), x(7), -2*x(3)-x(4);

	H = A.exp();
}

//Recover Rotation and translation from H1 and compare against reference R and t
bool extractRTfromH(vMatrix3f& vRot, vVector3f& vtrans, Eigen::Matrix3f& H, Eigen::Matrix3f& Rr)
{
	// We recover 8 motion hypotheses using the method of Faugeras et al.
	// Motion and structure from motion in a piecewise planar environment.
	// International Journal of Pattern Recognition and Artificial Intelligence, 1988

	//SVD
	Eigen::Matrix3f U, Vt, V;	//Orthogonal matrices
	Eigen::Vector3f w;			//singular values
	Eigen::JacobiSVD<Eigen::Matrix3f> svd(H, Eigen::ComputeFullV | Eigen::ComputeFullU);
	U  = svd.matrixU();
	w  = svd.singularValues();
	V  = svd.matrixV();
	Vt = V.transpose();

	//Check the SVD reconstruction
	//Eigen::Matrix3f chk = U * w.asDiagonal() * Vt;
	//Eigen::Matrix3f diff = H - chk;
	//std::cout << "diff: " << diff.array().abs().sum() << std::endl << std::endl;

	float s = U.determinant()*Vt.determinant();

	float d1 = w(0);
	float d2 = w(1);
	float d3 = w(2);

	//Check for degenerate cases
	if((d1/d2<1.00001f) || (d2/d3<1.00001f))
	{
		return false;
	}

	vMatrix3f vR;
	vVector3f vt, vn;
	vR.reserve(8);
	vt.reserve(8);
	vn.reserve(8);

	//n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
	float d1d1 = d1*d1;
	float d2d2 = d2*d2;
	float d3d3 = d3*d3;
	float aux1 = sqrt((d1d1-d2d2)/(d1d1-d3d3));
	float aux3 = sqrt((d2d2-d3d3)/(d1d1-d3d3));
	float x1[] = {aux1,aux1,-aux1,-aux1};
	float x3[] = {aux3,-aux3,aux3,-aux3};

	//case d'=d2
	float aux_stheta = sqrt((d1d1-d2d2)*(d2d2-d3d3))/((d1+d3)*d2);

	float ctheta = (d2d2+d1*d3)/((d1+d3)*d2);
	float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

	for(int i=0; i<4; i++)
	{
		Eigen::Matrix3f Rp=Eigen::Matrix3f::Identity();
		Rp(0,0)=ctheta;
		Rp(0,2)=-stheta[i];
		Rp(2,0)=stheta[i];
		Rp(2,2)=ctheta;

		Eigen::Matrix3f R = s*U*Rp*Vt;
		vR.push_back(R);

		Eigen::Vector3f tp;
		tp(0)=x1[i];
		tp(1)=0.0f;
		tp(2)=-x3[i];
		tp*=d1-d3;

		Eigen::Vector3f t = U*tp;
		vt.push_back(t);

		Eigen::Vector3f np;
		np(0)=x1[i];
		np(1)=0.0f;
		np(2)=x3[i];

		Eigen::Vector3f n = V*np;
		if(n(2)<0)
			n=-n;
		vn.push_back(n);
	}

	//case d'=-d2
	float aux_sphi = sqrt((d1d1-d2d2)*(d2d2-d3d3))/((d1-d3)*d2);

	float cphi = (d1*d3-d2d2)/((d1-d3)*d2);
	float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

	for(int i=0; i<4; i++)
	{
		Eigen::Matrix3f Rp=Eigen::Matrix3f::Identity();
		Rp(0,0)=cphi;
		Rp(0,2)=sphi[i];
		Rp(1,1)=-1.0f;
		Rp(2,0)=sphi[i];
		Rp(2,2)=-cphi;

		Eigen::Matrix3f R = s*U*Rp*Vt;
		vR.push_back(R);

		Eigen::Vector3f tp;
		tp(0)=x1[i];
		tp(1)=0.0f;
		tp(2)=x3[i];
		tp*=d1+d3;

		Eigen::Vector3f t = U*tp;
		vt.push_back(t);

		Eigen::Vector3f np;
		np(0)=x1[i];
		np(1)=0.0f;
		np(2)=x3[i];

		Eigen::Vector3f n = V*np;
		if(n(2)<0)
			n=-n;
		vn.push_back(n);
	}

	//Inspect and compare against reference Rr and tr
	float R_diff;
	Eigen::Vector3f w_chk, n_ref;

	//The assumption is that n is along the z-axis at a distance of 1
	n_ref << 0.0f, 0.0f, 1.0f;
	std::vector< std::pair<float, int> > scoreVec;
	for (size_t i = 0; i < 8; i++)
	{
		//Eigen::Matrix3f Hr = homographyESM(vR[i], vt[i], vn[i]);
		//std::cout << "Hr:\n" << Hr <<std::endl;
		//std::cout << "sl(3) from Hr: " << logMatExpSE3(Hr).transpose() << std::endl;

		float diff = (n_ref - vn[i]).array().abs().sum();
		if (diff < 0.01f)
		{
			//std::cout << "RT: " << i << std::endl;
			//std::cout << vR[i] << std::endl << std::endl;
			//std::cout << vt[i].transpose() << std::endl;
			//std::cout << vn[i].transpose() << std::endl << std::endl;

			R_diff = (vR[i]-Rr).array().abs().sum();
			//std::cout << "Comparison to Reference Rotation: " << R_diff << std::endl << std::endl;

			//Log score
			scoreVec.push_back( std::make_pair(R_diff, i));
		}
	}

	//Sort scores
	std::sort(scoreVec.begin(), scoreVec.end());
	//Return the 2 lowest scoring Rotation differences and associated t
	vRot.push_back(vR[scoreVec[0].second]); vtrans.push_back(vt[scoreVec[0].second]);
	vRot.push_back(vR[scoreVec[1].second]); vtrans.push_back(vt[scoreVec[1].second]);

	/*
	for (int i=0; i<scoreVec.size(); i++)
		std::cout << scoreVec[i].second << ", " << scoreVec[i].first << std::endl;
	*/

	return true;
}

} //namespace lieAlgebras
