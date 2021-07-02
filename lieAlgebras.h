#ifndef LIE_ALGEBRAS_H__
#define LIE_ALGEBRAS_H__

#include <Eigen/Dense>
#include <Eigen/StdVector>

namespace lieAlgebras
{

//CONSTANTS
const float PI       = std::acos(-1.0);
const float minAngle = 0.01f*(PI/180.0f);
//TYPEDEFS
typedef Eigen::Matrix< float,6,1 > Vector6f;
typedef Eigen::Matrix< float,8,1 > Vector8f;
typedef Eigen::Matrix< float,6,6 > Matrix6f;
typedef std::vector<Eigen::Matrix3f, Eigen::aligned_allocator<Eigen::Matrix3f> > vMatrix3f;
typedef std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f> > vVector3f;

//////////////////
// EULER ANGLES //
//////////////////
//Rotation matrix computation from Euler Angles
void composeRotationFromEuler(const Eigen::Vector3f& EA, Eigen::Matrix3f& Rot);

// Computes yaw, roll, pitch Euler Angles from rotation matrix
void getEulerAngles(Eigen::Matrix3f& R, Eigen::Vector3f& Ang);

///////////
// SO(3) //
///////////
//The Rodrigues' equation for the matrix exponential
Eigen::Matrix3f RodriguesRotation(Eigen::Vector3f& w);

//The Rodrigues' equation for the matrix exponential: unit w and theta version
Eigen::Matrix3f RodriguesRotation(Eigen::Vector3f& w, float theta);

//log of matrix exponential for converting from SO(3) to so(3)
Eigen::Vector3f logMatExpR(Eigen::Matrix3f& R);

///////////
// SE(3) //
///////////
//log from SE(3) to se(3)
void logMatExpSE3(Eigen::Vector3f& u, Eigen::Vector3f& w, Eigen::Matrix3f& R, Eigen::Vector3f& t);

//Exponential Map for SE(3)
void expMapSE3(Eigen::Matrix3f& R, Eigen::Vector3f& t, Eigen::Vector3f& u, Eigen::Vector3f& w);

//Adjoints and Inverse for g = [R t;0 1];
Matrix6f Adjoint(Eigen::Matrix3f& R, Eigen::Vector3f& t);
Matrix6f Adjoint(Eigen::Vector3f& u, Eigen::Vector3f& w);
Matrix6f invAdjoint(Eigen::Matrix3f& R, Eigen::Vector3f& t);

//Checks on translation mapping
void expmapSE3transV(Eigen::Matrix3f& V, Eigen::Matrix3f& Vinv, Eigen::Vector3f& w);

/////////
// SL3 //
/////////
//Compose a homography from Rotation Matrix and translation
Eigen::Matrix3f homographyESM(Eigen::Matrix3f& R, Eigen::Vector3f& t);
Eigen::Matrix3f homographyESM(Eigen::Matrix3f& R, Eigen::Vector3f& t, Eigen::Vector3f& n);

//log of matrix exponential for converting from SL(3) to sl(3)
Vector8f logMatExpSL3(Eigen::Matrix3f& H);

//Exponential Map for SL(3)
void expMapSL3(Eigen::Matrix3f& H, Vector8f& x);

//Recover Rotation and translation from H
bool extractRTfromH(vMatrix3f& vRot, vVector3f& vtrans, Eigen::Matrix3f& H, Eigen::Matrix3f& Rr);

} //namespace lieAlgebras

#endif  //LIE_ALGEBRAS_H__