#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  if ((estimations.size() == 0) || (estimations.size() != ground_truth.size()))
  {
      std::cout << "Error: Size mismatch between estimations and ground_truth vectors" << std::endl;
      return rmse;
  }
  // accumulate squared residuals
  for (int i=0; i < estimations.size(); ++i) {
    VectorXd R = estimations[i] - ground_truth[i];
    R = R.array() * R.array();
    rmse += R;
  }
  // calculate the mean
  rmse = rmse / estimations.size();  
  // calculate the squared root
  rmse = rmse.array().sqrt();  
  // return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

  float rho = sqrt(px*px + py*py);
  // check division by zero
  if (fabs(rho) < 0.00001)
  {
      std::cout << "Error: Division by Zero" << std::endl;
      return Hj;
  }

  // compute the Jacobian matrix
  Hj << px / rho, py / rho, 0, 0,
        -py / pow(rho, 2), px / pow(rho, 2), 0, 0,
        py * (vx*py - vy*px) / pow(rho, 3), px * (vy*px - vx*py) / pow(rho, 3), px / rho, py / rho;
        
  return Hj;
}
