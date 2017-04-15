#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  is_initialized_ = false;

  // state and augmented state dimension
  n_x_ = 5;
  n_aug_ = 7;

  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd::Zero(n_x_);

  // initial state covariance matrix
  P_ = MatrixXd::Identity(n_x_, n_x_);

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd::Zero(n_x_, 2*n_aug_+1);

  // time measured in us
  time_us_ = true;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = M_PI/4;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  // weights of sigma points
  weights_ = VectorXd::Zero(2*n_aug_+1);

  // set weights
  weights_(0) = lambda_ / (lambda_+n_aug_);
  for (int i=1; i<2*n_aug_+1; i++)
    weights_(i) = 0.5 / (lambda_+n_aug_);

  // normalized innovation squared
  NIS_radar_ = 0;
  NIS_laser_ = 0;
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /****************************************************************************
   *  Initialization
   ****************************************************************************/
  if ((meas_package.sensor_type_ == MeasurementPackage::RADAR && !use_radar_) ||
      (meas_package.sensor_type_ == MeasurementPackage::LASER && !use_laser_))
    return;

  if (!is_initialized_) {
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      // convert radar from polar to Cartesian coordinates and initialize state.
      x_ << meas_package.raw_measurements_[0] *\
                cos(meas_package.raw_measurements_[1]),
                meas_package.raw_measurements_[0] *\
                sin(meas_package.raw_measurements_[1]),
                0, 0, 0;
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      x_ << meas_package.raw_measurements_[0],
            meas_package.raw_measurements_[1],
            0, 0, 0;
    }

    // update timestamp
    previous_timestamp_ = meas_package.timestamp_;

    // initialization complete
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
  double dt;
  if (time_us_)
    dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;
  else
    dt = (meas_package.timestamp_ - previous_timestamp_);

  previous_timestamp_ = meas_package.timestamp_;

  // slice time longer than 100ms
  while (dt > 0.1) {
    double delta_t = 0.05;
    Prediction(delta_t);
    dt -= delta_t;
  }
  Prediction(dt);

  /*****************************************************************************
   *  Update
   ****************************************************************************/
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
    UpdateRadar(meas_package);
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    UpdateLidar(meas_package);
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  // create augmented mean state
  VectorXd x_aug = VectorXd::Zero(n_aug_);
  x_aug << x_, 0, 0;

  // create augmented covariance matrix
  MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
  P_aug.topLeftCorner(n_x_, n_x_) = P_;
  P_aug(n_x_, n_x_) = std_a_ * std_a_;
  P_aug(n_x_+1, n_x_+1) = std_yawdd_ * std_yawdd_;

  // calculate square root of P
  MatrixXd A = P_aug.llt().matrixL();

  // create augmented sigma points
  MatrixXd Xsig_aug = MatrixXd::Zero(n_aug_, 2*n_aug_+1);

  // set sigma points as columns of matrix Xsig_aug
  Xsig_aug.col(0) = x_aug;
  Xsig_aug.block<7,7>(0,1) = x_aug.rowwise().replicate(n_aug_) + sqrt(lambda_+n_aug_) * A;
  Xsig_aug.block<7,7>(0,8) = x_aug.rowwise().replicate(n_aug_) - sqrt(lambda_+n_aug_) * A;

  // predict sigma points
  for (int i=0; i<(2*n_aug_+1); i++) {
    VectorXd x(n_x_), xd(n_x_), nud(n_x_);
    xd << VectorXd::Zero(n_x_);

    double px, py, v, phi, phi_dot, nu_a, nu_phi_dot_dot;
    x(0) = px = Xsig_aug(0,i);
    x(1) = py = Xsig_aug(1,i);
    x(2) = v = Xsig_aug(2,i);
    x(3) = phi = Xsig_aug(3,i);
    x(4) = phi_dot = Xsig_aug(4,i);
    nu_a = Xsig_aug(5,i);
    nu_phi_dot_dot = Xsig_aug(6,i);

    // avoid division by zero
    if (fabs(phi_dot < 0.0001)) {
      xd(0) = v*cos(phi)*delta_t;
      xd(1) = v*sin(phi)*delta_t;
    } else {
      xd(0) = v/phi_dot*(sin(phi+phi_dot*delta_t)-sin(phi));
      xd(1) = v/phi_dot*(-cos(phi+phi_dot*delta_t)+cos(phi));
    }
    xd(3) = phi_dot*delta_t;

    double dt12 = 0.5*delta_t*delta_t;
    nud(0) = dt12*cos(phi)*nu_a;
    nud(1) = dt12*sin(phi)*nu_a;
    nud(2) = delta_t*nu_a;
    nud(3) = dt12*nu_phi_dot_dot;
    nud(4) = delta_t*nu_phi_dot_dot;

    Xsig_pred_.col(i) = x + xd + nud;
  }

  // predict state mean
  VectorXd x = VectorXd::Zero(n_x_);
  for (int i=0; i<2*n_aug_+1; i++)
    x += weights_(i) * Xsig_pred_.col(i);

  // predice state covariance matrix
  MatrixXd P = MatrixXd::Zero(n_x_, n_x_);
  for (int i=0; i<2*n_aug_+1; i++) {
    VectorXd diff = Xsig_pred_.col(i) - x;

    diff(3) = atan2(sin(diff(3)), cos(diff(3)));

    P += weights_(i) * diff * diff.transpose();
  }

  // output predicted state and state covariance matrix
  x_ = x;
  P_ = P;
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  // measurement dimension
  int n_z = 2;

  // state mean
  VectorXd x = VectorXd::Zero(n_x_);
  x << x_;

  // state covariance matrix
  MatrixXd P = MatrixXd::Zero(n_x_, n_x_);
  P << P_;

  // mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z);

  // predicted measurement covariance
  MatrixXd S = MatrixXd::Zero(n_z, n_z);

  // transform sigma points into measurement space
  MatrixXd Zsig = MatrixXd::Zero(n_z, 2*n_aug_+1);
  for (int i=0; i<2*n_aug_+1; i++) {
    double px, py;
    px = Xsig_pred_(0,i);
    py = Xsig_pred_(1,i);
    Zsig(0,i) = px;
    Zsig(1,i) = py;
  }

  // calculate mean predicted measurement
  for (int i=0; i<2*n_aug_+1; i++)
    z_pred += weights_(i) * Zsig.col(i);

  // measurement covariance matrix S
  for (int i=0; i<2*n_aug_+1; i++) {
    VectorXd diff = Zsig.col(i) - z_pred;

    S += weights_(i) * diff * diff.transpose();
  }

  // calculate measurement covariance matrix S
  MatrixXd R = MatrixXd::Zero(n_z, n_z);
  R(0,0) = std_laspx_ * std_laspx_;
  R(1,1) = std_laspy_ * std_laspy_;

  S += R;

  // incoming laser measurement
  VectorXd z = VectorXd::Zero(n_z);
  z << meas_package.raw_measurements_;

  // calculate cross correlation matrix
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
  for (int i=0; i<2*n_aug_+1; i++) {
    VectorXd xdiff = Xsig_pred_.col(i) - x;
    VectorXd zdiff = Zsig.col(i) - z_pred;

    Tc += weights_(i) * xdiff * zdiff.transpose();
  }

  // calculate Kalman gain
  MatrixXd K = Tc * S.inverse();

  // update state mean and covariance matrix
  VectorXd zdiff = z - z_pred;

  x_ += K * zdiff;
  P_ -= K * S * K.transpose();

  NIS_laser_ = zdiff.transpose() * S.inverse() * zdiff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  // measurement dimension
  int n_z = 3;

  // state mean
  VectorXd x = VectorXd::Zero(n_x_);
  x << x_;

  // state covariance matrix
  MatrixXd P = MatrixXd::Zero(n_x_, n_x_);
  P << P_;

  // transform sigma points into measurement space
  MatrixXd Zsig = MatrixXd::Zero(n_z, 2*n_aug_+1);
  for (int i=0; i<2*n_aug_+1; i++) {
    double px, py, v, phi, rho;
    px = Xsig_pred_(0,i);
    py = Xsig_pred_(1,i);
    v = Xsig_pred_(2,i);
    phi = Xsig_pred_(3,i);
    rho = sqrt(px*px+py*py);

    Zsig(0,i) = rho;
    Zsig(1,i) = atan2(py,px);
    // divide by zero check
    if (fabs(rho) < 0.0001) {
      rho = 0.0001;
    }
    Zsig(2,i) = (px*cos(phi)*v + py*sin(phi)*v) / rho;
  }

  // calculate mean predicted measurement
  VectorXd z_pred = VectorXd::Zero(n_z);
  for (int i=0; i<2*n_aug_+1; i++)
    z_pred += weights_(i) * Zsig.col(i);

  // measurement covariance matrix S
  MatrixXd S = MatrixXd::Zero(n_z, n_z);
  for (int i=0; i<2*n_aug_+1; i++) {
    VectorXd diff = Zsig.col(i) - z_pred;

    diff(1) = atan2(sin(diff(1)), cos(diff(1)));

    S += weights_(i) * diff * diff.transpose();
  }

  // calculate measurement covariance matrix S
  MatrixXd R = MatrixXd::Zero(n_z, n_z);
  R(0,0) = std_radr_ * std_radr_;
  R(1,1) = std_radphi_ * std_radphi_;
  R(2,2) = std_radrd_ * std_radrd_;

  S += R;

  // incoming radar measurement
  VectorXd z = VectorXd::Zero(n_z);
  z << meas_package.raw_measurements_;

  // calculate cross correlation matrix
  MatrixXd Tc = MatrixXd::Zero(n_x_, n_z);
  for (int i=0; i<2*n_aug_+1; i++) {
    VectorXd xdiff = Xsig_pred_.col(i) - x;
    VectorXd zdiff = Zsig.col(i) - z_pred;

    xdiff(3) = atan2(sin(xdiff(3)), cos(xdiff(3)));
    zdiff(1) = atan2(sin(zdiff(1)), cos(zdiff(1)));

    Tc += weights_(i) * xdiff * zdiff.transpose();
  }

  //c alculate Kalman gain
  MatrixXd K = Tc * S.inverse();

  // update state mean and covariance matrix
  VectorXd zdiff = z - z_pred;

  zdiff(1) = atan2(sin(zdiff(1)), cos(zdiff(1)));

  // output updated state and state covariance matrix
  x_ += K * zdiff;
  P_ -= K * S * K.transpose();

  NIS_radar_ = zdiff.transpose() * S.inverse() * zdiff;
}
