#include <ceres/ceres.h>
#include <glog/logging.h>
#include <Eigen/SVD>
#include "cost_functions.h"

using ceres::Solver;
using ceres::Solve;
using ceres::Problem;
using ceres::CostFunction;
using ceres::AutoDiffCostFunction;
using namespace Eigen;

typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;
typedef Eigen::Matrix<double, 3, 3> Mat3;
typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Matrix<double, Eigen::Dynamic, 9> MatX9;
typedef Eigen::Vector3d Vec3;

template<typename T = double>
class Homography2DNormalizedParameterization {
public:
	typedef Eigen::Matrix<T, 9, 1> Parameters;     // a, b, ... g, h
	typedef Eigen::Matrix<T, 3, 3> Parameterized;  // H

	// Convert from the 8 parameters to a H matrix.
	static void To(const Parameters& p, Parameterized* h) {
		*h << p(0)/p(8), p(1)/p(8), p(2)/p(8),
			p(3)/p(8), p(4)/p(8), p(5)/p(8),
			p(6)/p(8), p(7)/p(8), p(8)/p(8);
	}

	// Convert from a H matrix to the 8 parameters.
	static void From(const Parameterized& h, Parameters* p) {
		*p << h(0, 0), h(0, 1), h(0, 2),
			h(1, 0), h(1, 1), h(1, 2),
			h(2, 0), h(2, 1),1;
	}
};



bool create_optimization_problem(const Mat &x1, const Mat &x2, Mat &x3_new,Mat3 *H) {

	// few checks 
	assert(2 == x1.rows());
	assert(4 <= x1.cols());
	assert(x1.cols() == x2.cols());
	assert(x1.rows() == x2.rows());

	// minimize the algebric error

	int n = x1.cols();
	MatX9 A = Mat::Zero(n * 2, 9);

	for (int i = 0; i < n; ++i) {
		int j = 2 * i;
		A(j, 0) = 0;
		A(j, 1) = 0;
		A(j, 2) = 0;
		A(j, 3) = -x1(0, i);
		A(j, 4) = -x1(1, i);
		A(j, 5) = -1;
		A(j, 6) = x2(1, i) * x1(0, i);
		A(j, 7) = x2(1, i) * x1(1, i);
		A(j, 8) = x2(1, i) * 1;

		++j;

		A(j, 0) = x1(0, i);
		A(j, 1) = x1(1, i);
		A(j, 2) = 1;
		A(j, 3) = 0;
		A(j, 4) = 0;
		A(j, 5) = 0;
		A(j, 6) = -x2(0, i) * x1(0, i);
		A(j, 7) = -x2(0, i) * x1(1, i);
		A(j, 8) = -x2(0, i) * 1;

		Mat V = A.jacobiSvd(ComputeFullU | ComputeFullV).matrixV();
		Vec h = V.col(8);
		Homography2DNormalizedParameterization<double>::To(h, H);
		double k = {2};

		// Improve the matrix using the Iterative minimization using the reprojection error as the cost Funxtion
		ceres::Problem problem;
		Vec2 x_new_vec;
		for (int i = 0; i < x1.cols();i++) {

		 double	x_new_vec[2] = {x3_new(0, i), x3_new(1, i)};
			ceres::CostFunction* mycost_o = new AutoDiffCostFunction<sampson_error_cost, 1,9>(new sampson_error_cost(x1.col(i), x2.col(i)));
			problem.AddResidualBlock(mycost_o, nullptr,H->data());
					
		}
		// Configure the solve.
		ceres::Solver::Options solver_options;
		solver_options.linear_solver_type = ceres::DENSE_QR;
		solver_options.update_state_every_iteration = true;
		solver_options.max_num_iterations = 50;


		// Run the solve.
		ceres::Solver::Summary summary;
		ceres::Solve(solver_options, &problem, &summary);

		LOG(INFO) << "Summary:\n" << summary.FullReport();
		LOG(INFO) << "Final refined matrix:\n" << *H;

		return summary.IsSolutionUsable();
	}
}

int main(int argc, char **argv) {

	google::InitGoogleLogging(argv[0]);


	Mat x1(2, 100);
	// Data initiliztion

	for (int i = 0; i < x1.cols(); i++) {
		x1(0,i) = rand() % 1024;
		x1(1, i) = rand() % 1024;
	}
	
	Mat3 Hmatrix;

	// true h_matrix

	Hmatrix << 1.243715, -0.461057, -111.964454,
			   0.0, 0.617589, -192.379252,
			   0.0, -0.000983, 1.0;
	Mat x2 = x1;

	Mat x_new = x2;

	// create the correspondence points 

	for (int i = 0; i < x2.cols(); ++i) {
     
		Vec3 homogeneous_x1 = Vec3(x1(0, i), x1(1, i), 1);
		Vec3 homogeneous_x2 = Hmatrix * homogeneous_x1;
		x2(0, i) = homogeneous_x2(0) / homogeneous_x2(2);
		x2(1,i) = homogeneous_x2(1) / homogeneous_x2(2);

		//apply some noise to model error in the correspondence measurement
		x2(0, i) += static_cast<double>(rand() % 1000) / 5000.0;
		x2(1, i) += static_cast<double>(rand() % 1000) / 5000.0;

	}


	for (int i = 0; i < x_new.cols(); ++i) {
		
		x_new(0, i) = x1(0,i) + static_cast<double>(rand() % 1000) / 5000.0;
		x_new(1, i) = x1(1,i) + static_cast<double>(rand() % 1000) / 5000.0;

	}
	Mat3 estimated_Hmatrix;

	create_optimization_problem(x1,x2,x_new,&estimated_Hmatrix);

	estimated_Hmatrix /= estimated_Hmatrix(2, 2);

	std::cout << "Original matrix:\n" << Hmatrix << "\n";
	std::cout << "Estimated matrix:\n" << estimated_Hmatrix << "\n";

	return 0;
}