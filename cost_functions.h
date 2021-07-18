#include <ceres/ceres.h>
#include <ceres/rotation.h>


typedef Eigen::MatrixXd Mat;
typedef Eigen::VectorXd Vec;
typedef Eigen::Matrix<double, 3, 3> Mat3;
typedef Eigen::Matrix<double, 2, 1> Vec2;
typedef Eigen::Matrix<double, Eigen::Dynamic, 9> MatX9;
typedef Eigen::Vector3d Vec3;


struct Homography_transfer_cost {

	Homography_transfer_cost(const Vec2& x , const Vec2&y ) \
		: x_(x) , y_(y) {}

	template<typename T>
	bool operator()(const T* homography_parameters, T* residuals) const {

		typedef Eigen::Matrix<T, 3, 3> Mat3;
		typedef Eigen::Matrix<T, 3, 1> Vec3;
		typedef Eigen::Matrix<T, 2, 1> Vec2;

		Vec3 x(T(x_(0)), T(x_(1)), T(1.0));
		Vec3 y(T(y_(0)), T(y_(1)), T(1.0));
		Vec2 forward_error;
		Mat3 H(homography_parameters);
		
		Vec3 H_x = H * x;
		H_x /= H_x(2);

		forward_error[0] = T(H_x(0)) - T(y_(0));
		forward_error[1] = T(H_x(1)) - T(y_(1));

		residuals[0] = forward_error.squaredNorm();
		return true;
	}

	const Vec2 x_;
	const Vec2 y_;

};


struct Homography_symmetric_transfer_cost {

	Homography_symmetric_transfer_cost(const Vec2& x, const Vec2& y) \
		: x_(x), y_(y) {}

	template<typename T>
	bool operator()(const T* homography_parameters, T* residuals) const {

		typedef Eigen::Matrix<T, 3, 3> Mat3;
		typedef Eigen::Matrix<T, 3, 1> Vec3;
		typedef Eigen::Matrix<T, 2, 1> Vec2;

		Vec3 x(T(x_(0)), T(x_(1)), T(1.0));
		Vec3 y(T(y_(0)), T(y_(1)), T(1.0));
		Vec2 forward_error;
		Vec2 backward_error;

		Mat3 H(homography_parameters);

		Vec3 H_x = H * x;
		Vec3 Hinv_y = H.inverse() * y;

		H_x /= H_x(2);
		Hinv_y /= Hinv_y(2);

		forward_error[0] = T(H_x(0)) - T(y_(0));
		forward_error[1] = T(H_x(1)) - T(y_(1));

		backward_error[0] = T(Hinv_y(0)) - T(x(0));
		backward_error[1] = T(Hinv_y(1)) - T(x(1));

		// minizide d(x,Hx')^2 +  d(x',H_inv*x)^2

		residuals[0] = forward_error.squaredNorm()  + backward_error.squaredNorm();

		return true;
	}

	const Vec2 x_;
	const Vec2 y_;

};


struct Homography_reprojection_cost {

	Homography_reprojection_cost(const Vec2& x, const Vec2& y) \
		: x_(x), y_(y) {}

	template<typename T>
	bool operator()(const T* homography_parameters, const T* x_new,T* residuals) const {

		typedef Eigen::Matrix<T, 3, 3> Mat3;
		typedef Eigen::Matrix<T, 3, 1> Vec3;
		typedef Eigen::Matrix<T, 2, 1> Vec2;

		Vec3 x(T(x_(0)), T(x_(1)), T(1.0));
		Vec3 y(T(y_(0)), T(y_(1)), T(1.0));
		Vec2 forward_error;
		Vec2 backward_error;

		Mat3 H(homography_parameters);
	
		Vec3 x_new_vec;
		x_new_vec = Vec3(T(x_new[0]), T(x_new[0]),T(1));

		Vec3  y_new = H * x_new_vec;
		y_new /= y_new(2);
		
		forward_error[0] = T(x_new[0]) - T(x_(0));
		forward_error[1] = T(x_new[1]) - T(x_(1));

		backward_error[0] = T(y_new(0)) - T(y_(0));
		backward_error[1] = T(y_new(1)) - T(y_(1));

		// minizide d(x,x_new)^2 +  d(x',x'_new)^2

		residuals[0] = forward_error.squaredNorm() + backward_error.squaredNorm();

		return true;
	}

	const Vec2 x_;
	const Vec2 y_;

};
