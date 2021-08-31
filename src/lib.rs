// Work by Samuel Rembisz 2021

//! # Yet Another Linear Algebra Library
//! A helper library for Linear Algebra functions and data structures. It provides Vectors and Matrices for developers to use in their projects.
//!
//!
//! As an ease of use, the library provides Vectors and a Matrix structs.
//!
//! ## Vectors
//! - [`vector::Vector`] : A 2d Vector with an x and y component.
//! - [`vector::Vector3d`] : A 3d Vector with an x, y and z component.
//! - [`vector::VectorN`] : A Vector with any number of components held in a [`Vec<f64>`]
//! ```
//! use yalal::vector::Vector3d;
//!
//! let u = Vector3d::new(1.0, 2.0, 3.0);
//! let v = Vector3d::new(4.0, 5.0, 6.0);
//!
//! println!("{}", u + v); // Vector addition, should output <5.0, 6.0, 7.0>
//! println!("{}", u - v); // Vector subtration, should output <-3.0, -3.0, -3.0>
//! println!("{}", u.dot(&v)); // Dot product, should output 32.0
//! ```
//!
//! ## Matrix
//! - [`matrix::Matrix`] : a matrix defined by the dimensions (rows, cols) and a Vec<f64> of values.
//!
//! ```
//! use yalal::matrix::Matrix;
//!
//! let m = Matrix::new(2usize, 2usize, vec![1.0, 2.0,
//!                                      3.0, 4.0]).unwrap();
//! let n = Matrix::new(2usize, 2usize, vec![5.0, 6.0, 
//!                                      7.0, 8.0]).unwrap();
//!
//! println!("{}", m); // should output [1.0, 2.0]
//!                    //               [3.0, 4.0]
//! println!("{}", m.rows()); // should output 2
//! println!("{}", m.cols()); // should output 2
//! println!("{}", m.get(0u16, 0u16).unwrap()); // should output 1.0
//! println!("{}", m.dot(&n).unwrap()); // should output [19.0, 22.0]
//! ```

pub mod line;
pub mod matrix;
pub mod vector;

#[cfg(test)]
mod tests {
    use crate::{matrix::{Matrix, Triangular}, vector::{Vector, Vector3d, VectorN}};

    #[test]
    fn test_vector_from_angle() {
        let v = Vector::from_angle(90.0);
        assert_eq!(v, Vector::new(0.0, 1.0));
    }

    #[test]
    fn test_vector_add() {
        let u = Vector::new(1.0, 2.0);
        let v = Vector::new(3.0, 4.0);
        assert_eq!(u + v, Vector::new(4.0, 6.0));
    }

    #[test]
    fn test_vector_sub() {
        let u = Vector::new(1.0, 2.0);
        let v = Vector::new(3.0, 4.0);
        assert_eq!(u - v, Vector::new(-2.0, -2.0));
    }

    #[test]
    fn test_vector_normalize() {
        let v = Vector::new(5.0, 5.0);
        let vn = v.get_normalized();
        assert_eq!(vn, Vector::new(0.7071067811865476, 0.7071067811865476));
    }

    #[test]
    fn test_vector_distance() {
        let v1 = Vector::new(1.0, 1.0);
        let v2 = Vector::new(2.0, 2.0);
        assert_eq!(v1.distance(v2), 2.0f64.sqrt());
    }

    #[test]
    fn test_vector_angle_between() {
        let v1 = Vector::new(0.0, 5.0);
        let v2 = Vector::new(-10.0, 0.0);
        assert_eq!(v1.angle_between(&v2), 90.0);
    }

    #[test]
    fn test_vector_dot() {
        let v1 = Vector::new(1.0, 2.0);
        let v2 = Vector::new(3.0, 4.0);
        assert_eq!(v1.dot(&v2), 11.0);
    }

    #[test]
    fn test_vector_cross() {
        let v1 = Vector::new(1.0, 2.0);
        let v2 = Vector::new(3.0, 4.0);
        assert_eq!(v1.cross(&v2), Vector3d::new(0.0, 0.0, 2.0));
    }

    #[test]
    fn test_vector_projection() {
        let u = Vector::new(1.0, 2.0);
        let v = Vector::new(3.0, 4.0);
        assert_eq!(v.project(u), Vector::new(11.0 / 5.0, 22.0 / 5.0));
    }

    #[test]
    fn test_scalar_component() {
        let u = Vector::new(1.0, 2.0);
        let v = Vector::new(3.0, 4.0);
        assert_eq!(u.scalar_comp(&v), 4.919349550499537);
    }

    #[test]
    fn test_zero_vector_mag() {
        let v = Vector::new(0.0, 0.0);
        assert_eq!(v.mag(), 0.0);
    }

    #[test]
    fn test_vector3d_add() {
        let u = Vector3d::new(1.0, 2.0, 3.0);
        let v = Vector3d::new(4.0, 5.0, 6.0);
        assert_eq!(u + v, Vector3d::new(5.0, 7.0, 9.0));
    }

    #[test]
    fn test_vector3d_sub() {
        let u = Vector3d::new(1.0, 2.0, 3.0);
        let v = Vector3d::new(4.0, 5.0, 6.0);
        assert_eq!(u - v, Vector3d::new(-3.0, -3.0, -3.0));
    }

    #[test]
    fn test_vector3d_dot() {
        let u = Vector3d::new(1.0, 2.0, 3.0);
        let v = Vector3d::new(4.0, 5.0, 6.0);
        assert_eq!(u.dot(&v), 32.0);
    }

    #[test]
    fn test_vector3d_cross() {
        let u = Vector3d::new(1.0, 2.0, 3.0);
        let v = Vector3d::new(4.0, 5.0, 6.0);
        assert_eq!(u.cross(&v), Vector3d::new(-3.0, 6.0, -3.0));
    }

    #[test]
    fn test_vector3d_projection() {
        let u = Vector3d::new(1.0, 2.0, 3.0);
        let v = Vector3d::new(4.0, 5.0, 6.0);
        assert_eq!(
            v.project(&u),
            Vector3d::new(16.0 / 7.0, 32.0 / 7.0, 48.0 / 7.0)
        );
    }

    #[test]
    fn test_scalar_component_3d() {
        let u = Vector3d::new(1.0, 2.0, 3.0);
        let v = Vector3d::new(4.0, 5.0, 6.0);
        assert_eq!(u.scalar_comp(&v), 8.55235974119758);
    }

    #[test]
    fn test_zero_vector_3d_mag() {
        let v = Vector3d::new(0.0, 0.0, 0.0);
        assert_eq!(v.mag(), 0.0);
    }

    #[test]
    fn test_vectorn_add() {
        let u = VectorN::new(vec![1.0, 2.0, 3.0]);
        let v = VectorN::new(vec![4.0, 5.0, 6.0]);
        assert_eq!(u + v, VectorN::new(vec![5.0, 7.0, 9.0]));
    }

    #[test]
    fn test_vectorn_sub() {
        let u = VectorN::new(vec![1.0, 2.0, 3.0]);
        let v = VectorN::new(vec![4.0, 5.0, 6.0]);
        assert_eq!(u - v, VectorN::new(vec![-3.0, -3.0, -3.0]));
    }

    #[test]
    fn test_vectorn_dot() {
        let u = VectorN::new(vec![1.0, 2.0, 3.0]);
        let v = VectorN::new(vec![4.0, 5.0, 6.0]);
        assert_eq!(u.dot(&v), 32.0);
    }

    #[test]
    fn test_vectorn_projection() {
        let u = VectorN::new(vec![1.0, 2.0, 3.0]);
        let v = VectorN::new(vec![4.0, 5.0, 6.0]);
        assert_eq!(
            v.project(&u),
            VectorN::new(vec![16.0 / 7.0, 32.0 / 7.0, 48.0 / 7.0])
        );
    }

    #[test]
    fn test_scalar_component_n() {
        let u = VectorN::new(vec![1.0, 2.0, 3.0]);
        let v = VectorN::new(vec![4.0, 5.0, 6.0]);
        assert_eq!(u.scalar_comp(&v), 8.55235974119758);
    }

    #[test]
    fn test_zero_vector_n_mag() {
        let v = VectorN::new(vec![0.0, 0.0, 0.0]);
        assert_eq!(v.mag(), 0.0);
    }

    #[test]
    fn test_vector_n_add_size_difference() {
        let u = VectorN::new(vec![1.0, 2.0, 3.0]);
        let v = VectorN::new(vec![4.0, 5.0]);
        assert_eq!(u + v, VectorN::new(vec![5.0, 7.0, 3.0]));
    }

    #[test]
    fn test_vector_n_sub_size_difference() {
        let u = VectorN::new(vec![1.0, 2.0, 3.0]);
        let v = VectorN::new(vec![4.0, 5.0]);
        assert_eq!(u - v, VectorN::new(vec![-3.0, -3.0, 3.0]));
    }

    #[test]
    fn test_vector_n_dot_size_difference() {
        let u = VectorN::new(vec![1.0, 2.0, 3.0]);
        let v = VectorN::new(vec![4.0, 5.0]);
        let u_s = Vector3d::new(1.0, 2.0, 3.0);
        let v_s = Vector3d::new(4.0, 5.0, 0.0);
        assert_eq!(u.dot(&v), u_s.dot(&v_s));
    }

    #[test]
    fn test_matrix_add() {
        let u = Matrix::new(3usize, 3usize, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        let v = Matrix::new(3usize, 3usize, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        assert_eq!(u + v, Matrix::new(3usize, 3usize, vec![
            2.0, 4.0, 6.0,
            8.0, 10.0, 12.0,
            14.0, 16.0, 18.0,
        ]).unwrap());
    }

    #[test]
    fn test_matrix_sub() {
        let u = Matrix::new(3usize, 3usize, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        let v = Matrix::new(3usize, 3usize, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        assert_eq!(u - v, Matrix::new(3usize, 3usize, vec![
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0,
        ]).unwrap());
    }

    #[test]
    fn test_matrix_mul() {
        let u = Matrix::new(3usize, 3usize, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        let v = Matrix::new(3usize, 3usize, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        assert_eq!(u * v, Matrix::new(3usize, 3usize, vec![
            30.0, 36.0, 42.0,
            66.0, 81.0, 96.0,
            102.0, 126.0, 150.0,
        ]).unwrap());
    }

    #[test]
    fn test_matrix_mul_scalar() {
        let u = Matrix::new(3usize, 3usize, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        assert_eq!(u * 2.0, Matrix::new(3usize, 3usize, vec![
            2.0, 4.0, 6.0,
            8.0, 10.0, 12.0,
            14.0, 16.0, 18.0,
        ]).unwrap());
    }

    #[test]
    fn test_matrix_mul_vector() {
        let u = Matrix::new(3usize, 3usize, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        let v = VectorN::new(vec![1.0, 2.0, 3.0]);
        assert_eq!(u * v, VectorN::new(vec![14.0, 32.0, 50.0]));
    }

    #[test]
    #[should_panic]
    fn test_matrix_mul_vector_size_difference() {
        let u = Matrix::new(3usize, 3usize, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        let v = VectorN::new(vec![1.0, 2.0]);
        assert_eq!(u * v, VectorN::new_empty(1usize)); // Should panic!
    }

    #[test]
    fn test_matrix_inverse() {
        let u = Matrix::new(3usize, 3usize, vec![
            3.0, 0.0, 2.0,
            2.0, 0.0, -2.0,
            0.0, 1.0, 1.0,
        ]).unwrap();
        let v = Matrix::new(3usize, 3usize, vec![
            0.2, 0.2, 0.0,
            -0.2, 0.3, 1.0,
            0.2, -0.3, 0.0
        ]).unwrap();
        assert_eq!(u.inverse().unwrap(), v);
    }

    #[test]
    fn test_matrix_adjugate() {
        let u = Matrix::new(3usize, 3usize, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]).unwrap();
        let v = Matrix::new(3usize, 3usize, vec![
            1.0, 4.0, 7.0,
            2.0, 5.0, 8.0,
            3.0, 6.0, 9.0,
        ]).unwrap();
        assert_eq!(u.adjugate().unwrap(), v);
    }

    #[test]
    fn test_matrix_identity() {
        let u = Matrix::identity(3usize).unwrap();
        let v = Matrix::new(3usize, 3usize, vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]).unwrap();
        assert_eq!(u, v);
    }

    #[test]
    fn test_matrix_upper_triangle() {
        let u = Matrix::new(3usize, 3usize, vec![
            1.0, 2.0, 3.0,
            0.0, 4.0, 5.0,
            0.0, 0.0, 6.0,
        ]).unwrap();
        assert_eq!(u.triangular(), Triangular::Upper);
    }

    #[test]
    fn test_matrix_lower_triangle() {
        let u = Matrix::new(3usize, 3usize, vec![
            1.0, 0.0, 0.0,
            2.0, 4.0, 0.0,
            3.0, 5.0, 6.0,
        ]).unwrap();
        assert_eq!(u.triangular(), Triangular::Lower);
    }

    fn test_matrix_get_index() {
        let m = Matrix::new(2usize, 3usize, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        ]).unwrap();
        assert_eq!(m.get(0usize,0usize).unwrap(), 1.0);
        assert_eq!(m.get(0usize,1usize).unwrap(), 2.0);
        assert_eq!(m.get(0usize,2usize).unwrap(), 3.0);
        assert_eq!(m.get(1usize,0usize).unwrap(), 4.0);
        assert_eq!(m.get(1usize,1usize).unwrap(), 5.0);
        assert_eq!(m.get(1usize,2usize).unwrap(), 6.0);
    }

    #[test]
    fn test_matrix_dot() {
        let m1 = Matrix::new(2usize, 3usize, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0
        ]).unwrap();
        let m2 = Matrix::new(3usize, 2usize, vec![
            1.0, 2.0,
            3.0, 4.0,
            5.0, 6.0
        ]).unwrap();
        let result1 = Matrix::new(2usize, 2usize, vec![
            22.0, 28.0,
            49.0, 64.0
        ]).unwrap();

        let m3 = Matrix::new(2usize, 2usize, vec![
            1.0, 2.0,
            3.0, 4.0
        ]).unwrap();
        let result2 = Matrix::new(2usize, 2usize, vec![
            7.0, 10.0,
            15.0, 22.0
        ]).unwrap();
        assert_eq!(m1.dot(&m2).unwrap(), result1);
        assert_eq!(m3.dot(&m3).unwrap(), result2);
    }

    #[test]
    fn test_matrix_determinant() {
        // TODO - Fix determinant algorithm
        let m1 = Matrix::new(2usize, 2usize, vec![
            4.0, 6.0,
            3.0, 8.0
        ]).unwrap();
        let m2 = Matrix::new(3usize, 3usize, vec![
            6.0, 1.0, 1.0,
            4.0, -2.0, 5.0,
            2.0, 8.0, 7.0
        ]).unwrap();
        assert_eq!(m1.determinant().unwrap(), 14.0);
        assert_eq!(m2.determinant().unwrap(), -306.0);
    }

}

// TODO - Write rotate by angle for vectors
// TODO - Write docstrings for all functions and structs