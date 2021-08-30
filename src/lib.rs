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
//! let m = Matrix::new(2u16, 2u16, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
//! let n = Matrix::new(2u16, 2u16, vec![5.0, 6.0, 7.0, 8.0]).unwrap();
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
    use crate::{matrix::*, vector::*};

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
}
