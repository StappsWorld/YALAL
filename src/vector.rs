use num_traits::cast::AsPrimitive;

#[derive(Debug, Clone, Copy, PartialOrd, Default)]
pub struct Vector {
    x_y: [f64; 2],
    heading: f64,
    mag: f64,
}
impl Vector {
    /// Creates a new Vector
    /// # Arguments
    /// * 'x' - The x component of this Vector
    /// * 'y' - The y component of this Vector
    ///
    /// # Example
    /// ```
    /// use yalal::vector::Vector;
    /// let v = Vector::new(2, 2);
    /// ```
    pub fn new<T: AsPrimitive<f64>>(x: T, y: T) -> Vector {
        let x: f64 = x.as_();
        let y: f64 = y.as_();
        let mut v = Vector {
            x_y: [x, y],
            heading: y.atan2(x).to_degrees(),
            mag: 0.0,
        };
        v.update_mag();
        v
    }

    /// Creates a standard unit Vector with an x and y component of 1
    ///
    /// # Example
    /// ```
    /// use yalal::vector::Vector;
    /// let v = Vector::standard_unit();
    /// assert_eq(v, Vector::new(1, 1));
    /// ```
    pub fn standard_unit() -> Vector {
        Vector::new(1, 1)
    }

    /// Creates a unit vector with the heading of the angle provided
    /// # Arguments
    /// * 'heading' - The heading of this unit vector provided in **degrees**
    ///
    /// # Example
    /// ```
    /// use yalal::vector::Vector;
    /// let v = Vector::from_angle(90);
    /// ```
    pub fn from_angle<T: AsPrimitive<f64>>(heading: T) -> Vector {
        let heading_deg: f64 = heading.as_();
        let mut v = Vector::default();
        v.mag = 1.0;
        v.set_heading(heading_deg);
        v
    }

    /// Creates a vector with a random x and y component, both ranging from -1.0 to 1.0
    ///
    /// # Example
    /// ```
    /// use yalal::vector::Vector;
    /// let v = Vector::random();
    /// ```
    pub fn random() -> Vector {
        let mut rng = rand::thread_rng();

        let x = rand::Rng::gen_range(&mut rng, -1.0..1.0);
        let y = rand::Rng::gen_range(&mut rng, -1.0..1.0);
        Vector::new(x, y)
    }

    /// Copies this vector's x and y component into a tuple.
    ///
    /// # Example
    /// ```
    /// use yalal::vector::Vector;
    /// let v = Vector::new(5, 10);
    /// let (x, y) = v.x_y();
    /// println!("This vector has the coordinates ({}, {})!", x, y); // Should print '... coordinates (5, 10)!'
    /// ```
    pub fn x_y(&self) -> (f64, f64) {
        (self.x_y[0], self.x_y[1])
    }

    /// Helper function to get reference to x component of this vector
    fn x(&mut self) -> &mut f64 {
        &mut self.x_y[0]
    }

    /// Helper function to get reference to y component of this vector
    fn y(&mut self) -> &mut f64 {
        &mut self.x_y[1]
    }

    /// Sets this vector's x component to the argument 'x'
    ///
    /// # Example
    /// ```
    /// use yalal::vector::Vector;
    /// let mut v = Vector::new(5, 10);
    /// println!("Our vector is {}", v); // Should print '... <5, 10>'
    /// v.set_x(12.0);
    /// println!("Our vector is now {}", v); // Should print '... <12, 10>'
    /// ```
    pub fn set_x<T: AsPrimitive<f64>>(&mut self, x: T) {
        *self = Vector::new(x.as_(), *self.y());
    }

    /// Sets this vector's y component to the argument 'y'
    /// # Arguments
    /// * 'y' - The y component of this Vector to be set
    ///
    /// # Example
    /// ```
    /// use yalal::vector::Vector;
    /// let mut v = Vector::new(5, 10);
    /// println!("Our vector is {}", v); // Should print '... <5.0, 10.0>'
    /// v.set_y(12.0);
    /// println!("Our vector is now {}", v); // Should print '... <5.0, 12.0>'
    /// ```
    pub fn set_y<T: AsPrimitive<f64>>(&mut self, y: T) {
        *self = Vector::new(*self.x(), y.as_());
    }

    /// Internally normalizes this vector.
    /// **Warning**: This will overwrite the internal components of this vector. If you want a normalized version of this vector without losing this vector, call `get_normalized()`
    /// # Example
    /// ```
    /// use yalal::vector::Vector;
    /// let mut v = Vector::new(3, 4);
    /// println!("Our vector is {}", v); // Should print '... <3.0, 4.0>'
    /// v.normalize();
    /// println!("Out vector is now {}", v); // Should print '... <0.6, 0.8>'
    /// ```
    pub fn normalize(&mut self) {
        for comp in self.x_y.iter_mut() {
            *comp /= self.mag;
        }
        self.mag = 1.0;
    }

    /// Generates a normalized version of this vector.
    /// # Example
    /// ```
    /// use yalal::vector::Vector;
    /// let v = Vector::new(3, 4);
    /// println!("The normalized version of v is {}", v.get_normalized()); // Should print '... <0.6, 0.8>'
    /// println!("v is still {}", v); // Should print '... <3.0, 4.0>'
    /// ```
    pub fn get_normalized(&self) -> Vector {
        let mut v = self.clone();
        v.normalize();
        v
    }

    /// Calculates the distance between this vector and another
    /// # Arguments
    /// * 'other' - The other vector to be measured
    /// # Example
    /// ```
    /// use yalal::vector::Vector;
    /// let u = Vector::new(5, 5);
    /// let v = Vector::new(2, 1);
    /// println!("The distance between u and v is {}", u.distance(v)); // Should print '... 5.0'
    /// ```
    pub fn distance(self, other: Vector) -> f64 {
        (self - other).mag
    }

    /// Sets the magnitude of this vector and updates the x and y components.
    /// **Warning**: This will overwrite the internal components of this vector!
    /// # Arguments
    /// * 'mag' - The magnitude to set this vector to
    /// # Example
    /// ```
    /// use yalal::vector::Vector;
    /// let mut v = Vector::new(3.0 / 5.0, 4.0 / 5.0);
    /// println!("Our vector is {}", v); // Should print '... <0.6, 0.8>
    /// v.set_mag(5.0);
    /// println!("Our vector is now {}", v); // Should print '... <3.0, 4.0>'
    /// ```
    pub fn set_mag<T: AsPrimitive<f64>>(&mut self, mag: T) {
        let mag = mag.as_();
        self.normalize();
        for comp in self.x_y.iter_mut() {
            *comp *= mag;
        }
        self.mag = mag;
    }

    /// Returns the magnitude of this function
    /// # Example
    /// ```
    /// use yalal::vector::Vector;
    /// let v = Vector::new(3.0, 4.0);
    /// println!("The magnitude of v is {}", v.mag()); // Should print '... 5.0'
    /// ```
    pub fn mag(&self) -> f64 {
        self.mag
    }

    fn update_mag(&mut self) {
        let mut sum = 0.0;
        for comp in self.x_y.iter() {
            sum += comp.powi(2);
        }
        self.mag = sum.sqrt();
    }

    /// Returns the heading of the vector in **degrees**
    /// # Example
    /// ```
    /// use yalal::vector::Vector;
    /// let v = Vector::new(0.0, 1.0);
    /// println!("The heading of this vector is {} degrees!", v.heading()); // Should print '... 90.0 degrees!'
    /// ```
    pub fn heading(&self) -> f64 {
        self.heading
    }

    /// Sets this vectors heading to the angle provided and updates its x and y components
    /// # Arguments
    /// * 'heading' - The angle that this vector will be set to
    /// # Example
    /// ```
    /// use yalal::vector::Vector;
    /// let mut v = Vector::new(1.0, 0.0);
    /// println!("Our vector {} has an angle of {}", v, v.heading()); // Should print '... <1.0, 0.0> ... 0.0'
    /// v.set_heading(90.0);
    /// println!("Our vector is now {} and has an angle of {}", v, v.heading()); // Should print '... <0.0, 1.0> ... 90.0'
    /// ```
    pub fn set_heading<T: AsPrimitive<f64>>(&mut self, heading: T) {
        let heading = heading.as_();
        let radians = heading.to_radians();

        let mut x = radians.cos();

        if x - x.floor() < 1e-10 {
            x = x.floor();
        }

        let mut y = radians.sin();

        if y - y.floor() < 1e-10 {
            y = y.floor();
        }

        assert_eq!((x * x + y * y).sqrt(), 1.0);
        self.x_y = [x, y];
        self.heading = heading;
    }

    /// Adds to this vectors current heading by the angle provided
    /// # Arguments
    /// * 'angle' - The angle to add to this vector's heading
    /// # Example
    /// ```
    /// use yalal::vector::Vector;
    /// let mut v = Vector::new(1.0, 0.0);
    /// println!("Our vector {} has an angle of {}", v, v.heading()); // Should print '... <1.0, 0.0> ... 0.0'
    /// v.rotate(90.0);
    /// println!("Our vector is now {} and has an angle of {}", v, v.heading()); // Should print '... <0.0, 1.0> ... 90.0'
    /// ```
    pub fn rotate<T: AsPrimitive<f64>>(&mut self, angle: T) {
        self.set_heading(self.heading() + angle.as_());
    }

    pub fn limit_mag<T: AsPrimitive<f64>>(&mut self, max: T) {
        let max = max.as_();
        if self.mag > max * max {
            self.set_mag(max);
        }
    }

    pub fn angle_between(&self, other: &Vector) -> f64 {
        self.cos_angle_between(other).acos().to_degrees()
    }

    pub fn dot(&self, other: &Vector) -> f64 {
        let mut sum = 0.0;
        for (a, b) in self.x_y.iter().zip(other.x_y.iter()) {
            sum += a * b;
        } // Using Rust optimization will allow for SIMD optimizations here
        sum
    }

    pub fn cos_angle_between(&self, other: &Vector) -> f64 {
        self.dot(other) / (self.mag() * other.mag())
    }

    // This function returns the vector projection of self onto other
    pub fn project(&self, other: Vector) -> Vector {
        let dot = self.dot(&other);
        let mag = other.mag * other.mag;
        let frac = dot / mag;
        other * frac
    }

    // This function returns the scalar component of self in the direction of other
    pub fn scalar_comp(&self, other: &Vector) -> f64 {
        other.mag() * other.cos_angle_between(self)
    }

    // Functionally the same as self * other
    pub fn cross(&self, other: &Vector) -> Vector3d {
        Vector3d::new(
            0.0,
            0.0,
            self.mag() * other.mag() * self.angle_between(other).to_radians().sin(),
        )
    }

    pub fn angle_given_dot<T: AsPrimitive<f64>>(mag_a: T, mag_b: T, dot: T) -> f64 {
        let mag_a = mag_a.as_();
        let mag_b = mag_b.as_();
        let dot = dot.as_();
        let mag = mag_a * mag_b;
        (dot / mag).acos().to_degrees()
    }

    pub fn angle_given_cross<T: AsPrimitive<f64>>(mag_a: T, mag_b: T, cross: T) -> f64 {
        let mag_a = mag_a.as_();
        let mag_b = mag_b.as_();
        let cross = cross.as_();
        let mag = mag_a * mag_b;
        (cross / mag).asin().to_degrees()
    }
}
impl std::ops::Add<Vector> for Vector {
    fn add(self, other: Vector) -> Vector {
        let mut out = [0.0; 2];
        for (n, (a, b)) in self.x_y.iter().zip(other.x_y.iter()).enumerate() {
            out[n] = a + b;
        }
        Vector::from(out)
    }

    type Output = Vector;
}
impl std::ops::Sub<Vector> for Vector {
    fn sub(self, other: Vector) -> Vector {
        let mut out = [0.0; 2];
        for (n, (a, b)) in self.x_y.iter().zip(other.x_y.iter()).enumerate() {
            out[n] = a - b;
        }
        Vector::from(out)
    }

    type Output = Vector;
}
impl std::ops::AddAssign for Vector {
    fn add_assign(&mut self, other: Vector) {
        *self = *self + other;
    }
}
impl std::ops::SubAssign for Vector {
    fn sub_assign(&mut self, other: Vector) {
        *self = *self - other;
    }
}
impl std::ops::Mul<f64> for Vector {
    type Output = Vector;

    fn mul(self, rhs: f64) -> Vector {
        let mut out = [0.0; 2];
        for (n, a) in self.x_y.iter().enumerate() {
            out[n] = a * rhs;
        }
        Vector::from(out)
    }
}
impl std::ops::Mul<Vector> for Vector {
    type Output = Vector3d;

    fn mul(self, other: Vector) -> Vector3d {
        Vector3d::new(
            0.0,
            0.0,
            self.mag() * other.mag() * self.angle_between(&other).to_radians().sin(),
        )
    }
}
impl std::ops::MulAssign<f64> for Vector {
    fn mul_assign(&mut self, rhs: f64) {
        *self = *self * rhs;
    }
}
impl std::ops::Div<f64> for Vector {
    type Output = Vector;

    fn div(self, other: f64) -> Vector {
        self * (1.0 / other)
    }
}
impl std::ops::DivAssign<f64> for Vector {
    fn div_assign(&mut self, rhs: f64) {
        *self *= 1.0 / rhs;
    }
}
impl std::ops::Neg for Vector {
    type Output = Vector;
    fn neg(self) -> Vector {
        self * -1.0
    }
}
impl From<Vector3d> for Vector {
    fn from(v: Vector3d) -> Vector {
        Vector::new(v.x, v.y)
    }
}
impl From<VectorN> for Vector {
    fn from(v: VectorN) -> Vector {
        Vector::new(
            v.get_component(0).unwrap_or(0.0),
            v.get_component(1).unwrap_or(0.0),
        )
    }
}
impl From<[f64; 2]> for Vector {
    fn from(v: [f64; 2]) -> Vector {
        Vector::new(v[0], v[1])
    }
}
impl std::cmp::PartialEq for Vector {
    fn eq(&self, other: &Vector) -> bool {
        self.x_y
            .iter()
            .zip(other.x_y.iter())
            .all(|(a, b)| a - b < 1e-10)
    }
}
impl std::fmt::Display for Vector {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let (x, y) = self.x_y();
        write!(f, "<{}, {}>", x, y)
    }
}

#[derive(Debug, Clone, Copy, PartialOrd, Default)]
pub struct Vector3d {
    x: f64,
    y: f64,
    z: f64,
    heading: (f64, f64),
    mag: f64,
}
impl Vector3d {
    /// Creates a new Vector3d
    /// # Arguments
    /// * 'x' - The x component of this Vector
    /// * 'y' - The y component of this Vector
    /// * 'z' - The z component of this Vector
    ///
    /// # Example
    /// ```
    /// use yalal::vector::Vector3d;
    /// let v = Vector::new(2, 2, 2);
    /// ```
    pub fn new<T: AsPrimitive<f64>>(x: T, y: T, z: T) -> Vector3d {
        let x: f64 = x.as_();
        let y: f64 = y.as_();
        let z: f64 = z.as_();
        let mag = (x * x + y * y + z * z).sqrt();
        Vector3d {
            x: x,
            y: y,
            z: z,
            heading: ((y / x).atan(), (z / mag).acos()),
            mag: mag,
        }
    }

    /// Generates a Vector3d from a theta and phi in degrees.
    /// See [this](https://en.wikipedia.org/wiki/Spherical_coordinate_system#Cartesian_coordinates) for more information.
    /// # Arguments
    /// * 'x' - The x component of this Vector
    /// * 'y' - The y component of this Vector
    /// * 'z' - The z component of this Vector
    ///
    /// # Example
    /// ```
    /// use yalal::vector::Vector3d;
    /// let v = Vector3d::from_heading(90, 45);
    /// ```
    pub fn from_heading<T: AsPrimitive<f64>>(theta: T, phi: T) -> Vector3d {
        let theta_deg: f64 = theta.as_();
        let theta: f64 = theta_deg.to_radians();
        let phi_deg: f64 = phi.as_();
        let phi: f64 = phi_deg.to_radians();
        let (x, y, z) = (theta.cos() * phi.sin(), theta.sin() * phi.sin(), phi.cos());
        Vector3d {
            x,
            y,
            z,
            heading: (theta, phi),
            mag: 1.0,
        }
    }

    /// Generates a random Vector3d with all coordinates ranging from -1.0 to 1.0
    ///
    /// # Example
    /// ```
    /// use yalal::vector::Vector3d;
    /// let v = Vector3d::random();
    /// ```
    pub fn random() -> Vector3d {
        let mut rng = rand::thread_rng();

        let x: f64 = rand::Rng::gen_range(&mut rng, -1.0..1.0);
        let y: f64 = rand::Rng::gen_range(&mut rng, -1.0..1.0);
        let z: f64 = rand::Rng::gen_range(&mut rng, -1.0..1.0);
        Vector3d::new(x, y, z)
    }

    /// Returns this Vectors x, y, and z components in a tuple of f64s
    ///
    /// # Example
    /// ```
    /// use yalal::vector::Vector3d;
    /// let v = Vector3d::new(1, 2, 3);
    /// let (x, y, z) = v.x_y_z();
    /// ```
    pub fn x_y_z(&self) -> (f64, f64, f64) {
        (self.x, self.y, self.z)
    }

    /// Sets this vector's x component to the argument 'x'
    ///
    /// # Example
    /// ```
    /// use yalal::vector::Vector3d;
    /// let mut v = Vector3d::new(5, 10, 15);
    /// v.set_x(12.0);
    /// assert_eq!(v, Vector3d::new(12, 10, 15));
    /// ```
    pub fn set_x<T: AsPrimitive<f64>>(&mut self, x: T) {
        let x: f64 = x.as_();
        *self = Vector3d::new(x, self.y, self.z);
    }

    /// Sets this vector's y component to the argument 'y'
    ///
    /// # Example
    /// ```
    /// use yalal::vector::Vector3d;
    /// let mut v = Vector3d::new(5, 10, 15);
    /// v.set_y(12.0);
    /// assert_eq!(v, Vector3d::new(5, 12, 15));
    /// ```
    pub fn set_y<T: AsPrimitive<f64>>(&mut self, y: T) {
        let y: f64 = y.as_();
        *self = Vector3d::new(self.x, y, self.z);
    }

    /// Sets this vector's z component to the argument 'z'
    ///
    /// # Example
    /// ```
    /// use yalal::vector::Vector3d;
    /// let mut v = Vector3d::new(5, 10, 15);
    /// v.set_z(12.0);
    /// assert_eq!(v, Vector3d::new(5, 10, 12));
    /// ```
    pub fn set_z<T: AsPrimitive<f64>>(&mut self, z: T) {
        let z: f64 = z.as_();
        *self = Vector3d::new(self.x, self.y, z);
    }

    /// Sets this vector's theta
    ///
    /// # Example
    /// ```
    /// use yalal::vector::Vector3d;
    /// let mut v = Vector3d::new(3, 4, 1);
    /// v.set_theta(90);
    // TODO - Fix this
    /// assert_eq!(v, Vector3d::new(0, 5, 1));
    /// ```
    pub fn set_theta<T: AsPrimitive<f64>>(&mut self, theta: T) {
        self.set_heading(theta.as_(), self.heading.1);
    }

    // This function sets phi (angle from y axis [v.y, v.z]) in degrees.
    pub fn set_phi<T: AsPrimitive<f64>>(&mut self, phi_raw: T) {
        self.set_heading(self.heading.0, phi_raw.as_());
    }

    // This function sets the heading in (theta, phi) format in degrees.
    pub fn set_heading<T: AsPrimitive<f64>>(&mut self, theta: T, phi: T) {
        *self = Vector3d::from_heading(theta, phi) * self.mag;
    }

    // A function that normalizes this vector.
    pub fn normalize(&mut self) {
        self.x /= self.mag;
        self.y /= self.mag;
        self.z /= self.mag;
        self.mag = 1.0;
    }

    pub fn get_normalized(&self) -> Vector3d {
        let mut v = self.clone();
        v.normalize();
        v
    }

    // A function that finds the distance between this vector and another.
    pub fn distance(self, other: Vector3d) -> f64 {
        (self - other).mag
    }

    pub fn set_mag(&mut self, mag: f64) {
        self.normalize();
        *self *= mag;
    }

    pub fn mag(&self) -> f64 {
        self.mag
    }

    pub fn heading(&self) -> (f64, f64) {
        self.heading
    }

    pub fn limit_mag(&mut self, max: f64) {
        if self.mag > max * max {
            self.set_mag(max);
        }
    }

    // This angle returns the angle between two Vector3d's in degrees.
    pub fn angle_between(&self, other: &Vector3d) -> f64 {
        (1.0 / (self.dot(other) / ((self.mag * other.mag).sqrt())).cos()).to_degrees()
    }

    pub fn cos_angle_between(&self, other: &Vector3d) -> f64 {
        self.dot(other) / (self.mag * other.mag)
    }

    pub fn dot(&self, other: &Vector3d) -> f64 {
        (self.x * other.x) + (self.y * other.y) + (self.z * other.z)
    }

    // This function returns the vector projection of self onto other
    pub fn project(&self, other: &Vector3d) -> Vector3d {
        let dot = self.dot(other);
        let mag = other.mag * other.mag;
        let frac = dot / mag;
        *other * frac
    }

    // This function returns the scalar component of self in the direction of other
    pub fn scalar_comp(&self, other: &Vector3d) -> f64 {
        other.mag() * other.cos_angle_between(self)
    }

    // Functionally the same as self * other
    pub fn cross(&self, other: &Vector3d) -> Vector3d {
        let x = (self.y * other.z) - (self.z * other.y);
        let y = (self.z * other.x) - (self.x * other.z);
        let z = (self.x * other.y) - (self.y * other.x);
        Vector3d::new(x, y, z)
    }
}
impl std::ops::Add<Vector3d> for Vector3d {
    fn add(self, other: Vector3d) -> Vector3d {
        let x = self.x + other.x;
        let y = self.y + other.y;
        let z = self.z + other.z;
        Vector3d::new(x, y, z)
    }

    type Output = Vector3d;
}
impl std::ops::Sub<Vector3d> for Vector3d {
    fn sub(self, other: Vector3d) -> Vector3d {
        let x = self.x - other.x;
        let y = self.y - other.y;
        let z = self.z - other.z;
        Vector3d::new(x, y, z)
    }

    type Output = Vector3d;
}
impl std::ops::AddAssign for Vector3d {
    fn add_assign(&mut self, other: Vector3d) {
        let x = self.x + other.x;
        let y = self.y + other.y;
        let z = self.z + other.z;
        *self = Vector3d::new(x, y, z);
    }
}
impl std::ops::SubAssign for Vector3d {
    fn sub_assign(&mut self, other: Vector3d) {
        let x = self.x - other.x;
        let y = self.y - other.y;
        let z = self.z - other.z;
        *self = Vector3d::new(x, y, z);
    }
}
impl std::ops::Mul<f64> for Vector3d {
    type Output = Vector3d;

    fn mul(self, other: f64) -> Vector3d {
        let x = self.x * other;
        let y = self.y * other;
        let z = self.z * other;
        Vector3d::new(x, y, z)
    }
}
impl std::ops::Mul<Vector3d> for Vector3d {
    type Output = Vector3d;

    fn mul(self, other: Vector3d) -> Vector3d {
        let x = (self.y * other.z) - (self.z * other.y);
        let y = (self.z * other.x) - (self.x * other.z);
        let z = (self.x * other.y) - (self.y * other.x);
        Vector3d::new(x, y, z)
    }
}
impl std::ops::Div<f64> for Vector3d {
    type Output = Vector3d;

    fn div(self, other: f64) -> Vector3d {
        self * (1.0 / other)
    }
}
impl std::ops::DivAssign<f64> for Vector3d {
    fn div_assign(&mut self, other: f64) {
        *self *= 1.0 / other;
    }
}
impl std::ops::MulAssign<f64> for Vector3d {
    fn mul_assign(&mut self, rhs: f64) {
        let x = self.x * rhs;
        let y = self.y * rhs;
        let z = self.z * rhs;
        *self = Vector3d::new(x, y, z);
    }
}
impl std::ops::Neg for Vector3d {
    type Output = Vector3d;
    fn neg(self) -> Vector3d {
        let x = -self.x;
        let y = -self.y;
        let z = -self.z;
        Vector3d::new(x, y, z)
    }
}
impl From<Vector> for Vector3d {
    fn from(vec: Vector) -> Vector3d {
        Vector3d::new(vec.x_y().0, vec.x_y().1, 0.0)
    }
}
impl From<VectorN> for Vector3d {
    fn from(vec: VectorN) -> Vector3d {
        Vector3d::new(
            vec.get_component(0).unwrap_or(0.0),
            vec.get_component(1).unwrap_or(0.0),
            vec.get_component(2).unwrap_or(0.0),
        )
    }
}
impl std::cmp::PartialEq for Vector3d {
    fn eq(&self, other: &Vector3d) -> bool {
        // Equal with an error margin of 0.0000000001 :)
        let fuzzy_eq = |a: f64, b: f64| a == b || (a - b).abs() < 1e-10;

        fuzzy_eq(self.x, other.x) && fuzzy_eq(self.y, other.y) && fuzzy_eq(self.z, other.z)
    }
}
impl std::fmt::Display for Vector3d {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "<{}, {}, {}>", self.x, self.y, self.z)
    }
}

#[derive(Debug, Clone, PartialOrd, Default)]
pub struct VectorN {
    data: Vec<f64>,
    mag: f64,
    angles: Vec<f64>,
}
impl VectorN {
    pub fn new<T: AsPrimitive<f64>>(data: Vec<T>) -> VectorN {
        let mut v = VectorN {
            data: data.iter().map(|x| x.as_()).collect(),
            mag: 0.0,
            angles: vec![0.0; data.len() - 1],
        };
        v.update_mag();
        v.update_angles();
        v
    }

    pub fn new_empty(size: usize) -> VectorN {
        VectorN {
            data: vec![0.0; size],
            mag: 0.0,
            angles: vec![0.0; size - 1],
        }
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }

    pub fn resize(&mut self, size: usize) {
        self.data.resize(size, 0.0);
        self.update_angles();
        self.update_mag();
    }

    pub fn get_component(&self, index: usize) -> Option<f64> {
        if index >= self.data.len() {
            None
        } else {
            Some(self.data[index])
        }
    }

    pub fn set_component<T: AsPrimitive<f64>>(&mut self, index: usize, val: T) -> Option<()> {
        if index >= self.data.len() {
            return None;
        }
        self.data[index] = val.as_();
        Some(())
    }

    pub fn get_angles(&self) -> &Vec<f64> {
        &self.angles
    }

    pub fn get_angle(&self, index: usize) -> Option<f64> {
        if index >= self.angles.len() {
            None
        } else {
            Some(self.angles[index].to_degrees())
        }
    }

    fn update_angles(&mut self) {
        // Algorithm from: https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
        for i in 0..self.angles.len() {
            self.angles[i] = if i == self.angles.len() - 1 {
                let current_x = self.data[i + 1];
                let last_x = self.data[i];
                let algo = (last_x / (current_x * current_x - last_x * last_x).sqrt()).acos();
                if current_x >= 0.0 {
                    algo
                } else {
                    const TWO_PI: f64 = 2.0 * std::f64::consts::PI;
                    TWO_PI - algo
                }
            } else {
                let current_x = self.data[i];
                let mut denom = 0.0;
                for j in i + 1..self.data.len() {
                    denom += self.data[j] * self.data[j];
                }
                current_x / denom.sqrt()
            }
        }
    }

    pub fn set_angle<T: AsPrimitive<f64>>(&mut self, index: usize, angle: T) -> Option<()> {
        // Algorithm from: https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates
        let angle = angle.as_();
        if index >= self.angles.len() {
            None
        } else {
            self.angles[index] = angle;
            for i in 0..self.data.len() {
                let mut product = self.mag;
                if i == self.data.len() - 1 {
                    for j in 0..self.angles.len() {
                        product *= self.angles[j].sin();
                    }
                } else {
                    for j in 0..i {
                        if j == i - 1 {
                            product *= self.angles[j].cos();
                        } else {
                            product *= self.angles[j].sin();
                        }
                    }
                }
                self.data[i] = product;
            }
            Some(())
        }
    }

    pub fn dot(&self, other: &VectorN) -> f64 {
        let mut sum = 0.0;
        let largest = std::cmp::max(self.data.len(), other.data.len());
        for i in 0..largest {
            sum += self.data.get(i).unwrap_or(&0.0) * other.data.get(i).unwrap_or(&0.0);
        }
        sum
    }

    pub fn mag(&self) -> f64 {
        self.mag
    }

    pub fn update_mag(&mut self) {
        self.mag = self.dot(&self).sqrt();
    }

    pub fn normalize(&mut self) {
        let mag = self.mag();
        for i in 0..self.size() {
            self.data[i] /= mag;
        }
    }

    pub fn cos_angle_between(&self, other: &VectorN) -> f64 {
        self.dot(other) / (self.mag() * other.mag())
    }

    // This function returns the vector projection of self onto other
    pub fn project(&self, other: &VectorN) -> VectorN {
        let dot = self.dot(other);
        let mag = other.mag * other.mag;
        let frac = dot / mag;
        other.clone() * frac
    }

    // This function returns the scalar component of self in the direction of other
    pub fn scalar_comp(&self, other: &VectorN) -> f64 {
        other.mag() * other.cos_angle_between(self)
    }

    // I may implement cross product later. Please see http://sciencewise.info/media/pdf/1408.5799v1.pdf
}
impl std::ops::Add<VectorN> for VectorN {
    fn add(self, other: VectorN) -> VectorN {
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        VectorN::new(data)
    }

    type Output = VectorN;
}
impl std::ops::Sub<VectorN> for VectorN {
    fn sub(self, other: VectorN) -> VectorN {
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();
        VectorN::new(data)
    }

    type Output = VectorN;
}
impl std::ops::AddAssign for VectorN {
    fn add_assign(&mut self, other: VectorN) {
        *self = self.clone() + other;
    }
}
impl std::ops::SubAssign for VectorN {
    fn sub_assign(&mut self, other: VectorN) {
        *self = self.clone() + other;
    }
}
impl std::ops::Mul<f64> for VectorN {
    type Output = VectorN;

    fn mul(self, rhs: f64) -> VectorN {
        let data = self.data.iter().map(|a| a * rhs).collect();
        VectorN::new(data)
    }
}
impl std::ops::MulAssign<f64> for VectorN {
    fn mul_assign(&mut self, rhs: f64) {
        *self = self.clone() * rhs;
    }
}
impl std::ops::Div<f64> for VectorN {
    type Output = VectorN;

    fn div(self, rhs: f64) -> VectorN {
        self * (1.0 / rhs)
    }
}
impl std::ops::DivAssign<f64> for VectorN {
    fn div_assign(&mut self, rhs: f64) {
        *self = self.clone() / rhs;
    }
}
impl std::ops::Neg for VectorN {
    type Output = VectorN;
    fn neg(self) -> VectorN {
        self * -1.0
    }
}
impl From<Vector> for VectorN {
    fn from(vec: Vector) -> VectorN {
        VectorN::new(vec.x_y.to_vec())
    }
}
impl From<Vector3d> for VectorN {
    fn from(vec: Vector3d) -> VectorN {
        VectorN::new(vec![vec.x, vec.y, vec.z])
    }
}
impl std::cmp::PartialEq for VectorN {
    fn eq(&self, other: &VectorN) -> bool {
        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(a, b)| a - b < 1e-10)
    }
}
impl std::fmt::Display for VectorN {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "<")?;
        for i in 0..self.size() {
            if i == self.size() - 1 {
                write!(f, "{}", self.data[i])?;
            } else {
                write!(f, "{}, ", self.data[i])?;
            }
        }
        write!(f, ">")
    }
}
