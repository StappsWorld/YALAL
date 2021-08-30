use crate::vector::VectorN;

#[derive(Debug, Clone, Copy)]
pub enum Triangular {
    Upper,
    Lower,
    Not,
}

#[derive(Debug, Clone)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
    triangular: Triangular,
}
impl Matrix {
    pub fn new_empty<T: 'static + Into<usize> + Copy>(rows_raw: T, cols_raw: T) -> Matrix {
        let rows = rows_raw.into();
        let cols = cols_raw.into();
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
            triangular: Triangular::Not,
        }
    }

    pub fn new<T: 'static + Into<usize> + Copy>(
        rows_raw: T,
        cols_raw: T,
        data: Vec<f64>,
    ) -> Option<Matrix> {
        let rows = rows_raw.into();
        let cols = cols_raw.into();
        if rows * cols != data.len() {
            return None;
        }
        let mut m = Matrix {
            rows,
            cols,
            data,
            triangular: Triangular::Not,
        };
        m.update_triangular();
        Some(m)
    }

    pub fn identity<T: 'static + Into<usize> + Copy>(size_raw: T) -> Matrix {
        let size = size_raw.into();
        let mut m = Matrix::new_empty(size, size);
        for i in 0..size {
            m.set(i, i, 1.0);
        }
        m
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn data(&self) -> &Vec<f64> {
        &self.data
    }

    pub fn get<T: 'static + Into<usize> + Copy>(&self, row: T, col: T) -> Option<f64> {
        let row = row.into();
        let col = col.into();
        if row >= self.rows || col >= self.cols {
            return None;
        }

        Some(self.data[row * self.cols + col])
    }

    pub fn set<T: 'static + Into<usize> + Copy>(
        &mut self,
        row: T,
        col: T,
        value: f64,
    ) -> Option<()> {
        let row = row.into();
        let col = col.into();
        if row >= self.rows || col >= self.cols {
            return None;
        }
        self.data[row * self.cols + col] = value;
        Some(())
    }

    pub fn dot(&self, other: &Matrix) -> Option<Matrix> {
        if self.cols != other.rows {
            return None;
        }
        let mut result = Matrix::new_empty(self.rows, other.cols);
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k).unwrap() * other.get(k, j).unwrap();
                }
                result.set(i, j, sum);
            }
        }
        Some(result)
    }

    pub fn determinant(&self) -> f64 {
        if self.rows != self.cols {
            return 0.0;
        }
        if self.rows == 1 {
            return self.data[0];
        }
        if self.rows == 2 {
            return self.data[0] * self.data[3] - self.data[1] * self.data[2];
        }
        let mut sum = 0.0;
        for i in 0..self.cols {
            sum += self.data[i] * self.cofactor(0, i);
        }
        sum
    }

    pub fn cofactor(&self, row: usize, col: usize) -> f64 {
        let mut m = Matrix::new_empty(self.rows - 1, self.cols - 1);
        let mut i = 0;
        let mut j = 0;
        for r in 0..self.rows {
            if r == row {
                continue;
            }
            for c in 0..self.cols {
                if c == col {
                    continue;
                }
                m.data[i * (m.cols - 1) + j] = self.data[r * self.cols + c];
                j += 1;
            }
            i += 1;
            j = 0;
        }
        m.determinant() * if (row + col) % 2 == 0 { 1.0 } else { -1.0 }
    }

    pub fn eigenvalues_eigenvectors(&self) -> Option<Vec<(VectorN, f64)>> {
        if self.rows != self.cols {
            return None;
        }
        // See https://en.wikipedia.org/wiki/Eigenvalue_algorithm#Direct_calculation for this algorithm
        match self.rows {
            2 => {
                let trace = self.trace().unwrap();
                let val_pos = (trace + ((trace * trace) - (4.0 * self.determinant())).sqrt()) / 2.0;
                let val_neg = (trace - ((trace * trace) - (4.0 * self.determinant())).sqrt()) / 2.0;

                let vec_pos = self.clone() - Matrix::identity(self.rows) * val_pos;
                let vec_neg = self.clone() - Matrix::identity(self.rows) * val_neg;
                Some(vec![
                    (vec_pos.get_column(0).unwrap(), val_pos),
                    (vec_neg.get_column(1).unwrap(), val_neg),
                ])
            }
            3 => unimplemented!(),
            _ => unimplemented!(),
        }
    }

    pub fn update_triangular(&mut self) {
        if self.rows != self.cols {
            self.triangular = Triangular::Not;
            return;
        }

        let mut is_upper = true;
        let mut is_lower = true;
        for i in 0..self.rows {
            for j in 0..self.cols {
                if is_upper && i > j && self.get(i, j).unwrap() != 0.0 {
                    is_upper = false;
                }
                if is_lower && i < j && self.get(i, j).unwrap() != 0.0 {
                    is_lower = false;
                }
            }
        }

        if is_upper && is_lower {
            panic!("Matrix is both an upper and lower triangular matrix... Please take a look and submit a bug report...\n{}", self);
        }

        self.triangular = if is_upper {
            Triangular::Upper
        } else if is_lower {
            Triangular::Lower
        } else {
            Triangular::Not
        };
    }

    pub fn triangular(&self) -> Triangular {
        self.triangular
    }

    pub fn trace(&self) -> Option<f64> {
        if self.rows != self.cols {
            return None;
        }

        let mut sum = 0.0;
        for i in 0..self.rows {
            sum += self.get(i, i).unwrap();
        }
        Some(sum)
    }

    pub fn get_column(&self, col: usize) -> Option<VectorN> {
        if col >= self.cols {
            return None;
        }
        let mut vec = VectorN::new_empty(self.rows);
        for i in 0..self.rows {
            vec.set_component(i, self.get(i, col).unwrap()).unwrap();
        }
        Some(vec)
    }
}
impl std::ops::Mul<f64> for Matrix {
    type Output = Matrix;
    fn mul(self, rhs: f64) -> Matrix {
        let mut m = Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self.data.iter().map(|x| x * rhs).collect(),
            triangular: Triangular::Not,
        };
        m.update_triangular();
        m
    }
}
impl std::ops::MulAssign<f64> for Matrix {
    fn mul_assign(&mut self, rhs: f64) {
        for i in 0..self.data.len() {
            self.data[i] *= rhs;
        }
    }
}
impl std::ops::Mul<VectorN> for Matrix {
    type Output = VectorN;

    fn mul(self, rhs: VectorN) -> VectorN {
        if self.cols != rhs.size() {
            panic!("Matrix and Vector have different dimensions");
        }

        let mut result = VectorN::new_empty(self.rows);
        for i in 0..self.rows {
            let mut sum = 0.0;
            for j in 0..self.cols {
                sum += self.get(i, j).unwrap() * rhs.get_component(j).unwrap();
            }
            result.set_component(i, sum);
        }

        result
    }
}
impl std::ops::Mul<Matrix> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: Matrix) -> Matrix {
        if self.cols != rhs.rows {
            panic!("Matrices have different dimensions");
        }

        let mut result = Matrix::new_empty(self.rows, rhs.cols);
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k).unwrap() * rhs.get(k, j).unwrap();
                }
                result.set(i, j, sum);
            }
        }
        result
    }
}
impl std::ops::Add<Matrix> for Matrix {
    type Output = Matrix;
    fn add(self, rhs: Matrix) -> Matrix {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            panic!("Matrix dimensions do not match");
        }
        let mut m = Matrix {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(rhs.data.iter())
                .map(|(x, y)| x + y)
                .collect(),
            triangular: Triangular::Not,
        };
        m.update_triangular();
        m
    }
}
impl std::ops::AddAssign<Matrix> for Matrix {
    fn add_assign(&mut self, rhs: Matrix) {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            panic!("Matrix dimensions do not match");
        }
        for i in 0..self.data.len() {
            self.data[i] += rhs.data[i];
        }
    }
}
impl std::ops::Sub<Matrix> for Matrix {
    type Output = Matrix;
    fn sub(self, rhs: Matrix) -> Matrix {
        self + -rhs
    }
}
impl std::ops::SubAssign<Matrix> for Matrix {
    fn sub_assign(&mut self, rhs: Matrix) {
        *self += -rhs
    }
}
impl std::ops::Neg for Matrix {
    type Output = Matrix;
    fn neg(self) -> Matrix {
        self * -1.0
    }
}
impl std::cmp::PartialEq<Matrix> for Matrix {
    fn eq(&self, other: &Matrix) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            false
        } else {
            self.data.iter().zip(other.data.clone()).all(|(a, b)| (*a - b).abs() < 1e-10)
        }
    }
}
impl std::fmt::Display for Matrix {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut longest_string: usize = 0;
        let mut rows: Vec<String> = Vec::new();

        for i in 0..self.rows {
            let mut row_string = String::default();
            for j in 0..self.cols {
                if j == self.cols - 1 {
                    row_string.push_str(&format!("{:.2}", self.get(i, j).unwrap()));
                } else {
                    row_string.push_str(&format!("{:.2} ", self.get(i, j).unwrap()));
                }
            }
            if row_string.len() > longest_string {
                longest_string = row_string.len();
            }
            rows.push(row_string);
        }

        writeln!(f, "┌{:width$}┐", " ", width = longest_string)?;
        for i in 0..rows.len() {
            writeln!(f, "│{:width$}│", rows[i], width = longest_string)?;
        }
        writeln!(f, "└{:width$}┘", " ", width = longest_string)?;

        Ok(())
    }
}
