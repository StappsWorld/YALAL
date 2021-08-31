use crate::vector::VectorN;
use std::convert::TryInto;

#[derive(Debug, Clone, Copy, PartialEq)]
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
    pub fn new_empty<T: TryInto<usize>>(rows_raw: T, cols_raw: T) -> Result<Matrix, &'static str> {
        let rows = match rows_raw.try_into() {
            Ok(r) => r,
            Err(_) => return Err("Failed to convert rows into usize!"),
        };
        let cols = match cols_raw.try_into() {
            Ok(c) => c,
            Err(_) => return Err("Failed to convert cols into usize!"),
        };
        Ok(Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
            triangular: Triangular::Not,
        })
    }

    pub fn new<T: TryInto<usize>>(
        rows_raw: T,
        cols_raw: T,
        data: Vec<f64>,
    ) -> Result<Matrix, &'static str> {
        let rows = match rows_raw.try_into() {
            Ok(r) => r,
            Err(_) => return Err("Failed to convert rows into usize!"),
        };
        let cols = match cols_raw.try_into() {
            Ok(r) => r,
            Err(_) => return Err("Failed to convert cols into usize!"),
        };
        if rows * cols != data.len() {
            return Err(
                "rows and cols are not the correct dimensions for a matrix with the provided data!",
            );
        }
        let mut m = Matrix {
            rows,
            cols,
            data,
            triangular: Triangular::Not,
        };
        m.update_triangular();
        Ok(m)
    }

    pub fn identity<T: TryInto<usize>>(size_raw: T) -> Option<Matrix> {
        let size = match size_raw.try_into() {
            Ok(s) => s,
            Err(_) => return None,
        };
        let mut m = Matrix::new_empty(size, size).unwrap();
        for i in 0..size {
            m.set(i, i, 1.0).unwrap();
        }
        Some(m)
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

    pub fn get<T: TryInto<usize>>(&self, row: T, col: T) -> Result<f64, &'static str> {
        let row = match row.try_into() {
            Ok(r) => r,
            Err(_) => return Err("Failed to convert row into usize!"),
        };
        let col = match col.try_into() {
            Ok(r) => r,
            Err(_) => return Err("Failed to convert col into usize!"),
        };
        if row >= self.rows {
            return Err("Parameter row was greater than the amount of rows in this matrix!");
        }
        if col >= self.cols {
            return Err("Parameter col was greater than the amount of cols in this matrix!");
        }

        Ok(self.data[row * self.cols + col])
    }

    pub fn get_mut<T: TryInto<usize>>(&mut self, row: T, col: T) -> Result<&mut f64, &'static str> {
        let row = match row.try_into() {
            Ok(r) => r,
            Err(_) => return Err("Failed to convert row into usize!"),
        };
        let col = match col.try_into() {
            Ok(r) => r,
            Err(_) => return Err("Failed to convert col into usize!"),
        };
        if row >= self.rows {
            return Err("Parameter row was greater than the amount of rows in this matrix!");
        }
        if col >= self.cols {
            return Err("Parameter col was greater than the amount of cols in this matrix!");
        }

        Ok(self.data.get_mut(row * self.cols + col).unwrap())
    }

    pub fn set<T: TryInto<usize>>(&mut self, row: T, col: T, value: f64) -> Result<(), &'static str> {
        let row = match row.try_into() {
            Ok(r) => r,
            Err(_) => return Err("Failed to convert row into usize!"),
        };
        let col = match col.try_into() {
            Ok(r) => r,
            Err(_) => return Err("Failed to convert col into usize!"),
        };
        if row >= self.rows {
            return Err("Parameter row was greater than the amount of rows in this matrix!");
        }
        if col >= self.cols {
            return Err("Parameter col was greater than the amount of cols in this matrix!");
        }
        self.data[row * self.cols + col] = value;
        Ok(())
    }

    pub fn dot(&self, other: &Matrix) -> Option<Matrix> {
        if self.cols != other.rows {
            return None;
        }
        let mut result = Matrix::new_empty(self.rows, other.cols).unwrap();
        for i in 0..self.rows {
            for j in 0..other.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k).unwrap() * other.get(k, j).unwrap();
                }
                result.set(i, j, sum).unwrap();
            }
        }
        Some(result)
    }

    /// Calculates the determinant of this matrix.
    pub fn determinant(&self) -> Option<f64> {
        if self.rows != self.cols {
            return None;
        }
        if self.rows == 1 {
            return Some(self.data[0]);
        }
        if self.rows == 2 {
            return Some(self.data[0] * self.data[3] - self.data[1] * self.data[2]);
        }
        let mut sum = 0.0;
        for i in 0..self.cols {
            sum += self.get(0, i).unwrap()
                * self.minor(0, i).unwrap()
                * if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        Some(sum)
    }

    /// Produces the minor, or the determinant of the submatrix denoted by removing row and col from this matrix.
    pub fn minor(&self, row: usize, col: usize) -> Option<f64> {
        if self.rows != self.cols {
            return None;
        }
        let mut m = Matrix::new_empty(self.rows - 1, self.cols - 1).unwrap();
        let mut i: usize = 0;
        let mut j: usize = 0;
        for r in 0..self.rows {
            if r == row {
                continue;
            }
            for c in 0..self.cols {
                if c == col {
                    continue;
                }
                m.set(i, j, self.get(r, c).unwrap()).unwrap();
                j += 1;
            }
            i += 1;
            j = 0;
        }
        m.determinant()
    }

    /// Produces a matrix of "minors," which are the determinants of all the possible matrices with one row and one column removed inside this matrix.
    pub fn minors(&self) -> Option<Matrix> {
        if self.rows != self.cols {
            return None;
        }
        let mut m = Matrix::new_empty(self.rows, self.cols).unwrap();
        for i in 0..self.rows {
            for j in 0..self.cols {
                m.set(i, j, self.minor(i, j).unwrap()).unwrap();
            }
        }
        Some(m)
    }

    /// Swaps all elements over the diagonal
    pub fn adjugate(&self) -> Option<Matrix> {
        if self.rows != self.cols {
            return None;
        }
        let mut m = Matrix::new_empty(self.rows, self.cols).unwrap();
        for i in 0..self.rows {
            for j in 0..self.cols {
                if i == j {
                    m.set(i, j, self.get(i, j).unwrap()).unwrap();
                } else {
                    m.set(i, j, self.get(j, i).unwrap()).unwrap();
                }
            }
        }
        Some(m)
    }

    pub fn cofactor(&self) -> Option<Matrix> {
        if self.rows != self.cols {
            return None;
        }
        let mut m = Matrix::new_empty(self.rows, self.cols).unwrap();
        for i in 0..self.rows {
            for j in 0..self.cols {
                m.set(i, j, self.cofactor_val(i, j).unwrap()).unwrap();
            }
        }
        Some(m)
    }

    pub fn cofactor_val(&self, row: usize, col: usize) -> Option<f64> {
        if row >= self.rows || col >= self.cols {
            return None;
        }
        let val = self.get(row, col).unwrap() * if (row + col) % 2 == 0 { 1.0 } else { -1.0 };
        Some(val)
    }

    pub fn inverse(&self) -> Option<Matrix> {
        // See https://www.mathsisfun.com/algebra/matrix-inverse-minors-cofactors-adjugate.html for algorithm
        if self.rows != self.cols {
            return None;
        }
        let mom = self.minors().unwrap();
        let moc = mom.cofactor().unwrap();
        let adc = moc.adjugate().unwrap();
        let det = self.determinant().unwrap();
        if det == 0.0 {
            return None;
        }
        Some(adc / det)
    }

    pub fn eigenvalues_eigenvectors(&self) -> Option<Vec<(VectorN, f64)>> {
        if self.rows != self.cols {
            return None;
        }
        // See https://en.wikipedia.org/wiki/Eigenvalue_algorithm#Direct_calculation for this algorithm
        match self.rows {
            2 => {
                let trace = self.trace().unwrap();
                let val_pos =
                    (trace + ((trace * trace) - (4.0 * self.determinant().unwrap())).sqrt()) / 2.0;
                let val_neg =
                    (trace - ((trace * trace) - (4.0 * self.determinant().unwrap())).sqrt()) / 2.0;

                let vec_pos = self.clone() - Matrix::identity(self.rows).unwrap() * val_pos;
                let vec_neg = self.clone() - Matrix::identity(self.rows).unwrap() * val_neg;
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

        let mut upper  = vec![];
        let mut lower  = vec![];
        let mut diagonal = vec![];
        for i in 0..self.rows {
            for j in 0..self.cols {
                if i == j {
                    diagonal.push(self.get(i, j).unwrap());
                } else if i < j {
                    upper.push(self.get(i, j).unwrap());
                } else {
                    lower.push(self.get(i, j).unwrap());
                }
            }
        }

        let not_diagonal = diagonal.contains(&0.0);
        let not_upper = upper.contains(&0.0);
        let not_lower = lower.contains(&0.0);


        self.triangular = match (not_diagonal, not_upper, not_lower) {
            (false, false, true) => Triangular::Upper,
            (false, true, false) => Triangular::Lower,
            _ => Triangular::Not,
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

        let mut result = Matrix::new_empty(self.rows, rhs.cols).unwrap();
        for i in 0..self.rows {
            for j in 0..rhs.cols {
                let mut sum = 0.0;
                for k in 0..self.cols {
                    sum += self.get(i, k).unwrap() * rhs.get(k, j).unwrap();
                }
                result.set(i, j, sum).unwrap();
            }
        }
        result
    }
}
impl std::ops::Div<f64> for Matrix {
    type Output = Matrix;
    fn div(self, rhs: f64) -> Matrix {
        self * (1.0 / rhs)
    }
}
impl std::ops::DivAssign<f64> for Matrix {
    fn div_assign(&mut self, rhs: f64) {
        *self = self.clone() / rhs;
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
impl std::cmp::PartialEq for Matrix {
    fn eq(&self, other: &Matrix) -> bool {
        if self.rows != other.rows || self.cols != other.cols {
            return false;
        }
        self.data
            .iter()
            .zip(other.data.iter())
            .all(|(x, y)| (x - y).abs() < 1e-10)
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
