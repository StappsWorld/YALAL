use crate::vector::VectorN;

pub struct Line {
    p1: VectorN,
    p2: VectorN,
    midpoint: VectorN,
    direction: VectorN,
    as_vector: VectorN,
}
impl Line {
    pub fn new(p1: VectorN, p2: VectorN) -> Line {
        let mut p1 = p1;
        let mut p2 = p2;
        if p1.size() != p2.size() {
            let largest = std::cmp::max(p1.size(), p2.size());
            p1.resize(largest);
            p2.resize(largest);
        }
        let mut l = Line {
            p1,
            p2,
            midpoint: VectorN::default(),
            direction: VectorN::default(),
            as_vector: VectorN::default(),
        };
        l.update_midpoint();
        l.update_direction();
        l.update_as_vector();
        l
    }

    pub fn from_p1(direction: VectorN, p1: VectorN) -> Line {
        let mut l = Line {
            p1: p1.clone(),
            p2: direction.clone() + p1,
            midpoint: VectorN::default(),
            direction,
            as_vector: VectorN::default(),
        };
        l.update_midpoint();
        l.update_as_vector();
        l
    }

    pub fn from_p2(direction: VectorN, p2: VectorN) -> Line {
        let mut l = Line {
            p1: p2.clone() - direction.clone(),
            p2,
            midpoint: VectorN::default(),
            direction,
            as_vector: VectorN::default(),
        };
        l.update_midpoint();
        l.update_as_vector();
        l
    }

    pub fn p1(&self) -> &VectorN {
        &self.p1
    }

    pub fn p2(&self) -> &VectorN {
        &self.p2
    }

    pub fn length(&self) -> f64 {
        self.as_vector.mag()
    }

    fn update_midpoint(&mut self) {
        self.midpoint = (self.p1.clone() + self.p2.clone()) / 2.0f64;
    }

    pub fn midpoint(&self) -> &VectorN {
        &self.midpoint
    }

    pub fn normal(&self) -> VectorN {
        let mut v = self.as_vector.clone();
        v.normalize();
        v
    }

    pub fn direction(&self) -> &VectorN {
        &self.direction
    }

    fn update_direction(&mut self) {
        let mut p = self.p2.clone() - self.p1.clone();
        p.normalize();
        self.direction = p;
    }

    pub fn as_vector(&self) -> &VectorN {
        &self.as_vector
    }

    fn update_as_vector(&mut self) {
        self.as_vector = self.p2.clone() - self.p1.clone();
    }
}
